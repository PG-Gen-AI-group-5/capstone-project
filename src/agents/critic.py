"""Critic (tasks 12.7-12.8): rule checks in code, then a review of what code cannot judge.

Rule checks, each with a code:
- LOOKAHEAD         a cited record was published at or after as_of                      -> reject
- UNKNOWN_EVIDENCE  a cited ID is not in the retrieved evidence                          -> revise
- LEVELS            entry/stop/target differ from the setup, or stop >= entry >= target  -> reject
- NO_CITATION       a long plan cites no evidence                                        -> revise
- LOW_RR            risk-to-reward below `min_risk_reward`                               -> revise
- CATALYST_IN_HOLD  a results release falls inside the holding period                    -> revise
- OVERCONFIDENT     confidence above `max_confidence` (until calibration, task 15.2)     -> revise
A no-trade plan passes the checks that only matter for a trade.
"""
from __future__ import annotations

from datetime import date

from src.agents.base import AgentContext
from src.contracts import AnalystCase, Catalyst, CheckResult, CriticVerdict, Evidence, Setup, TradePlan

REJECT_CODES = {"LOOKAHEAD", "LEVELS"}


def rule_checks(plan: TradePlan, setup: Setup, evidence: list[Evidence], catalysts: list[Catalyst],
                hold_end: date, settings: dict) -> list[CheckResult]:
    by_id = {e.evidence_id: e for e in evidence}
    trade = plan.direction == "long"
    late = [i for i in plan.evidence_ids if i in by_id and by_id[i].published_at >= plan.as_of]
    unknown = [i for i in plan.evidence_ids if i not in by_id]
    levels_ok = (abs(plan.entry - setup.entry) < 0.01 and abs(plan.stop - setup.stop) < 0.01
                 and abs(plan.target - setup.target) < 0.01 and plan.stop < plan.entry < plan.target)
    results = [k for k in catalysts if k.kind == "results" and k.event_date <= hold_end]
    return [
        CheckResult(code="LOOKAHEAD", passed=not late, detail=f"published at or after as_of: {late}" if late else ""),
        CheckResult(code="UNKNOWN_EVIDENCE", passed=not unknown,
                    detail=f"not retrieved: {unknown}; cite only: {sorted(by_id)}" if unknown else ""),
        CheckResult(code="LEVELS", passed=levels_ok or not trade,
                    detail="" if levels_ok else "levels differ from the code-computed setup"),
        CheckResult(code="NO_CITATION", passed=bool(plan.evidence_ids) or not trade,
                    detail="" if plan.evidence_ids else "the plan cites no evidence"),
        CheckResult(code="LOW_RR", passed=setup.risk_reward >= settings["min_risk_reward"] or not trade,
                    detail=f"risk-to-reward {setup.risk_reward} < {settings['min_risk_reward']}"),
        CheckResult(code="CATALYST_IN_HOLD", passed=not results or not trade,
                    detail=results[0].description if results else ""),
        CheckResult(code="OVERCONFIDENT", passed=plan.confidence <= settings["max_confidence"] or not trade,
                    detail=f"confidence {plan.confidence} > {settings['max_confidence']}"),
    ]


def rule_verdict(checks: list[CheckResult]) -> CriticVerdict:
    failed = [c for c in checks if not c.passed]
    if any(c.code in REJECT_CODES for c in failed):
        verdict = "reject"
    elif failed:
        verdict = "revise"
    else:
        verdict = "accept"
    return CriticVerdict(verdict=verdict, checks=checks,
                         reasons=[f"{c.code}: {c.detail}" for c in failed] or ["all rule checks passed"])


def review(ctx: AgentContext, plan: TradePlan, setup: Setup, evidence: list[Evidence], catalysts: list[Catalyst],
           bull: AnalystCase, bear: AnalystCase) -> CriticVerdict:
    days = ctx.day.data.calendar_days_after(ctx.day.trade_date, plan.max_hold_days)
    hold_end = days[-1].date() if len(days) else ctx.day.trade_date.date()
    checks = rule_checks(plan, setup, evidence, catalysts, hold_end, ctx.settings)
    ctx.tracer.step("critic", "critique", symbol=plan.symbol, input_summary="rule checks",
                    output={"failed": [c.code for c in checks if not c.passed]})
    by_rules = rule_verdict(checks)
    variables = {"symbol": plan.symbol, "trade_date": str(ctx.day.trade_date.date()),
                 "plan": plan.model_dump(mode="json"), "checks": [c.model_dump() for c in checks],
                 "bull": bull.model_dump(mode="json"), "bear": bear.model_dump(mode="json")}
    verdict = ctx.llm.complete("critic", variables, CriticVerdict, lambda: by_rules, symbol=plan.symbol).output
    # the LLM can make a verdict stricter, never looser than the rule checks
    order = {"accept": 0, "revise": 1, "reject": 2}
    if order[verdict.verdict] < order[by_rules.verdict]:
        verdict = verdict.model_copy(update={"verdict": by_rules.verdict, "reasons": by_rules.reasons + verdict.reasons})
    return verdict.model_copy(update={"checks": checks})
