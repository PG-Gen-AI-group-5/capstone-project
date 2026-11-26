"""Forecaster (task 12.6): turns the setup, the base model forecast and the debate into a trade plan.

Entry, stop and target are copied from the code-computed setup; whatever the LLM returns for them is
overwritten. Offline rules: edge = bull confidence - bear confidence + (base 5-day p_up - 0.5); long when the
edge reaches `min_edge` and the setup's risk-to-reward reaches `min_risk_reward`. The holding period comes
from the setup type, shortened to end before a results catalyst when the Critic flags one.
"""
from __future__ import annotations

from src.agents.base import AgentContext
from src.agents.context import citable
from src.contracts import AnalystCase, Catalyst, CriticVerdict, Evidence, Setup, TradePlan


def forecast(ctx: AgentContext, symbol: str, setup: Setup, bull: AnalystCase, bear: AnalystCase,
             catalysts: list[Catalyst], evidence: list[Evidence], feedback: CriticVerdict | None = None) -> TradePlan:
    s = ctx.settings
    fc = ctx.day.base_forecast(symbol)
    p5 = fc.p_up.get(5) if fc else None
    vol = fc.vol_forecast.get(5) if fc else None
    failed = {c.code for c in feedback.checks if not c.passed} if feedback else set()

    def rules() -> TradePlan:
        edge = bull.confidence - bear.confidence + ((p5 - 0.5) if p5 is not None else 0.0)
        hold = s["hold_days"].get(setup.setup_type, 5)
        results = sorted(k.event_date for k in catalysts if k.kind == "results")
        if "CATALYST_IN_HOLD" in failed and results:
            days = ctx.day.data.calendar_days_after(ctx.day.trade_date, 11)
            before = [d for d in days if d.date() < results[0]]
            hold = max(1, len(before) - 1)
        confidence = min(max(0.5 + edge / 2, 0.0), 1.0)
        if "OVERCONFIDENT" in failed:
            confidence = min(confidence, s["max_confidence"])
        go = edge >= s["min_edge"] and setup.risk_reward >= s["min_risk_reward"] and "LOW_RR" not in failed
        if "CATALYST_IN_HOLD" in failed and hold < 2:
            go = False
        known = {e.evidence_id for e in evidence}
        ids = sorted({i for c in bull.claims + bear.claims for i in c.evidence_ids if i in known})
        why = (f"edge {edge:+.2f} (bull {bull.confidence:.2f}, bear {bear.confidence:.2f}"
               + (f", base p_up {p5:.2f}" if p5 is not None else "") + f"); setup {setup.setup_type}, "
               f"risk-to-reward {setup.risk_reward}")
        return TradePlan(symbol=symbol, as_of=ctx.day.as_of, direction="long" if go else "no_trade",
                         entry=setup.entry, stop=setup.stop, target=setup.target, max_hold_days=min(max(hold, 1), 10),
                         vol_forecast=vol if vol is not None else round(setup.atr_pct / 1.6, 4),
                         confidence=round(confidence, 3), rationale=why, evidence_ids=ids,
                         setup_type=setup.setup_type, base_forecast_p_up=p5)

    variables = {"symbol": symbol, "trade_date": str(ctx.day.trade_date.date()), "as_of": ctx.day.as_of.isoformat(),
                 "setup": setup.model_dump(mode="json"), "base_forecast": fc.model_dump(mode="json") if fc else None,
                 "bull": bull.model_dump(mode="json"), "bear": bear.model_dump(mode="json"),
                 "evidence_ids": citable(evidence), "feedback": feedback.model_dump(mode="json") if feedback else ""}
    plan = ctx.llm.complete("forecaster", variables, TradePlan, rules, symbol=symbol).output
    # prices come from code, whatever the model returned (task 10.4)
    return plan.model_copy(update={"entry": setup.entry, "stop": setup.stop, "target": setup.target,
                                   "as_of": ctx.day.as_of, "setup_type": setup.setup_type,
                                   "base_forecast_p_up": p5})
