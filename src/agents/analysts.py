"""Bull and Bear analysts (tasks 12.3-12.5): opening cases, then one rebuttal round each.

Offline rules turn the day's facts into cited claims. Bull: setup, trend, relative strength, a base forecast
above 50%, a recent results release. Bear: high India VIX, the Nifty below its 200-day EMA, an overbought RSI,
a base forecast below 50%, results due inside the holding period, high volatility.
Confidence: 0.5 plus 0.07 per claim (capped at 0.8), moved by the base forecast's distance from 50%.
"""
from __future__ import annotations

from src.agents.base import AgentContext
from src.agents.context import citable, context_id, setup_id
from src.contracts import AnalystCase, Catalyst, Claim, Evidence, EvidenceSummary, Setup


def _facts(ctx: AgentContext, symbol: str, setup: Setup | None, catalysts: list[Catalyst]) -> tuple[list, list, float]:
    c = ctx.day.context(symbol)
    fc = ctx.day.base_forecast(symbol)
    p5 = fc.p_up.get(5) if fc else None
    fid = [f"forecast:{fc.model}:{ctx.day.trade_date.date()}"] if fc else []
    cid, sid = [context_id(symbol, ctx.day.trade_date)], [setup_id(symbol, ctx.day.trade_date)]
    bull, bear = [], []
    if setup and setup.setup_type != "none":
        bull.append(Claim(text=f"code-detected {setup.setup_type} setup, risk-to-reward {setup.risk_reward}",
                          evidence_ids=sid))
    if (c.get("ema200_gap") or 0) > 0 and (c.get("ema20_50") or 0) > 0:
        bull.append(Claim(text="price above its 200-day EMA and EMA20 above EMA50 (uptrend)", evidence_ids=cid))
    if (c.get("rs_nifty_20") or 0) > 0.02:
        bull.append(Claim(text=f"outperformed the Nifty by {c['rs_nifty_20']:.1%} over 20 days", evidence_ids=cid))
    if p5 is not None and p5 > 0.5:
        bull.append(Claim(text=f"{fc.model} gives a 5-day rise probability of {p5:.2f}", evidence_ids=fid))
    if (c.get("days_since_results") or 99) <= 3:
        bull.append(Claim(text="results released in the last 3 trading days; the event is behind, not ahead"))
    if (c.get("vix") or 0) > ctx.settings["high_vix"]:
        bear.append(Claim(text=f"India VIX at {c['vix'] * 100:.1f}: a volatile market widens gaps through stops",
                          evidence_ids=cid))
    if (c.get("nifty_ema200_gap") or 0) < 0:
        bear.append(Claim(text="the Nifty is below its 200-day EMA (market downtrend)", evidence_ids=cid))
    if (c.get("rsi14") or 0) > 0.70:
        bear.append(Claim(text=f"RSI {c['rsi14'] * 100:.0f}: overbought, prone to pull back", evidence_ids=cid))
    if p5 is not None and p5 < 0.5:
        bear.append(Claim(text=f"{fc.model} gives a 5-day rise probability of only {p5:.2f}", evidence_ids=fid))
    results = [k for k in catalysts if k.kind == "results"]
    if results:
        bear.append(Claim(text=f"results catalyst inside the holding window: {results[0].description}"))
    if (c.get("atr14_pct") or 0) > 0.035:
        bear.append(Claim(text=f"high daily range (ATR {c['atr14_pct']:.1%} of price) makes the stop easy to hit",
                          evidence_ids=cid))
    return bull, bear, (p5 if p5 is not None else 0.5)


def _cite_events(summary: EvidenceSummary, claims: list[Claim]) -> list[Claim]:
    return claims + [cl for cl in summary.claims if cl.evidence_ids and not cl.evidence_ids[0].startswith("forecast")][:3]


def _confidence(n: int, p5: float, sign: int) -> float:
    return round(min(0.8, max(0.2, 0.5 + 0.07 * n - 0.07 + sign * (p5 - 0.5))), 3)


def opening_cases(ctx: AgentContext, symbol: str, setup: Setup | None, summary: EvidenceSummary,
                  catalysts: list[Catalyst], evidence: list[Evidence]) -> tuple[AnalystCase, AnalystCase]:
    bull_claims, bear_claims, p5 = _facts(ctx, symbol, setup, catalysts)
    variables = {"symbol": symbol, "trade_date": str(ctx.day.trade_date.date()),
                 "setup": setup.model_dump(mode="json") if setup else None, "context": ctx.day.context(symbol),
                 "evidence_summary": summary.model_dump(mode="json"), "evidence_ids": citable(evidence),
                 "opponent": ""}
    bull = ctx.llm.complete("bull", variables, AnalystCase, lambda: AnalystCase(
        side="bull", symbol=symbol, thesis="; ".join(c.text for c in bull_claims[:3]) or "no clear case for the trade",
        claims=_cite_events(summary, bull_claims), confidence=_confidence(len(bull_claims), p5, +1)), symbol).output
    bear = ctx.llm.complete("bear", variables, AnalystCase, lambda: AnalystCase(
        side="bear", symbol=symbol, thesis="; ".join(c.text for c in bear_claims[:3]) or "no clear case against",
        claims=bear_claims, confidence=_confidence(len(bear_claims), p5, -1)), symbol).output
    return bull, bear


def rebuttals(ctx: AgentContext, symbol: str, setup: Setup | None, summary: EvidenceSummary, bull: AnalystCase,
              bear: AnalystCase, evidence: list[Evidence]) -> tuple[AnalystCase, AnalystCase]:
    base = {"symbol": symbol, "trade_date": str(ctx.day.trade_date.date()),
            "setup": setup.model_dump(mode="json") if setup else None, "context": ctx.day.context(symbol),
            "evidence_summary": summary.model_dump(mode="json"), "evidence_ids": citable(evidence)}

    def answer(case: AnalystCase, other: AnalystCase) -> AnalystCase:
        text = (f"{len(other.claims)} opposing claim(s) noted; the {case.side} case keeps "
                f"{len(case.claims)} claim(s) and confidence {case.confidence:.2f}.")
        return case.model_copy(update={"rebuttal": text})

    bull2 = ctx.llm.complete("bull", {**base, "opponent": bear.model_dump(mode="json")}, AnalystCase,
                             lambda: answer(bull, bear), symbol).output
    bear2 = ctx.llm.complete("bear", {**base, "opponent": bull.model_dump(mode="json")}, AnalystCase,
                             lambda: answer(bear, bull), symbol).output
    ctx.tracer.step("debate", "debate", symbol=symbol, output={"bull_confidence": bull2.confidence,
                                                               "bear_confidence": bear2.confidence})
    return bull2, bear2
