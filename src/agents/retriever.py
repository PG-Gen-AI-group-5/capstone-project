"""Retriever (task 12.2): collects the evidence published before as_of and summarises it with citations.

Evidence: the stock's announcements from the last `evidence_lookback_days` (important ones first, newest
first) and the base model forecast. Offline, each important announcement becomes a cited claim. Retrieval
over announcement text (task 8) replaces the title-level evidence once the text corpus exists.
"""
from __future__ import annotations

from src.agents.base import AgentContext
from src.contracts import Claim, Evidence, EvidenceSummary


def retrieve(ctx: AgentContext, symbol: str) -> tuple[list[Evidence], EvidenceSummary]:
    evidence = ctx.day.evidence(symbol, ctx.settings["evidence_lookback_days"])
    ctx.tracer.step("retriever", "tool", symbol=symbol, input_summary="announcements + base forecast as of 08:30",
                    output={"evidence_ids": [e.evidence_id for e in evidence]})

    def rules() -> EvidenceSummary:
        claims = [Claim(text=f"{e.published_at:%d %b %Y}: {e.title}" + (f" - {e.text[:160]}" if e.text else ""),
                        evidence_ids=[e.evidence_id])
                  for e in evidence if e.important or e.kind == "forecast"][:8]
        n_ann = sum(e.kind == "announcement" for e in evidence)
        n_imp = sum(e.important for e in evidence)
        summary = (f"{n_ann} announcement(s) in the last {ctx.settings['evidence_lookback_days']} days, "
                   f"{n_imp} marked important." if n_ann else "No announcements in the lookback window.")
        return EvidenceSummary(symbol=symbol, summary=summary, claims=claims)

    variables = {"symbol": symbol, "trade_date": str(ctx.day.trade_date.date()),
                 "evidence": [e.model_dump(mode="json") for e in evidence]}
    summary = ctx.llm.complete("retriever", variables, EvidenceSummary, rules, symbol=symbol).output
    known = {e.evidence_id for e in evidence}
    for c in summary.claims:
        c.evidence_ids = [i for i in c.evidence_ids if i in known]
    return evidence, summary
