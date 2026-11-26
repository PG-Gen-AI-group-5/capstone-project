"""Dashboard (task 19): reads the files the pipeline writes; the sidebar can start an agent run.

    streamlit run src/dashboard/app.py

Pages: trade plans (19.2), trade journal (19.3), equity and metrics vs baselines (19.4), agent trace (19.6),
model metrics, data and pipeline status (19.9).
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.tracing.writer import read_trace

# categorical palette shared with report figures (src/analysis/plots.py)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7b5cd6", "#8a8a8a"]

st.set_page_config(page_title="NSE swing-trade agents", layout="wide")
cfg = Config.load()
RUNS = cfg.data_dir / "runs"
BACKTEST = cfg.derived_dir / "backtest"


@st.cache_data(ttl=60)
def list_runs() -> list[str]:
    return sorted((p.name for p in RUNS.glob("*") if (p / "manifest.json").exists()), reverse=True) if RUNS.exists() else []


@st.cache_data(ttl=60)
def load_plans(run_id: str) -> pd.DataFrame:
    path = RUNS / run_id / "plans.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def load_backtests() -> dict[str, dict]:
    out = {}
    for folder in sorted(BACKTEST.glob("*")) if BACKTEST.exists() else []:
        if (folder / "equity.parquet").exists():
            out[folder.name] = {
                "equity": pd.read_parquet(folder / "equity.parquet"),
                "journal": pd.read_parquet(folder / "journal.parquet") if (folder / "journal.parquet").exists() else None,
                "summary": json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                if (folder / "summary.json").exists() else {},
            }
    return out


# ---------------------------------------------------------------- sidebar: choose a run, start a run
st.sidebar.title("NSE swing-trade agents")
runs = list_runs()
run_id = st.sidebar.selectbox("Agent run", runs) if runs else None
with st.sidebar.expander("Run the agents", expanded=not runs):
    day = st.date_input("Forecast day", value=date(2023, 6, 1))
    syms = st.text_input("Stocks (blank = all Nifty 50)", "")
    if st.button("Run"):
        from src.run import run_days
        symbols = [s.strip().upper() for s in syms.split(",") if s.strip()] or None
        with st.spinner(f"Running the agents for {day}..."):
            folder = run_days(cfg, day, day, run_id=f"ui-{day:%Y%m%d}", symbols=symbols, log=lambda *_: None)
        st.cache_data.clear()
        st.success(f"Run written to {folder.name}")
llm = cfg.get("llm") or {}
provider = llm.get("provider", "offline")
if provider == "openai_compatible":
    from src.llm.openai_compatible import endpoint_settings
    model = endpoint_settings(llm)["model"]
else:
    model = llm.get("model")
st.sidebar.caption(f"LLM provider: {provider}" + (f" ({model})" if provider != "offline" else " (rule-based stand-ins)"))

page = st.sidebar.radio("Page", ["Trade plans", "Trade journal", "Equity vs baselines", "Agent trace",
                                 "Model metrics", "Data and pipeline"])

# ---------------------------------------------------------------- pages
if page == "Trade plans":
    st.header("Trade plans")
    if not run_id:
        st.info("No agent runs yet. Start one from the sidebar, or run `python -m src.run --date 2023-06-01`.")
    else:
        plans = load_plans(run_id)
        if plans.empty:
            st.info("This run has no plans.")
        else:
            days = sorted(plans["trade_date"].unique(), reverse=True)
            d = st.selectbox("Day", days)
            today = plans[plans["trade_date"] == d]
            c1, c2, c3 = st.columns(3)
            c1.metric("Stocks analysed", len(today))
            c2.metric("Accepted trades", int(today["accepted"].sum()))
            c3.metric("Rejected or no trade", int((~today["accepted"]).sum()))
            for r in today.sort_values("accepted", ascending=False).itertuples():
                status = "ACCEPTED" if r.accepted else f"{r.verdict.upper()} / {r.direction}"
                with st.container(border=True):
                    st.subheader(f"{r.symbol} - {status}")
                    a, b, c, e = st.columns(4)
                    a.metric("Entry (ref close)", f"{r.entry:,.2f}")
                    b.metric("Stop", f"{r.stop:,.2f}", f"{r.stop / r.entry - 1:.1%}")
                    c.metric("Target", f"{r.target:,.2f}", f"{r.target / r.entry - 1:.1%}")
                    e.metric("Confidence", f"{r.confidence:.0%}", f"hold {r.max_hold_days} days", delta_color="off")
                    st.write(r.rationale)
                    if r.failed_checks:
                        st.warning("Critic: " + ", ".join(r.failed_checks))
                    if r.evidence_ids:
                        st.caption("Evidence: " + ", ".join(r.evidence_ids))

elif page == "Trade journal":
    st.header("Trade journal")
    bts = load_backtests()
    if not bts:
        st.info("No backtests yet: `python -m src.pipeline run backtest`.")
    else:
        src = st.selectbox("Strategy", [k for k, v in bts.items() if v["journal"] is not None])
        j = bts[src]["journal"]
        f1, f2 = st.columns(2)
        symbol = f1.multiselect("Stock", sorted(j["symbol"].unique()))
        outcome = f2.multiselect("Exit", sorted(j["exit_reason"].unique()))
        keep = pd.Series(True, index=j.index)
        if symbol:
            keep &= j["symbol"].isin(symbol)
        if outcome:
            keep &= j["exit_reason"].isin(outcome)
        view = j[keep]
        st.dataframe(view, width="stretch", hide_index=True)

elif page == "Equity vs baselines":
    st.header("Equity curves and metrics")
    bts = load_backtests()
    if not bts:
        st.info("No backtests yet: `python -m src.pipeline run backtest`.")
    else:
        chosen = st.multiselect("Strategies", list(bts), default=list(bts))
        curves = pd.concat([bts[k]["equity"].assign(strategy=k) for k in chosen], ignore_index=True)
        fig = px.line(curves, x="trade_date", y="equity", color="strategy", color_discrete_sequence=SERIES)
        fig.update_layout(legend={"orientation": "h", "y": -0.2}, yaxis_title="equity (Rs)", xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        table = pd.DataFrame({k: bts[k]["summary"] for k in chosen}).T
        st.dataframe(table.drop(columns=["exits"], errors="ignore"), width="stretch")

elif page == "Agent trace":
    st.header("Agent trace")
    if not run_id:
        st.info("No agent runs yet.")
    else:
        trace = pd.DataFrame(read_trace(RUNS / run_id / "trace.jsonl"))
        if trace.empty:
            st.info("This run has no trace.")
        else:
            symbol = st.selectbox("Stock", ["(all)"] + sorted(trace["symbol"].dropna().unique()))
            view = trace if symbol == "(all)" else trace[trace["symbol"] == symbol]
            st.caption(f"{len(view)} steps; LLM cost ${view['cost_usd'].sum():.4f}")
            for r in view.itertuples():
                with st.expander(f"{r.step}. {r.agent} - {r.kind}" + (f" - {r.symbol}" if r.symbol else "")):
                    st.json(r.output)

elif page == "Model metrics":
    st.header("Model forecasts")
    folder = cfg.derived_dir / "metrics"
    periods = sorted(p.stem for p in folder.glob("*.parquet") if "_" not in p.stem) if folder.exists() else []
    if not periods:
        st.info("No metrics yet: `python -m src.pipeline run models`.")
    else:
        period = st.selectbox("Period", periods)
        m = pd.read_parquet(folder / f"{period}.parquet")
        ref = folder / f"{period}_majority.parquet"
        st.dataframe(m, width="stretch", hide_index=True)
        if ref.exists():
            st.caption("Constant majority guess for the same days (hindsight reference)")
            st.dataframe(pd.read_parquet(ref), hide_index=True)

else:
    st.header("Data and pipeline")
    from src.pipeline.runner import status
    st.dataframe(status(cfg), width="stretch", hide_index=True)
    import sqlite3
    if cfg.tracker_db.exists():
        con = sqlite3.connect(cfg.tracker_db)
        try:
            tasks = pd.read_sql("SELECT source, status, COUNT(*) AS tasks FROM tasks GROUP BY source, status", con)
            st.subheader("Download tracker")
            st.dataframe(tasks.pivot(index="source", columns="status", values="tasks").fillna(0).astype(int),
                         width="stretch")
        finally:
            con.close()
