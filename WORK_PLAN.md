# Work plan: Agentic Swing-Trade Intelligence (NSE)

The complete work, divided into 20 sections of small tasks, then grouped into milestones in build order.
Scope comes from `Agentic Swing-Trade Intelligence for NSE Stocks.pdf` and the original capstone brief. Design ideas are borrowed from the reference project `D:\Projects\trading_with_ai`.
How to collect and store each dataset: [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md).

---

## Assumptions and decisions

| Topic | Decision |
|---|---|
| Universe | **Two universes.** Nifty 50 for the agents, text data and all evaluation. Nifty 200 prices (optional) for training price models and setup statistics. Every comparison is made on the same 50 stocks and days |
| Stock list | Current constituents (survivorship bias noted in the report) |
| Horizon | Trade plans held 3–10 trading days. Next-day and next-week (5-day) direction accuracy are also reported, as the original brief asks |
| Forecast time | 08:30 IST on trading day `d`; entry at the 09:15 open of day `d` |
| Periods | Train 2016–2022, validate 2023, test 2024 to today. Agent results are scored only on dates after the LLM's training cutoff |
| Trade side | Long-only in the backtest (cash-market shorts can't be held overnight); bearish calls still scored for direction |
| Data sources | NSE for prices, announcements, results and flows (BSE and Stooq block scripted access); yfinance as price backup; FRED; RBI (entered by hand); GDELT as a secondary news source |
| Storage | Raw files → Parquet lake → DuckDB for analysis; SQLite tracker for every download |
| Stack | Python 3.12, LangGraph, Parquet + DuckDB, SQLite, a vector store, Streamlit, pytest, Docker |

**Time rule used everywhere:** a record may be used only if `published_at < as_of`. Times are stored in UTC and shown in IST.

---

## Part A: Work breakdown

### 1. Project setup

| ID | Task | Done when |
|---|---|---|
| 1.1 | Repo layout (`src/`, `tests/`, `docs/`, `data/`), `pyproject.toml`, pinned dependencies | `pip install -e .` works on a fresh machine |
| 1.2 | `.gitignore` (all of `data/` except `data/manual/`, cache, `.env`), `.env.example` | No keys or bulk data can be committed by accident |
| 1.3 | CI: ruff + pytest on every pull request; protect `main` | A failing test blocks merge |
| 1.4 | `config.yaml` + loader: paths, universes, dates, costs, LLM settings | Every module reads settings from one place |
| 1.5 | Time utilities: IST↔UTC, NSE trading calendar and holidays, `as_of` for a date | Tests for holidays, weekends, time zones pass |
| 1.6 | Shared data formats in `src/contracts.py` (pydantic): prices, announcements, chunks, news, flows, macro, catalysts, evidence, analyst case, trade plan, critic verdict, trace step, trade record, metric | Reviewed and approved by the whole team |
| 1.7 | Sample data in `tests/fixtures/`: 5 stocks × 6 months prices, 30 announcements, FII/DII, macro, news rows | Every later module can be tested without real data |
| 1.8 | Logging setup (structured, per run) | One log format across modules |
| 1.9 | Data-source access check. **First pass done 30 Oct 2026**; results and open items in `docs/DATA_COLLECTION.md` | Remaining open items closed in the pilot (1.11) |
| 1.10 | Choose the LLM and record its training cutoff | Cutoff date in `config.yaml`; clean test window fixed |
| 1.11 | **5-stock pilot** (RELIANCE, HDFCBANK, INFY, TCS, LT): first versions of every collector, run through the tracker | Open items in `docs/DATA_COLLECTION.md` answered; real volumes, scanned-PDF share, GDELT match precision and setup frequency measured |
| 1.12 | Decide final scope from the pilot: trade-count check (section C), DII and bulk deals in or out, GDELT history route, Nifty 200 prices yes or no | Decisions recorded in this file |

### 2. Data: prices

| ID | Task | Done when |
|---|---|---|
| 2.1 | Universe file `data/manual/universe.csv` from NSE's Nifty 50 list: symbol, ISIN, sector, aliases, `data_start`, `old_symbols` (ZOMATO → ETERNAL, TATAMOTORS → TMPV), tier (nifty50 / nifty200) | One file used by every module |
| 2.2 | NSE bhavcopy downloader, 2016 to today, one task per trading day; two file formats (old up to 5 Jul 2024, new from 8 Jul 2024) | Daily OHLCV for all 50 stocks in Parquet |
| 2.3 | yfinance fallback for missing days (Stooq blocks scripted downloads) | Gaps filled, `source` recorded |
| 2.4 | Corporate actions from the NSE API: parse splits, bonuses, dividends, demergers; adjustment factors; hand-check every split, bonus and demerger (including TMPV) | Adjusted series has no jumps on corporate-action days |
| 2.5 | Nifty 50 and India VIX from NSE's daily index file (same loop as 2.2) | Benchmark and regime series ready |
| 2.6 | Price quality checks: missing days vs calendar, duplicates, zero volume, unexplained jumps | Quality report with zero unexplained errors |
| 2.7 | (Optional) Nifty 200 prices for training: keep 200 symbols from the same bhavcopy files; corporate actions for the extra 150 | Training price table for 200 stocks |

### 3. Data: corporate announcements and filings

| ID | Task | Done when |
|---|---|---|
| 3.1 | NSE announcements list, one task per stock per year, 2016 → today (BSE's API blocks scripted access, so NSE only) | Announcements with category, summary text, timestamp, attachment link |
| 3.2 | Tag important vs routine (category + keywords in the summary text; most items are routine "Updates") | Every announcement has `is_important` |
| 3.3 | Download PDFs for important announcements, Oct 2022 → today; extract text; flag scanned PDFs | Clean text stored for each important announcement |
| 3.4 | Quarterly results filing list with dissemination times, 2015 → today | Results dates and public times for all 50 stocks |
| 3.5 | Results numbers (revenue, profit, EPS): find the detailed or XBRL source, and where results from 2025 onward are published (the results list stops at the Dec 2024 quarter); YoY/QoQ growth as the "results surprise" proxy | Results table for all 50 stocks |
| 3.6 | Board meetings (results dates known in advance) | Table of meetings with announcement times |
| 3.7 | Insider trades, promoter pledging, shareholding | Tables with dates for each stock |
| 3.8 | Bulk deals: collect daily from now on (history endpoint blocked), or leave out | Decision made in 1.12 |
| 3.9 | Exchange dissemination timestamps converted IST → UTC | All announcement times pass the time-rule tests |

### 4. Data: market flows and macro

| ID | Task | Done when |
|---|---|---|
| 4.1 | FII/DII daily collection starting now (NSE gives only the latest day, so history must be built day by day) | Daily job running |
| 4.2 | FII history from NSDL; find a DII history source or drop DII history | History table, or a recorded decision |
| 4.3 | RBI policy decisions entered by hand in `data/manual/rbi_policy.csv` (announced around 10:00 IST, so usable from the next day) | ~65 decisions with dates and rates |
| 4.4 | FRED series: DGS10, VIXCLS (released daily); DEXINUS, DTWEXBGS, DCOILBRENTEU (released weekly) | Daily series in Parquet |
| 4.5 | `known_from` for every macro value: next day for market prices (all five FRED series); first-release date for any revisable economic release added later. Live job: daily source for USD/INR and Brent (FRED publishes them weekly) | Macro values pass the time-rule tests |

### 5. Data: news (GDELT, secondary source)

NSE announcements are the main evidence for the agents. GDELT adds headlines, news volume, tone, and sector and macro news.

| ID | Task | Done when |
|---|---|---|
| 5.1 | Choose the history route: raw 15-minute files (free, ~141,000 files ≈ 760 GB to stream and filter for Oct 2022 → today) or BigQuery (free up to 1 TB scanned a month) | Decision made in 1.12 |
| 5.2 | Collect history Oct 2022 → today, keeping only rows that mention the 50 companies | News rows: time, source, URL, title, organisations, tone, themes |
| 5.3 | Company matching with alias lists, plus a relevance score (company in the headline, number of mentions), so market summaries and passing mentions score low | Each row has a match and relevance score |
| 5.4 | Hand-check 100 matched rows (for example Reliance Industries vs Reliance Power) | Precision measured and recorded |
| 5.5 | Daily news features per stock: count, count z-score, average tone, tone change | Feature table for the analysts and ablations |

### 6. Storage, download tracker and point-in-time store

| ID | Task | Done when |
|---|---|---|
| 6.1 | Storage layout: `data/raw/` (as downloaded) → `data/lake/` (Parquet) → `data/derived/`; `data/manual/` for hand-made CSVs | Documented in `docs/DATA_COLLECTION.md` |
| 6.2 | Download tracker in SQLite (`data/ingest.sqlite`): sources, error types, runs, tasks, attempts, review items, trading calendar | Schema created; unit tests for status changes |
| 6.3 | Tracker rules: an error never stops a run; retryable errors retry 3 times with growing waits; errors that retrying can't fix go straight to review; a 404 on a holiday is "no data" | Tests for each rule |
| 6.4 | Tracker commands: `plan`, `run`, `status`, `review`, `resolve` | A failed day can be skipped, reviewed and re-run later |
| 6.5 | DuckDB views over the Parquet tables | One query interface for all data |
| 6.6 | `as_of(symbol, t)` functions for prices, announcements, news, flows, macro | Each returns only records public before `t` |
| 6.7 | Leakage tests: plant future records, confirm they are never returned | Tests in CI |
| 6.8 | Coverage table: one row per stock per trading day per source; missing expected data creates a review item | Gaps visible per stock per day |
| 6.9 | Daily update job (new prices, index, announcements, FII/DII, news) using the tracker | One command brings data up to today |
| 6.10 | Data-quality report across all sources | Generated after every update |

### 7. Catalyst calendar

| ID | Task | Done when |
|---|---|---|
| 7.1 | Results dates from board-meeting announcements (3.6) | Upcoming results date for each stock |
| 7.2 | Market-wide events: RBI MPC (4.3), Union Budget, F&O monthly expiry, NSE holidays, US FOMC and CPI | Market calendar table |
| 7.3 | Stock events: ex-dividend, record date (from 2.4), AGM | Stock calendar table |
| 7.4 | Query: "catalysts for stock X between date A and B" | Used by Planner, Critic and Exit Monitor |

### 8. Retrieval

| ID | Task | Done when |
|---|---|---|
| 8.1 | Chunk announcement text (and news headlines; GDELT has no article text) with chunk IDs and character offsets | Every chunk can be traced to its source |
| 8.2 | Pick the embedding model; embed all chunks | Embeddings stored, cached |
| 8.3 | Vector index with metadata filters (symbol, `published_at < as_of`) | Search never returns future chunks (test) |
| 8.4 | Hybrid search: keyword (BM25) + embeddings, then rerank | Better results than either alone on 8.6 |
| 8.5 | Citation checker: every quote must be an exact substring of its chunk | Fake quotes are flagged by code |
| 8.6 | Retrieval test set: 20 hand-picked events with the documents that should be found | Recall@5 measured |

### 9. Event impact memory

| ID | Task | Done when |
|---|---|---|
| 9.1 | Build event records from the announcement list (2016 → today) and prices: stock, date, type, market regime, return over 1/3/5/10 days vs Nifty | Table of past events with reactions |
| 9.2 | "Find similar past events" search, limited to events before `as_of` | Returns top matches with their reactions |
| 9.3 | Summary for analysts: "similar events moved the stock X% on average, N cases" | Ready to pass into Bull/Bear prompts |

### 10. Technical features and swing-setup detector

| ID | Task | Done when |
|---|---|---|
| 10.1 | Daily indicators: EMA 20/50/200, RSI, ATR, Bollinger bands, volume z-score, distance from 52-week high, strength vs Nifty | Feature table for all stocks in the price universe (start from the reference `indicator_service.py`) |
| 10.2 | Market regime: trend vs range, high vs low volatility (India VIX), FII flow trend | Regime label for each day |
| 10.3 | Setup rules: pullback to EMA in uptrend, breakout from base, gap-and-hold, oversold bounce | Each day/stock gets a setup type or none (start from the reference `pullback_service.py`) |
| 10.4 | Entry, stop-loss (ATR-based) and target candidates with risk-to-reward | Numbers come from code, not the LLM |
| 10.5 | Test the setups alone on history (on Nifty 200 if collected): hit rate and average return per setup type; setups per stock per month | Shows whether the setups carry any edge; feeds the trade-count check |

### 11. LLM layer

| ID | Task | Done when |
|---|---|---|
| 11.1 | LLM client: provider can be swapped, temperature 0, JSON output checked with pydantic, retries, timeouts | All agents call the LLM through it |
| 11.2 | Response cache keyed by prompt + model + settings | Re-runs cost nothing |
| 11.3 | Token, cost and latency log; daily budget limit | Cost per run is known |
| 11.4 | Prompt files with version numbers | Every output records the prompt version used |

### 12. Agents

Each agent needs: prompt, input/output schema, tools, unit tests on sample data.

| ID | Task | Done when |
|---|---|---|
| 12.1 | **Planner**: picks stocks for full analysis (setup present, event or catalyst nearby), skips the rest, sets the steps | Daily task list with reasons |
| 12.2 | **Retriever**: collects and summarises announcements (main evidence), news headlines and macro data before `as_of`, with citations | Cited summary; all quotes pass 8.5 |
| 12.3 | **Bull Analyst**: case for the trade (sentiment, results surprise, event type, regime, similar past events) | Structured case with cited claims |
| 12.4 | **Bear Analyst**: case against the trade, same fields | Structured case with cited claims |
| 12.5 | Debate protocol: opening cases, one rebuttal round each | Both sides respond to each other's claims |
| 12.6 | **Forecaster**: combines setup + debate into a trade plan: direction, entry, stop, target, holding period, volatility forecast, confidence, rationale | Valid trade plan; prices stay inside the limits set by code |
| 12.7 | **Critic** rule checks: look-ahead (evidence dated after `as_of`), missing citations, risk-to-reward too low, catalyst inside the holding period, overconfidence vs past calibration | Each check has a code and a test |
| 12.8 | **Critic** LLM review: weak evidence, contradictions; verdict accept / revise / reject | Verdict with reasons |
| 12.9 | **Exit Monitor**: daily re-check of open trades: stop/target hit, new adverse event, catalyst ahead → hold / tighten stop / exit | Exit decisions with reasons |

### 13. Orchestration (LangGraph)

| ID | Task | Done when |
|---|---|---|
| 13.1 | Graph state object | Holds everything passed between agents |
| 13.2 | Graph with all 7 agents as stubs | Runs end to end on sample data |
| 13.3 | Routing: Planner skip, Critic revise loop (max 2 rounds), reject path | Each path covered by a test |
| 13.4 | Daily run command: `python -m src.run --date D` for the Nifty 50 | Produces trade plans for one day |
| 13.5 | Historical runner over a date range, with caching and resume after a crash | Can run months of history unattended |
| 13.6 | Open-trades state (positions carried day to day) for the Exit Monitor | Positions open and close correctly across days |

### 14. Reasoning-trace logging

| ID | Task | Done when |
|---|---|---|
| 14.1 | Trace step format and JSON-lines writer | One file per run |
| 14.2 | Record every agent message, tool call, debate turn, critique and revision | A full run can be replayed from the trace |
| 14.3 | Run manifest: git commit, config hash, prompt versions, LLM model and cutoff | Every result can be reproduced |
| 14.4 | Helper functions to load and filter traces | Used by the dashboard |

### 15. Extra features

| ID | Task | Done when |
|---|---|---|
| 15.1 | What-if simulator: add, remove or change an event, re-run the graph, show the difference in the plan | Two plans shown side by side with the changed reasoning |
| 15.2 | Confidence calibration: reliability curve, fit on validation, apply to test | Calibrated confidence stored with each plan |

(The catalyst calendar is section 7 and the event impact memory is section 9.)

### 16. Labels and baselines

| ID | Task | Done when |
|---|---|---|
| 16.1 | Labels: direction over the next 1 day and 5 days (the brief's next-day / next-week), and 10 days; realised volatility over the holding period | Label table, no future data in features |
| 16.2 | Walk-forward splitter with a gap equal to the label horizon | Tests show no overlap between train and test |
| 16.3 | Buy-and-hold (Nifty 50 index and each stock) | Results in the common format |
| 16.4 | Moving-average crossover | Results in the common format |
| 16.5 | Historical-volatility baseline (for the volatility error comparison) | Results in the common format |
| 16.6 | ARIMA | Results in the common format |
| 16.7 | LSTM: one shared model across stocks (trained on Nifty 200 if collected), evaluated on the Nifty 50 | Results in the common format |
| 16.8 | Turn baseline outputs into trade signals the backtester accepts | All baselines run through the same backtester |
| 16.9 | **Freeze baseline results** | Tagged; this is the bar the agents must beat |

### 17. Backtester

| ID | Task | Done when |
|---|---|---|
| 17.1 | Trade simulation: entry at the open, exits on stop / target / time; if stop and target fall on the same day, assume the stop hit first | Hand-calculated trades match |
| 17.2 | Indian costs: STT, exchange charges, SEBI fee, stamp duty, GST, DP charge (values in `config.yaml`) | Cost per trade matches a broker calculator |
| 17.3 | Gaps and price bands: gap through the stop fills at the open; no fill when the stock is locked at its band | Tests for both cases |
| 17.4 | Position sizing and portfolio: starting capital, risk per trade, maximum open positions | Equity curve for a portfolio, not single trades |
| 17.5 | Outputs: trade journal and daily equity curve | Files in the common format |

### 18. Evaluation

All comparisons use the same 50 stocks and the same days.

| ID | Task | Done when |
|---|---|---|
| 18.1 | Direction metrics (accuracy, F1) for next-day, next-week and the trade-plan horizon; volatility metrics (MAE, RMSE) | Metric functions with tests |
| 18.2 | Trading metrics: win rate, profit factor, Sharpe ratio, maximum drawdown | Metric functions with tests |
| 18.3 | Bootstrap confidence intervals | Every metric has an interval |
| 18.4 | Full agent backtest on the test window | Results for the agent system |
| 18.5 | Ablation runs: price only, no news, single analyst instead of debate, no critic, no memory | Ablation table |
| 18.6 | Leakage checks: results after the LLM cutoff only; hidden-name test (company names and dates masked on a sample) | Gap between masked and unmasked results reported |
| 18.7 | Automatic explanation checks: citation coverage, quote accuracy, rationale agrees with direction | Scores for every plan |
| 18.8 | Human rating study: rubric, 60 sampled plans rated blind, agreement between raters | Ratings table and agreement score |
| 18.9 | Final results tables and charts for the report | Exported to `report/figures/` |

### 19. Dashboard (Streamlit)

| ID | Task | Done when |
|---|---|---|
| 19.1 | App skeleton and data loading (results and trace files) | Runs on sample/mock data |
| 19.2 | Today's trade plans | Plan cards with evidence links |
| 19.3 | Trade journal | Filter by stock, date, outcome |
| 19.4 | Equity curves and metrics vs baselines | Agent vs every baseline |
| 19.5 | Event timeline per stock | Announcements, news, catalysts on the price chart |
| 19.6 | Agent trace viewer | Debate, critique and revisions for any plan |
| 19.7 | What-if simulator page | Runs 15.1 from the UI |
| 19.8 | Calibration and ablation pages | Charts from 15.2 and 18.5 |
| 19.9 | Data status page: tracker progress, review list, coverage per stock per day | Team can see data completeness at a glance |

### 20. Packaging and delivery

| ID | Task | Done when |
|---|---|---|
| 20.1 | Dockerfile and `docker compose` for pipeline and dashboard | `docker compose up` works on a clean machine |
| 20.2 | README and setup guide | A new person can run it from the README alone |
| 20.3 | Reproducibility check from a clean clone with the saved LLM cache | Same results as reported |
| 20.4 | Final report | All chapters complete |
| 20.5 | Demo script and video | Video recorded |
| 20.6 | Presentation slides | Ready for review |

---

## Part B: Milestones

Milestones are in build order: each one depends on the ones before it. Each ends with a demo of its "done when" check.

| Milestone | Tasks | Done when |
|---|---|---|
| **M0: Setup** | 1.1–1.10, 6.1–6.4 | Repo, CI, contracts, sample data and download tracker merged; LLM cutoff known |
| **M1: Pilot** | 1.11–1.12, 4.1 (first versions of 2.1–2.5, 3.1, 3.4, 4.4, 5.2 for 5 stocks) | Every collector runs for 5 stocks through the tracker; open items answered; scope decided; FII/DII daily job started |
| **M2: Price data** | 2.1–2.7, 6.5–6.6, 6.8, 10.1 | Clean adjusted prices for the 50 (and optionally 200) stocks; coverage table shows no unexplained gaps; indicators computed |
| **M3: Walking skeleton** | 11.1–11.4, 13.1–13.2, 13.4, 14.1–14.2, 19.1–19.2 | One day runs through all 7 stub agents; trade plan and trace shown in the dashboard |
| **M4: Baselines and backtester v1** | 16.1–16.4, 16.8, 17.1–17.5, 18.1–18.2, 19.3–19.4 | Buy-and-hold and MA crossover backtested with Indian costs; equity curves in the dashboard |
| **M5: Text and market data** | 3.1–3.9, 4.2–4.5, 5.1–5.5, 6.7 | Announcements, results, flows, macro and news loaded; leakage tests pass |
| **M6: Retrieval, setups, calendar** | 7.1–7.4, 8.1–8.6, 10.2–10.5 | Cited search works with no future leaks; setups tested alone; catalyst query works |
| **M7: Agents v1** | 12.1–12.8, 13.3 | 5 stocks × 1 real day run end to end with real evidence; Critic rejects a planted future document |
| **M8: Full pipeline** | 12.9, 13.5–13.6, 14.3–14.4, 9.1–9.3, 16.5–16.7, 16.9, 6.9–6.10, 19.5–19.6, 19.9 | One month of history runs unattended; Exit Monitor closes trades; baselines frozen; data status page live |
| **M9: Agent backtest and extras** | 15.1–15.2, 18.4–18.5, 19.7–19.8 | Full agent backtest done; ablation table; what-if demo works |
| **M10: Evaluation** | 18.3, 18.6–18.9 | All results final with intervals; rating study done; code freeze |
| **M11: Delivery** | 20.1–20.6 | Report, video and slides done; clean-clone check passes |

M4 (baselines) and M5 (text data) don't depend on each other and can be built at the same time. Start the FII/DII daily job (4.1) as early as possible: its history only grows one day at a time.

---

## Part C: Decisions and risks

**Decided**

| Decision | Choice |
|---|---|
| Universe | Nifty 50 for agents and evaluation; optional Nifty 200 prices for training |
| Main data source | NSE (BSE and Stooq block scripted access); yfinance as price backup |
| News | GDELT as a secondary source; NSE announcements are the main evidence |
| Storage | Raw files → Parquet + DuckDB; SQLite download tracker |
| Brief alignment | Next-day and next-week direction metrics reported alongside the trade plans |

**Still open**

1. LLM provider and model; its training cutoff sets the clean test window (1.10)
2. Long-only, or shorts through futures (recommended: long-only)
3. Vector store and embedding model (8.2)
4. After the pilot (1.12): GDELT history route, DII and bulk deals in or out, Nifty 200 prices yes or no

**Trade-count check (after 1.10 and the pilot)**

Expected trades in the clean test window = stocks × months in the window × setups per stock per month × share passing the Critic (about 0.6).
At least about 1,000 trades are needed to detect an edge of roughly 4 percentage points. If the estimate is lower, add the Nifty Next 50 (making the Nifty 100) to the agent universe.

**Main risks**

| Risk | What to do |
|---|---|
| NSE changes or blocks its endpoints | Keep every raw file; the tracker retries and queues failures for review; yfinance for prices |
| Results numbers not available from the API after Dec 2024 | Find the new source in the pilot; fall back to parsing results PDFs |
| FII/DII history incomplete | Daily job from day one; NSDL for FII history; drop DII history if no source is found |
| Look-ahead from late-released data (weekly FRED series, RBI decisions during market hours, evening FII/DII figures) | A `known_from` time for every record; leakage tests (6.7) |
| LLM already knows past outcomes | Score after the cutoff; hidden-name test (18.6); Critic rejects claims without dated evidence |
| LLM cost over 50 stocks × hundreds of days | Planner skips days with no setup or event; cache every call; prototype on 10 stocks |
| Wrong company matched in GDELT | Alias lists, relevance score, hand check (5.4) |
| GDELT history download is large (~760 GB) | Stream, filter and delete each file; or use BigQuery |
| Too few trades in the clean window | Trade-count check; add the Nifty Next 50 |
| Indian data proves too hard overall | Fall back to US stocks with SEC EDGAR, as in the original brief; agents unchanged, only the data layer changes |
| Short clean test window | Bootstrap intervals; report results honestly, including no improvement |
| Overfitting prompts or thresholds | Tune on validation only; test set used once (M9) |
