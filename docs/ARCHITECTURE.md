# Architecture

How the code is organised and how data flows from downloads to trade plans, backtests and the dashboard. Task IDs refer to [WORK_PLAN.md](../WORK_PLAN.md). The Status column shows what exists today; everything marked "to build" follows the layout below.

---

## Principles

1. **One code path for backtest and live.** Every stage takes a date or a date range and reads only what was public before it. Running it for today and running it for a day in 2024 use the same code, so the backtest tests the same system that runs live.
2. **Layers depend downward.** ingest → store → features, labels, catalysts, retrieval, memory → models and agents → backtest → evaluation → dashboard. A module imports only from its own layer or from layers below it.
3. **Stages talk through files.** Each stage writes Parquet tables with a declared schema to `data/derived/`. Any layer can then be built, tested and re-run on its own, and the dashboard only reads files.
4. **Thin notebooks.** Notebooks call `src/` and present results. The pipeline and the notebooks run the same functions.
5. **Plain Python, one command-line tool.** The system runs on one machine once a day, so Windows Task Scheduler calling `python -m src.pipeline` is enough. There's no workflow engine.

---

## Folder layout

```
capstone-project/
├── CLAUDE.md                  rules for coding agents
├── WORK_PLAN.md               tasks, milestones, decisions
├── config.yaml                every setting (paths, periods, costs, LLM, models)
├── pyproject.toml             dependencies, split into extras by layer (see below)
├── data/                      see docs/DATA_COLLECTION.md
│   ├── manual/  raw/  lake/  ingest.sqlite  market.duckdb    exists
│   ├── derived/               stage outputs: prices_adj, features_daily, labels, forecasts, catalysts, chunks, events
│   ├── models/                trained models + manifest.json per model
│   ├── vector/                embeddings index
│   ├── cache/                 LLM responses
│   └── runs/<run_id>/         agent plans, traces, trade journal, equity curve, metrics
├── docs/
├── notebooks/
│   ├── eda/                   01_data_inventory, 02_prices_returns, 03_events, 04_macro_flows
│   └── models/                10_labels_splits, 11_baselines, 12_arima, 13_lstm, 14_tft, 15_model_comparison
├── report/
│   ├── eda/EDA_REPORT.md
│   └── figures/<topic>/
├── src/
│   ├── config.py  timeutil.py                      exists
│   ├── contracts.py                                1.6   shared pydantic models (v1 exists; team review pending)
│   ├── logsetup.py                                 1.8   structured logging, one file per run
│   ├── ingest/                                     exists: collectors, tracker, coverage, DuckDB views
│   ├── store/                                      point-in-time store
│   │   ├── adjust.py                               2.4   adjusted prices
│   │   ├── db.py                                   6.5   DuckDB connection over lake/ and derived/
│   │   └── pit.py                                  6.6   the time rule, AnnouncementIndex, prices_as_of (exist); news, flows, macro readers
│   ├── features/  indicators.py build.py events.py regime.py setups.py news.py   10.1–10.4, 5.5
│   ├── labels/    labels.py splits.py                               16.1–16.2
│   ├── catalysts/ calendar.py                                       7.1–7.4  v1: results (notices or quarterly cycle), expiry, budget
│   ├── retrieval/ chunk.py embed.py index.py search.py citations.py 8.1–8.5
│   ├── memory/    events.py                                         9.1–9.3
│   ├── models/    data.py base.py registry.py walkforward.py baselines.py arima.py sequence.py lstm.py tft.py __main__.py   16.3–16.9
│   ├── llm/       client.py openai_compatible.py promptlib.py prompts/<agent>/v<N>.md   11.1–11.4  exists
│   ├── agents/    context.py base.py planner.py retriever.py analysts.py forecaster.py critic.py exit_monitor.py   12.1–12.9  v1 (analysts.py = bull, bear, rebuttals)
│   ├── graph/     build.py                                          13.1–13.3  exists (state and routing in build.py)
│   ├── tracing/   writer.py                                         14.1–14.3  exists (trace JSONL, run manifest)
│   ├── backtest/  signals.py engine.py costs.py run.py              16.8, 17.1–17.5  exists (portfolio in engine.py)
│   ├── evaluation/ metrics.py bootstrap.py ablations.py explain_checks.py   18.1–18.7
│   ├── analysis/  event_study.py plots.py                           helpers for EDA notebooks
│   ├── pipeline/  stages.py runner.py __main__.py                   stage registry, runner, command line (exists)
│   ├── whatif.py                                                    15.1
│   ├── run.py                                                       13.4–13.6  agent runs: one day or a range, resume, positions (exists)
│   ├── desk/      store.py data.py service.py live.py views.py jobs.py __main__.py   desks (replay + live) and their SQLite store (exists)
│   ├── live/      env.py update.py models.py schedule.py __main__.py   live suggester: daily data, live models, schedule (exists)
│   ├── api/       app.py __main__.py                                FastAPI backend for the app (exists)
│   └── dashboard/ app.py                                            19.x  v1 exists (streamlit run src/dashboard/app.py)
├── frontend/      index.html styles.css app.js                      the desk app (docs/design/trading-with-ai.html made real)
├── scripts/       live.cmd                                          what the scheduled tasks run (python -m src.live ...)
└── tests/                     mirrors src/; fixtures/ holds small real data slices (1.7)
```

---

## Data flow

```
 NSE · FRED · GDELT · RBI (manual)
              │  src.ingest + download tracker
              ▼
 data/raw ──► data/lake (Parquet) ──► DuckDB views
                                          │
                                          │  src.store: adjusted prices, as_of readers
     ┌──────────────┬──────────────┬──────┴───────┬──────────────┐
     ▼              ▼              ▼              ▼              ▼
  features       labels        catalysts      retrieval       memory
  indicators,    direction,    results,       chunks,         similar past
  regime,        volatility,   RBI, expiry    vectors,        events and
  setups         splits                       citations       reactions
     │              │              │              │              │
     └──────┬───────┘              │              │              │
            ▼                      │              │              │
  models: baselines, ARIMA,        │              │              │
  LSTM, TFT ──► forecasts ─────────┴──────────────┴──────────────┤
            │                                                    ▼
            │                                   agents (LangGraph): Planner → Retriever
            │                                   → Bull/Bear debate → Forecaster → Critic
            │                                   (+ Exit Monitor) ──► trade plans + traces
            ▼                                                    │
         signals ─────────────► backtest ◄───────────────────────┘
                                   │  trade journal, equity curve
                                   ▼
                              evaluation ──► report/figures
                                   │
                                   ▼
                     dashboard (reads data/derived and data/runs)
```

---

## Pipeline stages

Each stage is a function registered in `src/pipeline/stages.py`. It declares the tables it reads and writes, and it writes a `_manifest.json` beside its output: stage, git commit, row count and build time, plus the runner's record of the settings hash, input times and runtime. Re-running a stage overwrites the same output.

| Stage | Module | Reads | Writes | Tasks |
|---|---|---|---|---|
| `ingest` | `src.ingest` | NSE, FRED, GDELT, manual CSVs | `data/lake/*` | 2–5, 6.9 |
| `prices_adj` | `src.store.adjust` | `prices_daily`, `corporate_actions`, `manual/ca_adjustments.csv` | `derived/prices_adj` | 2.4 |
| `features` | `src.features` | `prices_adj`, `index_daily`, `macro`, `announcements`, `financial_results` | `derived/features_daily`, `derived/calendar`, `derived/results_releases` | 10.1–10.3 |
| `labels` | `src.labels` | `prices_adj` | `derived/labels` | 16.1 |
| `catalysts` | `src.catalysts` | `board_meetings`, `financial_results`, `rbi_policy`, calendar | `derived/catalysts` | 7.1–7.3 |
| `chunks` | `src.retrieval` | `announcement_text`, `news` | `derived/chunks`, `data/vector/` | 8.1–8.3 |
| `events` | `src.memory` | `announcements`, `prices_adj`, `index_daily` | `derived/events` | 9.1 |
| `forecasts` | `src.models` | `features_daily`, `labels`, `catalysts` | `data/models/<m>/`, `derived/forecasts/<m>/` | 16.3–16.7b |
| `agent_run` | `src.run` → `src.graph` | features, forecasts, announcements, board meetings, calendar overrides (through `src.store.pit`) | `data/runs/agents-<period>/`: plans, exits, positions, trace, manifest | 12–14 |
| `backtest` | `src.backtest` | prices_adj (raw OHLC), forecasts → signals, agent plans and exits → signals | `derived/backtest/<strategy>/`: journal, equity, summary | 16.8, 17 |
| `evaluate` | `src.evaluation` | runs, labels | `runs/<id>/metrics`, `report/figures/` | 18 |

Built so far (`src/pipeline/`): `ingest`, `prices_adj`, `features`, `labels`, `forecasts`, `evaluate`, `agent_run` and `backtest`. The `agents` target runs the last two. Each stage in `stages.py` lists what it reads (`lake:<table>`, `manual:<file>`, `stage:<name>`) and the `config.yaml` keys it depends on. After a run, the runner adds a `pipeline` record to the stage's manifest: the hash of those settings, the time of each input, and the runtime. A stage is out of date when it was never built, an input changed after it, its settings changed, or (for `ingest`) it hasn't run today. Stages rebuild full history, which takes seconds; there is no date range.

```
python -m src.pipeline status                 # every stage: last build and whether it is up to date (and why not)
python -m src.pipeline run all                # ingest -> prices_adj -> features -> labels -> forecasts -> evaluate
python -m src.pipeline run all --no-ingest    # the same, offline
python -m src.pipeline run data               # ingest, prices_adj, features, labels (the evening job)
python -m src.pipeline run models --period validation --period inner
python -m src.pipeline run agents --period validation   # agent run over 2023, then every strategy backtested
python -m src.pipeline run features labels    # stages named explicitly always run
python -m src.pipeline run forecasts --models lstm,tft --force    # retrain these models
```

Group targets (`all`, `data`, `models`) skip stages that are up to date; `--force` runs them anyway. `forecasts` reuses saved forecasts whose model version (settings plus the content of the built inputs) is unchanged, so only changed models retrain. `evaluate` writes `derived/metrics/<period>.parquet`, the majority-guess reference and a split by VIX regime. The test period needs `--period test --freeze` (task 16.9).

The modelling commands also work on their own:

```
python -m src.models prepare                 # prices_adj -> features -> labels
python -m src.models train --model arima     # walk-forward over the validation year; --model all for every model
python -m src.models train --model all --period inner   # walk-forward folds inside the training years (2020-2022)
python -m src.models evaluate                # metrics, plus the constant majority guess for the same days
python -m src.models evaluate --by vix_regime   # the same, split by market regime (or --by trend_regime)
```

Model inputs leave out short special sessions (`features.skip_sessions`: Diwali Muhurat trading, which has no 09:15 open), in both features and labels. Saved forecasts are reused only when the config and the built inputs (the stage manifests) are unchanged.

**Live daily schedule** (task 6.9). Run the evening steps after FII/DII are published (20:00 IST):

1. **Evening, about 20:30 IST:** `ingest` → `prices_adj` → `features` → `labels` (for past days) → `catalysts` → `chunks`.
2. **Morning, 07:30 IST:** `forecasts` → `agent_run` for today (`as_of` 08:30 IST), including the Exit Monitor on open trades.

---

## Interfaces

### Point-in-time readers (`src/store/pit.py`)

Built so far: `as_of_utc(trade_dates)` (08:30 IST on each day) and `first_usable_day(published_at, days)`, which places each dated record on its day 0, the first trading day with `published_at < as_of`. The event features (`src/features/events.py`) and the EDA's event studies both use it. The per-symbol readers below are still to build (task 6.6):

```python
def prices_as_of(symbols: list[str], as_of: datetime, lookback_days: int) -> pd.DataFrame
def announcements_as_of(symbol: str, as_of: datetime, since: datetime) -> list[Announcement]
def macro_as_of(series: list[str], as_of: datetime) -> pd.DataFrame          # filters on known_from
def flows_as_of(as_of: datetime, lookback_days: int) -> pd.DataFrame
def news_as_of(symbol: str, as_of: datetime, since: datetime) -> list[NewsItem]
```

When each kind of data becomes usable:

| Data | Usable from |
|---|---|
| Prices | Bhavcopy is published at 18:00 IST, so at 08:30 on day `d` the latest close is day `d−1`. |
| Announcements and results | Their exchange release time (`published_at`). |
| FII/DII | 20:00 IST on the trading day. |
| Macro | `known_from`. |
| RBI decisions | The next trading day. |

### Forecast table (written by every model)

`data/derived/forecasts/<model>/<period>.parquet` has one row per model, symbol, day and horizon:

| Column | Meaning |
|---|---|
| `model`, `model_version` | Model name and the version recorded in its manifest |
| `symbol` | NSE symbol |
| `trade_date` | Trading day `d` the forecast is for (entry at its open) |
| `as_of` | 08:30 IST on `d`, stored as UTC |
| `horizon_days` | 1, 5 or 10 |
| `p_up` | Probability that the horizon return is positive (NaN for volatility-only models) |
| `direction` | +1, −1, or NaN for volatility-only models |
| `exp_return` | Expected horizon log return, if the model gives one |
| `vol_forecast` | Forecast realised volatility: root mean square of daily log returns over the horizon (not annualised) |
| `train_end` | Last date any training label looked at; always before `trade_date` (checked by `check_forecasts`) |

### Labels (`src/labels/labels.py`)

For the row of day `d` and horizon `h` (days `d .. d+h−1`), from adjusted prices:

| Column | Meaning |
|---|---|
| `fwd_ret_h` | Log return from the open of `d` (`labels.entry: open`) or the close of `d−1` (`close`) to the close of `d+h−1` |
| `up_h` | 1 if `fwd_ret_h > 0`, else 0 |
| `fwd_vol_h` | Root mean square of the daily close-to-close log returns of days `d .. d+h−1` |
| `end_h` | Date of `d+h−1`, used by the gap rule |

### Model interface (`src/models/base.py`)

Every model sees one aligned panel (`src/models/data.py`): features and labels for every stock and trading day. The runner passes boolean row masks.

```python
class ForecastModel(ABC):
    name: str
    def __init__(self, cfg: Config, params: dict | None = None): ...
    def fit(self, panel: Panel, train_mask: np.ndarray) -> None: ...          # labels only from these rows
    def predict(self, panel: Panel, mask: np.ndarray) -> pd.DataFrame: ...    # forecast-table rows
    def save(self, folder: Path) -> None: ...                                 # one folder per walk-forward fold
```

The LSTM and TFT also have `load(cfg, folder, panel)` to rebuild a saved fold's model. Register new models in `src/models/registry.py`.

### Walk-forward (`src/models/walkforward.py`)

- **Window:** expanding, refitted on the schedule in `models.refit` (ARIMA monthly; the others yearly).
- **The gap rule:** a training row for day `t` looks at prices up to `end_h` (day `t+h−1`). For predictions starting at `T`, a row trains only if its `end_10` is before `T`.
- **Early stopping (LSTM, TFT):** the last `early_stop_days` of the training rows are held out, with the same gap before them.
- **Validation:** the same procedure over 2023.
- **Inner folds:** the same procedure over 2020–2022 (`periods.inner`), inside the training years, so models are also checked on a stressed market (the 2020 crash, the 2022 fall). 2023 is calm: no high-VIX days (EDA 02 §10).
- **Test:** the same procedure over 2024 onwards, run once by task 16.9.

### Signals (what the backtester accepts)

`symbol`, `signal_date` (the `as_of`), `side` (long), `entry` (next open), `stop`, `target`, `max_hold_days`, `source` (model name or agent run ID) and an optional `ref` (the desk's plan ID). A position sold at the open on an Exit Monitor or manual request has exit reason `exit`.

- **Baselines:** the 16.8 rules turn forecasts into signals: for example, long when `p_up` is above a threshold tuned on validation, with the ATR stop and target from 10.4.
- **Agents:** each agent `TradePlan` maps one-to-one onto a signal.

### Agents

Built (v1): the seven agents run through the LangGraph graph in `src/graph/build.py`:

```
day:    START -> exit_monitor -> planner -> analyse (one branch per selected stock) -> END
stock:  START -> retrieve -> debate (opening cases + one rebuttal round) -> forecast -> critique -> END
        critique: accept -> END | reject -> END (no trade) | revise -> forecast (at most agents.critic_max_rounds times)
```

`python -m src.run --date D` (or `--from A --to B`) writes `data/runs/<run_id>/`. LLM providers (`llm.provider`):
`openai_compatible` (any OpenAI-style Chat Completions endpoint, from `JFF_LLM_*` in `.env`; `src/llm/openai_compatible.py`),
`anthropic` (Claude SDK) and `offline` (each agent's rule-based stand-in, no key needed). The agent's JSON schema is sent
with the prompt; a reply that fails validation is retried once with the error, then the offline rules take over. Entry, stop and target always come from `src/features/setups.py`; the Critic rejects a
plan that changes them, cites a record published at or after `as_of`, or cites evidence that wasn't retrieved.

- Each agent is a LangGraph node, `def node(state: GraphState) -> dict`, returning a partial state update.
- Prompts are versioned files in `src/llm/prompts/<agent>/v<N>.md`. Every output records the prompt version it used.
- Every LLM call goes through `src.llm.client`, which handles the cache, cost log, pydantic validation of the JSON output, and retries.
- Every call, tool use, debate turn and critique writes one trace step through `src.tracing.writer`.

### The desk app (`src/desk`, `src/api`, `frontend/`)

A **desk** is a paper-trading account that replays history one trading day at a time, so the agents run as daily
inference rather than as a batch over a period. For trading day D:

```
morning(D)  08:30 IST   Exit Monitor on the open positions -> Planner -> retrieve/debate/forecast/critique per stock
                        (the same day graph as src/run.py). Nothing after D-1's close is visible, in the agents or the app.
            a person    can remove a stock (its plan is not traded), buy an analysed stock, exit a position, or add a
                        stock for analysis. These are orders for D's open.
session(D)              Book.session in src/backtest/engine.py: the backtester's rules, one day at a time
                        (exits and entries at the open, stop before target, time exits at the close, Indian costs).
```

`advance` runs session(D) and then morning(next day). Each morning and each session is one SQLite transaction, so a
crash leaves the desk at its last whole step. Desks may replay only `desk.periods` (inner folds and validation by
default, where walk-forward forecasts exist); the test period stays frozen for task 18.4.

**Database** (`desk.db`, default `data/app/desk.sqlite`, outside git). Market data is not copied; it stays in the lake
and `data/derived`. Tables, defined in `src/desk/store.py`:

| Table | One row per |
|---|---|
| `desks` | desk: LLM provider, window, capital, frozen settings, current day and phase, the engine's book (cash, positions) |
| `days` | trading day reached: the Planner's output, LLM calls, cost and fallbacks |
| `watchlist` | stock on the desk that day: picked by the Planner or added by a person; removed or not |
| `plans` | analysed stock and day: the Forecaster's plan, the bull and bear cases, the Critic's verdict, revisions, catalysts |
| `evidence` | record a plan could cite (announcements, base forecast, indicator and level records) |
| `exit_decisions` | open position and day: the Exit Monitor's decision |
| `pending_orders` | a person's buy or exit for the next session |
| `orders` | fill or refused entry (exact prices and costs, so they add up to the cash) |
| `trades` | closed round trip, with its source (agents or manual), setup and exit reason |
| `equity` | close: cash, positions, equity |
| `events` | line in the desk feed |
| `trace_steps` | agent step (LLM call or offline rule), as in `runs/<id>/trace.jsonl` |
| `jobs` | background job: create, advance, analyse or live update, with progress |
| `holdings` | live desks: a trade you took, open until you report the sale |

**API** (`python -m src.api`, FastAPI on 127.0.0.1:8000, which also serves `frontend/`):

| Route | Does |
|---|---|
| `GET /api/meta` | LLM providers and whether each is configured, the replay window, symbols |
| `GET/POST /api/desks`, `GET/DELETE /api/desks/{id}` | list, create (starts the first morning as a job), summary, delete |
| `POST /api/desks/{id}/advance` | run N days as a job; `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel` |
| `GET/POST /api/desks/{id}/watchlist`, `DELETE .../watchlist/{symbol}` | today's list; add a stock (an analysis job); remove one |
| `GET /api/desks/{id}/stocks/{symbol}` | chart (daily bars to the last visible close, EMA20/50, forecast band), indicators, setup checks, plan and debate, evidence, allowed actions |
| `GET /api/desks/{id}/trace?date=&symbol=` | the agent steps behind a plan |
| `POST/DELETE /api/desks/{id}/orders` | a person's buy or exit for the next open; cancel it |
| `POST /api/desks/{id}/refresh` | live desks: download, rebuild, forecast and suggest (a job) |
| `GET/POST /api/desks/{id}/holdings`, `PATCH .../holdings/{hid}`, `POST .../holdings/{hid}/close` | your trades on a live desk: list with today's advice, add, change the stop, report the sale |
| `GET /api/live` | live data and model status, and the scheduled tasks |
| `GET /api/desks/{id}/book`, `/trades`, `/equity`, `/days`, `/events` | trade book, history and analysis, the feed |

The chart's range is the LSTM's volatility forecast (the model that held up in validation) around the base model's
expected return; charts show prices on the share basis of their last visible day, so a later split never shows early.
`python -m src.desk create|advance|list` runs the same steps without the server.

### The live suggester (`src/live`, `src/desk/live.py`)

A **live desk** suggests trades for the next trading day D and watches the trades you report. Every weekday at 08:00 IST
(Windows Task Scheduler, `python -m src.live schedule install`) and whenever you press "Update and suggest":

```
1. ingest       NSE bhavcopy and indices from a few days before the last price day (catches up a missed week),
                announcements and board meetings (re-read), weekly corporate actions, results and FRED -> the shared lake
2. build        data/live/derived: adjusted prices, labels, and features including the row for D, built before D's
                prices exist (features/build.py add_placeholder_day; a test checks it equals the row built afterwards)
3. forecast     the live TFT and LSTM predict D (data/live/derived/forecasts/<model>/live.parquet)
4. agents       per live desk: the Exit Monitor on your holdings, then the Planner, debate, forecast and critique
```

- **Live models** are refitted on every row whose labels end before D, weekly (Saturday 10:00, `live.refit_every_days`),
  so the morning run only predicts. They train on 2024+ data, so they live in `data/live/models` and are never used to
  evaluate anything; the walk-forward and frozen test forecasts in `data/derived/forecasts` are untouched.
- **Separate tables.** The live build writes `data/live/derived`, not `data/derived`, so the research pipeline's
  manifests don't change. The daily ingest does add new days to the shared lake (committed only by the data lead).
- **Holdings.** "I bought" records your date, price and quantity with the plan's stop, target and holding period. Each
  morning the desk checks whether the stop or target was touched since your last look, whether the holding period is
  over, and asks the Exit Monitor (hold, raise the stop, exit at the open). "I sold" closes it into `trades` with
  estimated delivery costs, for the history and analysis pages.
- A suggestion is only made for a day that hasn't passed: if the download brings no newer prices, the run stops and says so.

```
python -m src.live desk --name Live            # a live desk (LLM provider from config, or --provider offline)
python -m src.live daily                       # what the 08:00 task runs
python -m src.live refit                       # what the Saturday task runs (about an hour on CPU)
python -m src.live status | schedule install | schedule remove
```

---

## Dependencies by layer

Group dependencies as optional extras in `pyproject.toml`, so each person installs only what their layer needs. Docker installs all of them.

| Extra | For |
|---|---|
| `dev` | pytest, ruff (exists) |
| `notebooks` | jupyterlab, matplotlib |
| `models` | statsmodels, torch, one TFT library (record which one, and why, in the 16.7b notebook) |
| `agents` | langgraph, the chosen LLM SDK, the vector store, the embedding model |
| `dashboard` | streamlit, plotly |
| `app` | fastapi, uvicorn (the desk app) |

---

## Ownership

Taken from the project brief. The owner reviews every pull request that touches their packages.

| Packages | Owner |
|---|---|
| `ingest`, `store`, `catalysts`, data-quality checks | Sruthi Narayanan (data engineering) |
| `retrieval`, `memory`, `agents/retriever.py` | Mridul Shan (retrieval) |
| `features/setups.py`, `agents/bull.py`, `bear.py`, `forecaster.py`, `TradePlan` in `contracts.py` | Reventh Noel (analysis and forecasting) |
| `models`, `backtest`, `evaluation`, `agents/critic.py` | Rudresh Narwal (modelling and evaluation) |
| `graph`, `tracing`, `agents/planner.py`, `exit_monitor.py`, `whatif.py`, Docker, CI | Jithu Tagore (lead, agent architecture) |
| `dashboard`, explanation study, `report/` | Guru Raghavendran (dashboard and report) |

---

## Build order

| Phase | Milestone | Builds |
|---|---|---|
| 1 | M0 | `contracts.py`, config sections, `logsetup.py`, CI, `tests/fixtures/` |
| 2 | M2 | `store/` (adjust, pit, leakage tests), `features/indicators.py`, `labels/`, `pipeline/` with the first three stages |
| 3 | parallel with 2 | EDA notebooks 01–04, `analysis/`, the EDA report |
| 4 | M3 | `llm/`, `graph/` with stub agents, `run.py`, `tracing/`, dashboard skeleton |
| 5 | M4 | `models/` (baselines, ARIMA, LSTM, TFT), `backtest/`, `evaluation/metrics.py`, model notebooks 10–15 |
| 6 | M5–M8 | text data, `retrieval/`, `catalysts/`, `memory/`, real agents |
