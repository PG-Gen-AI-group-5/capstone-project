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
│   ├── contracts.py                                1.6   shared pydantic models
│   ├── logsetup.py                                 1.8   structured logging, one file per run
│   ├── ingest/                                     exists: collectors, tracker, coverage, DuckDB views
│   ├── store/                                      point-in-time store
│   │   ├── adjust.py                               2.4   adjusted prices
│   │   ├── db.py                                   6.5   DuckDB connection over lake/ and derived/
│   │   └── pit.py                                  6.6   as_of readers: prices, announcements, news, flows, macro
│   ├── features/  indicators.py regime.py setups.py news.py         10.1–10.4, 5.5
│   ├── labels/    labels.py splits.py                               16.1–16.2
│   ├── catalysts/ calendar.py                                       7.1–7.4
│   ├── retrieval/ chunk.py embed.py index.py search.py citations.py 8.1–8.5
│   ├── memory/    events.py                                         9.1–9.3
│   ├── models/    data.py base.py registry.py walkforward.py baselines.py arima.py sequence.py lstm.py tft.py __main__.py   16.3–16.9
│   ├── llm/       client.py cache.py budget.py prompts/<agent>/v<N>.md                               11.1–11.4
│   ├── agents/    planner.py retriever.py bull.py bear.py forecaster.py critic.py exit_monitor.py    12.1–12.9
│   ├── graph/     state.py build.py routing.py                      13.1–13.3
│   ├── tracing/   writer.py manifest.py load.py                     14.1–14.4
│   ├── backtest/  signals.py engine.py costs.py portfolio.py        16.8, 17.1–17.5
│   ├── evaluation/ metrics.py bootstrap.py ablations.py explain_checks.py   18.1–18.7
│   ├── analysis/  event_study.py plots.py                           helpers for EDA notebooks
│   ├── pipeline/  stages.py __main__.py                             stage registry and command line
│   ├── whatif.py                                                    15.1
│   ├── run.py                                                       13.4  daily agent run
│   └── dashboard/ app.py pages/                                     19.x  (streamlit run src/dashboard/app.py)
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

Each stage is a function registered in `src/pipeline/stages.py`. It declares the tables it reads and writes, and it writes a `_manifest.json` beside its output: stage, date range, git commit, config hash, row count, input manifests, build time. Re-running a stage for the same dates overwrites the same output.

| Stage | Module | Reads | Writes | Tasks |
|---|---|---|---|---|
| `ingest` | `src.ingest` | NSE, FRED, GDELT, manual CSVs | `data/lake/*` | 2–5, 6.9 |
| `prices_adj` | `src.store.adjust` | `prices_daily`, `corporate_actions`, `manual/ca_adjustments.csv` | `derived/prices_adj` | 2.4 |
| `features` | `src.features` | `prices_adj`, `index_daily`, `macro` | `derived/features_daily` | 10.1–10.3 |
| `labels` | `src.labels` | `prices_adj` | `derived/labels` | 16.1 |
| `catalysts` | `src.catalysts` | `board_meetings`, `financial_results`, `rbi_policy`, calendar | `derived/catalysts` | 7.1–7.3 |
| `chunks` | `src.retrieval` | `announcement_text`, `news` | `derived/chunks`, `data/vector/` | 8.1–8.3 |
| `events` | `src.memory` | `announcements`, `prices_adj`, `index_daily` | `derived/events` | 9.1 |
| `forecasts` | `src.models` | `features_daily`, `labels`, `catalysts` | `data/models/<m>/`, `derived/forecasts/<m>/` | 16.3–16.7b |
| `agent_run` | `src.run` → `src.graph` | everything, through `src.store.pit` | `runs/<id>/plans`, `runs/<id>/trace.jsonl` | 12–14 |
| `backtest` | `src.backtest` | forecasts → signals, or plans → signals | `runs/<id>/journal`, `runs/<id>/equity` | 16.8, 17 |
| `evaluate` | `src.evaluation` | runs, labels | `runs/<id>/metrics`, `report/figures/` | 18 |

Commands:

```
python -m src.pipeline run <stage> --from 2023-01-01 --to 2023-12-31
python -m src.pipeline run all --from ... --to ...
python -m src.pipeline status            # last build of each stage, its date range and inputs
```

Until `src.pipeline` exists, the modelling stages run through `python -m src.models`:

```
python -m src.models prepare                 # prices_adj -> features -> labels
python -m src.models train --model arima     # walk-forward over the validation year; --model all for every model
python -m src.models evaluate                # metrics table for every model with saved forecasts
```

**Live daily schedule** (task 6.9). Run the evening steps after FII/DII are published (20:00 IST):

1. **Evening, about 20:30 IST:** `ingest` → `prices_adj` → `features` → `labels` (for past days) → `catalysts` → `chunks`.
2. **Morning, 07:30 IST:** `forecasts` → `agent_run` for today (`as_of` 08:30 IST), including the Exit Monitor on open trades.

---

## Interfaces

### Point-in-time readers (`src/store/pit.py`)

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
- **Test:** the same procedure over 2024 onwards, run once by task 16.9.

### Signals (what the backtester accepts)

`symbol`, `signal_date` (the `as_of`), `side` (long), `entry` (next open), `stop`, `target`, `max_hold_days` and `source` (model name or agent run ID).

- **Baselines:** the 16.8 rules turn forecasts into signals: for example, long when `p_up` is above a threshold tuned on validation, with the ATR stop and target from 10.4.
- **Agents:** each agent `TradePlan` maps one-to-one onto a signal.

### Agents

- Each agent is a LangGraph node, `def node(state: GraphState) -> dict`, returning a partial state update.
- Prompts are versioned files in `src/llm/prompts/<agent>/v<N>.md`. Every output records the prompt version it used.
- Every LLM call goes through `src.llm.client`, which handles the cache, cost log, pydantic validation of the JSON output, and retries.
- Every call, tool use, debate turn and critique writes one trace step through `src.tracing.writer`.

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
