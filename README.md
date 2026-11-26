# Agentic Swing-Trade Intelligence for NSE Stocks

Capstone project: a multi-agent system that produces explainable 3–10 day swing-trade plans for Nifty 50 stocks.

- Work plan: [WORK_PLAN.md](WORK_PLAN.md)
- Data sources, storage and the download tracker: [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)
- Code structure, pipeline stages and interfaces: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Prompt templates for EDA, model training and pipeline work with a coding agent: [docs/PROMPTS.md](docs/PROMPTS.md)
- Project rules for coding agents: [CLAUDE.md](CLAUDE.md)

## Setup

```
py -3.11 -m venv .venv            # Python 3.10 or newer
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env            # then add your FRED API key
```

## Data collection

```
python -m src.ingest init         # tracker database + NSE trading calendar
python -m src.ingest universe     # stock list from NSE (then review aliases by hand)
python -m src.ingest pilot        # every source for 5 stocks, June 2024
python -m src.ingest status       # progress per source
python -m src.ingest review       # what needs checking
```

Full backfill of one source, for example prices since 2016:

```
python -m src.ingest plan --source nse_bhavcopy --from 2016-01-01
python -m src.ingest run  --source nse_bhavcopy
```

`python -m src.ingest --help` lists every command.

### Shared data

The clean tables are in git, so you don't need to download anything to start working:

| In git | What it is |
| --- | --- |
| `data/lake/` | Clean Parquet tables (prices, index, announcements, results, insider trades, macro, ...) |
| `data/ingest.sqlite` | Download tracker: what was collected, what failed, what was reviewed (`status`, `review`) |
| `data/logs/` | Logs of the collection runs |
| `data/manual/` | Hand-made files (stock list, RBI decisions, special trading days) |

After pulling, run `python -m src.ingest views` once to create the DuckDB views for your machine.

Raw downloads (`data/raw/`, about 570 MB) stay on the machine that ran the collection; they are only needed to re-parse after a parser fix. **One person runs the collection and commits `data/`**, so the tracker database never gets conflicting edits.

## Models

```
pip install -e ".[dev,models,notebooks]"
python -m src.models prepare                 # adjusted prices, features, labels -> data/derived/
python -m src.models train --model all       # walk-forward over the validation year 2023
python -m src.models train --model all --period inner   # walk-forward folds over 2020-2022
python -m src.models evaluate                # metrics for every model (--period inner, --by vix_regime)
```

The pipeline runs the same stages in order and skips what is up to date:

```
python -m src.pipeline status                # each stage's last build and whether it is current
python -m src.pipeline run all --no-ingest   # prices_adj -> features -> labels -> forecasts -> evaluate (offline)
python -m src.pipeline run all               # the same, after downloading today's data
```

Models: `base_rate`, `ma_crossover`, `hist_vol`, `arima`, `lstm`, `tft`. Notebooks in `notebooks/models/` (10 → 15) explain each step and load saved forecasts instead of retraining. Figures and metric tables go to `report/figures/models/`. Test-period (2024+) forecasts need `--period test --freeze` and are made once, by task 16.9.

## Agents, backtest and dashboard

```
pip install -e ".[dev,models,agents,dashboard]"
python -m src.run --date 2023-06-01                    # the 7 agents for one day -> data/runs/<run_id>/
python -m src.run --from 2023-01-01 --to 2023-03-31    # a range; re-running skips finished days
python -m src.pipeline run agents --period validation  # agent run over 2023 + backtest of every strategy
streamlit run src/dashboard/app.py                     # plans, journal, equity vs baselines, traces, status
```

The agents call an OpenAI-compatible Chat Completions endpoint (`llm.provider: openai_compatible`), set in `.env` with
`JFF_LLM_BASE_URL`, `JFF_LLM_API_KEY`, `JFF_LLM_MODEL` and `JFF_LLM_TIMEOUT` (see `.env.example`; NVIDIA NIM by default).
Other providers: `anthropic` (Claude, `ANTHROPIC_API_KEY`) and `offline`, where each agent's rule-based stand-in produces
the same output, so everything also runs without a key. A reply that doesn't fit the agent's schema is retried once,
then the rules take over; responses are cached, and a daily call limit and budget cap use. Backtests use raw prices with Indian delivery costs (`costs` in `config.yaml`).

## The desk app

```
pip install -e ".[dev,models,agents,app]"
python -m src.api                                       # http://127.0.0.1:8000
python -m src.desk create --name "2023 rules" --start 2023-01-02   # or from the command line
python -m src.desk advance --desk 1 --days 20
```

A desk replays history one trading day at a time: each morning the agents plan with data up to the previous close, then
the session fills the plans at the open on the backtester's engine. In the app you can review each plan with its debate,
evidence and trace, skip a trade, buy an analysed stock yourself, exit a position, and see the trade book, history and
analysis. Pick "Rule-based agents" for instant days, or an LLM provider (about 40 calls per day). The design it follows
is `docs/design/trading-with-ai.html`; the database and routes are in `docs/ARCHITECTURE.md`.

## Tests

```
pytest
ruff check src tests
```
