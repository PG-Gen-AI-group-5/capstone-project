# Agentic Swing-Trade Intelligence (NSE)

A multi-agent system that produces explainable 3–10 day swing-trade plans for Nifty 50 stocks. It is compared against ARIMA, LSTM, TFT and simple baselines with a walk-forward backtest that includes Indian trading costs.

- Task IDs, scope and each task's "done when": [WORK_PLAN.md](WORK_PLAN.md)
- Data sources, lake tables, download tracker, known data gaps: [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)
- Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before you add a module, file, table or notebook. It has the target folder layout, pipeline stages, model and forecast interfaces, and who owns each package.

## Environment

Windows, Python 3.11, virtual environment in `.venv`. Install with `pip install -e ".[dev]"`. Checks: `pytest` and `ruff check src tests`. The command-line tools document themselves (`python -m src.ingest --help`).

## The time rule

A record may inform a decision at time `as_of` only if it was public before then: `published_at < as_of`, or `known_from <= as_of` for macro values. The forecast for trading day `d` is made at 08:30 IST and the trade is entered at the 09:15 open of `d`.

- Store timestamps as timezone-aware UTC. Convert to IST only for display.
- Features, labels, models, agents and backtests get their data through the point-in-time readers in `src/store/` (task 6.6). If a reader you need doesn't exist yet, build it and its test first. EDA notebooks may query the DuckDB views directly.
- Every point-in-time reader has a leakage test: plant a record dated after `as_of` and check that the reader never returns it.

## Periods and the test set

Train 2016–2022, validate 2023, test 2024 onwards. Choose features, hyperparameters, thresholds and prompts using train and validation data only. Test-period results are produced once, by the freeze tasks: 16.9 for the baselines and 18.4 for the agents. In EDA, return and signal analysis covers 2016–2023; coverage and quality checks may cover the full range.

## Prices

`data/lake/prices_daily` holds raw, as-traded prices.

- Returns, indicators and model inputs come from the corporate-action-adjusted series (task 2.4). Build them from returns and ratios, so a later adjustment never changes a past feature value.
- Trade plans, stops, targets and backtest fills use the raw price as traded on that day.

## Code conventions

- Settings come from `config.yaml` through `src.config`. A new setting is a new key in `config.yaml`.
- Objects passed between modules are pydantic models in `src/contracts.py` (task 1.6), or DataFrame schemas declared next to the code that writes them.
- Keep notebooks thin: reusable logic goes in `src/`, with tests, and notebooks import it. A notebook explores data, calls `src/` to train, and presents the results.
- Pipeline stage outputs go to `data/derived/<table>/` as Parquet. Trained models go to `data/models/<model>/` with a manifest: git commit, config hash, training end date and seed.
- Tests run offline on the small datasets in `tests/fixtures/`.

## Notebooks

Notebooks go in `notebooks/eda/` and `notebooks/models/`, numbered in the order they run. Each one runs top to bottom on a fresh kernel (`jupyter nbconvert --to notebook --execute <file>`) and ends with a "Findings" cell: numbered findings, each backed by a number. Save the figures the report uses to `report/figures/<topic>/`. Clear outputs before committing.

## Git

- Use one branch per task, named `<task-id>-<short-name>` (for example `2.4-ca-adjustments`), merged into `main` through a pull request.
- `data/lake/`, `data/ingest.sqlite` and `data/logs/` are shared through git, and only the data lead commits them. Keep `data/` out of commits unless the task is data collection.

## Finishing a task

1. `pytest` and `ruff check src tests` pass.
2. Each "done when" item of the task in WORK_PLAN.md is either met, with evidence (a command and its output, or a number), or listed as not met with the reason.
3. Report the task IDs covered, the files changed, how you verified the work, and any open issues. New data gaps go in the "Open items" list in docs/DATA_COLLECTION.md.
