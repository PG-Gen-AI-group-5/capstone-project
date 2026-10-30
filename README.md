# Agentic Swing-Trade Intelligence for NSE Stocks

Capstone project: a multi-agent system that produces explainable 3–10 day swing-trade plans for Nifty 50 stocks.

- Work plan: [WORK_PLAN.md](WORK_PLAN.md)
- Data sources, storage and the download tracker: [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md)

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

`python -m src.ingest --help` lists every command. Downloaded data goes to `data/` (not committed); only `data/manual/` is in git.

## Tests

```
pytest
ruff check src tests
```
