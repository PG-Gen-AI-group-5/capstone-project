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

## Tests

```
pytest
ruff check src tests
```
