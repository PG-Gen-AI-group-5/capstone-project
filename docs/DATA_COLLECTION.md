# Data collection guide

How to collect, store and track every dataset for the project, step by step. All URLs, field names and sizes below were tested on 30 Oct 2026 from a home connection in India. NSE can change or block its endpoints at any time; if a step stops working, check the open items at the end first.

---

## 0. Before you start

### Universe

| Universe | Stocks | Used for | Source list |
|---|---|---|---|
| **Nifty 50** | 50 | Agents, announcements, news, all evaluation | `https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv` |
| **Nifty 200** (optional) | 200 | Price data only: training price models and setup statistics | `https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv` |
| Nifty 100 (if the trade-count check needs more agent stocks) | 100 | Agents, if expanded | `https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv` |

Every comparison between agents and baselines uses the same 50 stocks and days. Extra stocks only add training data.

### Install

```
pip install requests pandas pyarrow duckdb yfinance pdfplumber
```

### Keys

| Key | Needed for | How to get |
|---|---|---|
| FRED API key | Step 10 | Free: create an account on fred.stlouisfed.org → My Account → API Keys |
| Google Cloud project | Step 12, only if the BigQuery route is chosen | Free tier includes 1 TB of BigQuery queries a month |

Keys go in your own `.env`, never in git.

### Where files go

```
data/
  raw/            exactly as downloaded, never edited
  lake/           cleaned Parquet tables
  derived/        features, labels, catalysts, event memory, chunks
  manual/         small hand-made CSVs (universe, RBI decisions)   ← committed to git
  vector/         embeddings index
  runs/           agent outputs and backtests
  cache/          LLM responses
  ingest.sqlite   download tracker (section T)
  market.duckdb   SQL views over lake/ and derived/
```

Everything in `data/` except `data/manual/` is git-ignored. Share `lake/` as a dated snapshot (for example `lake_2026-11-15.zip`) through Google Drive.

### Rules for every collector

1. Send a browser-like `User-Agent` and `Referer: https://www.nseindia.com/` header. Without them NSE returns an error page.
2. At most **1 request per second** to NSE, **1 request every 5 seconds** to the GDELT API.
3. **Every download goes through the tracker** (section T). An error never stops the run: the tracker records it, retries later or queues it for review, and the run moves on to the next task.
4. **Resumable:** before downloading, check whether the raw file already exists.
5. Run long downloads outside market hours (before 09:00 or after 16:00 IST).
6. If NSE API calls start returning an HTML page instead of JSON, the session needs cookies: request `https://www.nseindia.com/` first with the same session, then retry.

### Shared HTTP helper

```python
import time
import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")

session = requests.Session()
session.headers.update({
    "User-Agent": UA,
    "Referer": "https://www.nseindia.com/",
    "Accept": "application/json, text/plain, */*",
})

def get(url, pause=1.0, timeout=30):
    """One request. The tracker decides what to do with errors and retries."""
    r = session.get(url, timeout=timeout)
    time.sleep(pause)
    return r
```

---

## Order of collection

Downloads run per **natural unit**: whatever one request returns. Running every source per stock per day would download the same bhavcopy file 50 times (~140,000 requests instead of ~2,800). Checking per stock per day happens afterwards, in the coverage table (section T).

| Step | Data | Period | One task = | Approx. tasks | Approx. time |
|---|---|---|---|---|---|
| 1 | Universe | Current | — | 1 | seconds |
| 2 | Daily prices (bhavcopy) | 2016 → today | 1 trading day, all stocks | ~2,800 | ~50 min |
| 3 | Nifty 50 + India VIX | 2016 → today | 1 trading day | ~2,800 | ~50 min |
| 4 | Corporate actions | 2016 → today | 1 stock | 50 (200) | 1 min |
| 5 | Quarterly results + board meetings | 2015 → today | 1 stock | 100 | 2 min |
| 6 | Announcements list | 2016 → today | 1 stock × 1 year | ~550 | 10 min |
| 7 | Announcement PDFs (important only) | Oct 2022 → today | 1 document | several thousand | 1–3 hours |
| 8 | Insider trades, pledging, shareholding | 2016 → today | 1 stock (× year for insider) | ~650 | 10 min |
| 9 | FII/DII flows | from now (daily) + history | 1 day | 1 a day | — |
| 10 | FRED macro | 2016 → today | 1 series | 5 | seconds |
| 11 | RBI policy decisions | 2016 → today | by hand | — | 1–2 hours |
| 12 | GDELT news | Oct 2022 → today | 1 15-minute file (or 1 query) | ~141,000 files | 1–2 days of downloading |
| 13 | Daily update job | from now on | — | ~10 a day | minutes |

**Start the FII/DII daily job (step 9) immediately**: NSE only gives the latest day, so its history grows one day at a time.

**Pilot first:** run every step for 5 stocks (RELIANCE, HDFCBANK, INFY, TCS, LT) before the full run.

---

## Step 1: Universe

Download the Nifty 50 list (and the Nifty 200 list if that option is used). Columns: `Company Name, Industry, Symbol, Series, ISIN Code`.

Add by hand and save as `data/manual/universe.csv`:

| Column | Example | Why |
|---|---|---|
| `tier` | `nifty50` or `nifty200` | Which universe the stock belongs to |
| `aliases` | `infosys; infosys ltd` | Matching company names in news (step 12); Nifty 50 only |
| `data_start` | `2023-08-21` for JIOFIN | Some stocks listed recently |
| `old_symbols` | `ZOMATO` for ETERNAL; `TATAMOTORS` for TMPV | Price files use the symbol in force on that day |

Known special cases: **JIOFIN** (listed 2023 after the Reliance demerger), **ETERNAL** (renamed from Zomato), **TMPV** (Tata Motors after its 2025 demerger; earlier history is under `TATAMOTORS` and needs a demerger price adjustment). Find others by checking every stock for gaps after step 2.

---

## Step 2: Daily prices (NSE bhavcopy)

One zip file per trading day containing every NSE security (~3,100 rows, ~2,260 equity stocks). One tracker task per weekday from 2016-01-01 to today; a `404` on a holiday is `no_data` (the tracker checks the trading calendar).

**NSE changed the file format in July 2024. Two URL patterns and two parsers are needed:**

| Dates | URL |
|---|---|
| Up to **5 Jul 2024** | `https://nsearchives.nseindia.com/content/historical/EQUITIES/{YYYY}/{MON}/cm{DD}{MON}{YYYY}bhav.csv.zip` (MON in capitals, e.g. `JAN`) |
| From **8 Jul 2024** | `https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip` |

**Column mapping to our `prices_daily` table**

| Our column | Old format | New format |
|---|---|---|
| `symbol` | `SYMBOL` | `TckrSymb` |
| `series` | `SERIES` | `SctySrs` |
| `trade_date` | `TIMESTAMP` | `TradDt` |
| `open` / `high` / `low` / `close` | `OPEN` / `HIGH` / `LOW` / `CLOSE` | `OpnPric` / `HghPric` / `LwPric` / `ClsPric` |
| `prev_close` | `PREVCLOSE` | `PrvsClsgPric` |
| `volume` | `TOTTRDQTY` | `TtlTradgVol` |
| `turnover` | `TOTTRDVAL` | `TtlTrfVal` |
| `trades` | `TOTALTRADES` | `TtlNbOfTxsExctd` |
| `isin` | `ISIN` | `ISIN` |

**Keep** rows with series `EQ` whose symbol (or old symbol) is in the universe: 50 stocks, or 200 with the Nifty 200 option. The extra stocks cost no extra downloads.

```python
from datetime import date

def bhavcopy_url(d: date) -> str:
    if d <= date(2024, 7, 5):
        mon = d.strftime("%b").upper()
        return (f"https://nsearchives.nseindia.com/content/historical/EQUITIES/"
                f"{d:%Y}/{mon}/cm{d:%d}{mon}{d:%Y}bhav.csv.zip")
    return (f"https://nsearchives.nseindia.com/content/cm/"
            f"BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")
```

**Save:** raw zip → `data/raw/nse_bhavcopy/{YYYY}/`; parsed rows → `data/lake/prices_daily/year={YYYY}/`.
**`published_at`:** trade date 18:00 IST (the file appears after the close), so day `d`'s bar is first usable for the forecast on the next trading day.
**Size:** ~300 MB of raw zips; ~130,000 rows for 50 stocks (~520,000 for 200).

**Backup source:** `yfinance` (`yf.download("INFY.NS", start="2016-01-01", auto_adjust=False)`). Use it only to fill gaps and cross-check closes; the tracker records `fallback_source = "yfinance"`. Stooq now blocks scripted downloads, so it can't be the backup.

---

## Step 3: Nifty 50 index and India VIX

**URL (one file per day, same loop as step 2):**
`https://nsearchives.nseindia.com/content/indices/ind_close_all_{DDMMYYYY}.csv`

**Columns:** `Index Name, Index Date, Open Index Value, High Index Value, Low Index Value, Closing Index Value, Points Change, Change(%), Volume, Turnover (Rs. Cr.), P/E, P/B, Div Yield`

**Keep** rows `Nifty 50` and `India VIX`. The Nifty P/E and P/B columns are a free valuation feature.

**Dates:** store the trading day from the file name, not the `Index Date` column. Most files write `Index Date` as DD-MM-YYYY, but the files for 6, 10 and 11 Apr 2023 write MM-DD-YYYY (`04-10-2023` = 10 Apr). The parser accepts either reading and blocks a file whose date matches neither.

**Missing file:** NSE has no file for 20 Jun 2016, a trading day (404 on both archive hosts). That day's Nifty 50 and India VIX come from yfinance (`^NSEI`, `^INDIAVIX`), with `source = 'yfinance'`.

**Save:** `data/raw/nse_indices/{YYYY}/` → `data/lake/index_daily/`.

---

## Step 4: Corporate actions (splits, bonuses, dividends)

**URL:** `https://www.nseindia.com/api/corporates-corporateActions?index=equities&symbol={SYMBOL}&from_date=01-01-2016&to_date={DD-MM-YYYY}`

**Fields:** `symbol, subject, exDate, recDate, faceVal, isin, bcStartDate, bcEndDate`

`subject` is free text, for example `Bonus 1:1`, `Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share`, `Dividend - Rs 20 Per Share`. Parse it with regular expressions into `type` (split, bonus, dividend, rights, demerger) and `ratio` or `amount`. Check every split, bonus and demerger by hand (a few hundred rows for 50 stocks).

**Price adjustment:** on the ex-date, multiply all earlier prices by a factor (split ₹10 → ₹2: × 0.2; bonus 1:1: × 0.5) and divide earlier volumes by it. Keep the raw prices and store the factors in their own table; compute adjusted prices in `derived/`.

**Check:** after adjustment, no daily move above 20% should remain unexplained.

---

## Step 5: Quarterly results and board meetings

**Results list URL:** `https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol={SYMBOL}&period=Quarterly`

**Fields:** `toDate` (quarter end), `consolidated` (Consolidated / Non-Consolidated), `exchdisstime` (time it became public), `audited`, `financialYear`, `params`.

What testing showed (Infosys):
- 144 filings going back to 2005, standalone and consolidated. Use consolidated where it exists.
- **The list has no revenue or profit numbers**, only the filing record. The numbers must come from the detailed filing or XBRL file (endpoint to be found in the pilot) or from the results PDF (step 7).
- **The list stops at the Dec 2024 quarter** (published 16 Jan 2025). Later results are probably under NSE's new integrated-filing pages. Finding them is a pilot task.

**Board meetings URL:** `https://www.nseindia.com/api/corporate-board-meetings?index=equities&symbol={SYMBOL}`

**Fields:** `bm_date` (meeting date), `bm_purpose`, `bm_desc`, `bm_timestamp` (when announced). A board meeting "to consider financial results" tells us the results date in advance; this feeds the catalyst calendar. The test returned the 20 most recent meetings; check whether `from_date`/`to_date` parameters return older ones.

---

## Step 6: Announcements list

**URL:** `https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={SYMBOL}&from_date={DD-MM-YYYY}&to_date={DD-MM-YYYY}`

One tracker task per stock per year (50 stocks × 11 years ≈ 550 tasks). BSE's announcements API returned "Access Denied" in testing, so NSE is the only source.

**Fields we keep**

| Field | Our use |
|---|---|
| `seq_id` | Unique ID → `doc_id = "nse:{seq_id}"` |
| `symbol` | Stock |
| `desc` | Category, e.g. Press Release, Financial Result Updates, Updates |
| `attchmntText` | One-paragraph summary. Useful text even without the PDF |
| `attchmntFile` | Link to the PDF or zip |
| `an_dt`, `exchdisstime` | Time it became public (IST) → `published_at` in UTC |
| `hasXbrl` | Whether a structured data file exists |

**Volume:** Infosys had 281 announcements in 2024 but only 14 in Jan–Mar 2016, so older years are much thinner.

**Important vs routine:** 184 of Infosys's 281 announcements in 2024 had the generic category `Updates`, and many others are routine (newspaper publication, loss of share certificates). Tag each announcement `is_important` using:
1. Category: Financial Result Updates, Outcome of Board Meeting, Press Release, Acquisition, Credit Rating, change in directors or management, award or receipt of orders, SEBI takeover disclosures.
2. Keywords in `attchmntText` for the `Updates` category (order, contract, acquisition, rating, resignation, appointment, fire, plant, USFDA, penalty and so on).

The full list (2016 → today) feeds the event impact memory; only important items get their PDFs downloaded.

**Save:** `data/raw/nse_api/announcements/{SYMBOL}/{YYYY}.json` → `data/lake/announcements/`.

---

## Step 7: Announcement PDFs (important ones only, Oct 2022 → today)

The agents only read documents from the validation and test periods (plus a 3-month lookback), so full text is needed only from about Oct 2022. One tracker task per document.

1. Download `attchmntFile` (some links are `.pdf`, some `.zip` containing a PDF) → `data/raw/nse_pdfs/{SYMBOL}/{seq_id}.pdf`.
2. Extract text with `pdfplumber` → `data/lake/announcement_text/` (`doc_id, text, n_pages, n_chars`).
3. If a PDF has almost no text, it is a scanned image. The tracker marks the task `success` with a `scanned_pdf` review item (severity info); OCR can be added later if many important ones are scanned.

---

## Step 8: Insider trades, pledging, shareholding, bulk deals

| Data | URL | Main fields |
|---|---|---|
| Insider trades | `https://www.nseindia.com/api/corporates-pit?index=equities&symbol={SYMBOL}&from_date=..&to_date=..` | `acqName, personCategory` (Promoter, Director…), `acqMode` (Market Purchase, Market Sale, Pledge…), `buyValue, sellValue, befAcqSharesPer, afterAcqSharesPer, acqfromDt, intimDt` (date disclosed = public time) |
| Pledging | `https://www.nseindia.com/api/corporate-pledgedata?index=equities&symbol={SYMBOL}` | Promoter shares pledged (check how much history it returns) |
| Shareholding | `https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol={SYMBOL}` | Quarterly promoter %, public % |
| Bulk deals | `https://nsearchives.nseindia.com/content/equities/bulk.csv` | **Today only.** The history endpoint was blocked in testing. Collect daily from now on, or leave bulk deals out (decided after the pilot). |

---

## Step 9: FII/DII flows

| Part | Source | Status |
|---|---|---|
| Daily, from now on | `https://www.nseindia.com/api/fiidiiTradeReact` → `category` (FII/FPI, DII), `buyValue, sellValue, netValue` (₹ crore), `date` | Works, but **returns only the latest day**. Start the daily job now to build our own history |
| FII history | NSDL FPI reports: `https://www.fpi.nsdl.co.in/web/Reports/Archive.aspx` | Page reachable; file format to be checked in the pilot |
| DII history | No confirmed free source yet | If none is found, use FII flows only in training |

`published_at`: NSE posts provisional figures in the evening, so day `d`'s flows are usable on the next trading day.

---

## Step 10: FRED macro series

**URL:** `https://api.stlouisfed.org/fred/series/observations?series_id={ID}&api_key={KEY}&file_type=json&observation_start=2016-01-01`

Returns `date` and `value` (a `.` value means missing). One tracker task per series.

| Series | Meaning | How FRED publishes it |
|---|---|---|
| `DGS10` | US 10-year Treasury yield | Daily |
| `VIXCLS` | US VIX | Daily |
| `DEXINUS` | USD/INR | Weekly (Fed H.10 release) |
| `DTWEXBGS` | Broad dollar index | Weekly (H.10) |
| `DCOILBRENTEU` | Brent crude | Weekly (EIA) |

**`known_from`:** all five are market prices. The value for day `d` was publicly known at that day's US close, so it is usable from the next day: `known_from` = `d` + 1 day at 00:00 UTC (05:30 IST), before our 08:30 IST forecast. The lag is set per series in `config.yaml` (`fred.series.*.lag_days`).

**FRED's weekly publishing does not cause look-ahead in the backtest**, because the underlying price was already public. It matters for the **live daily job**: the last few days of the weekly series are not on FRED yet. For live runs, take USD/INR and Brent from a daily source (for example yfinance `INR=X` and `BZ=F`). Revisable economic releases (CPI, GDP) would be different: for those, use the first-release date.

---

## Step 11: RBI policy decisions (by hand)

Only about 65 decisions since 2016, so a hand-made CSV is faster and more reliable than a scraper.

**File:** `data/manual/rbi_policy.csv`

| Column | Example |
|---|---|
| `decision_date` | 2025-02-07 |
| `announce_time_ist` | 10:00 |
| `repo_rate_before`, `repo_rate_after` | 6.50, 6.25 |
| `stance` | neutral |
| `source_url` | link to the RBI press release |

Source: RBI monetary policy statements on rbi.org.in. Decisions are usually announced at 10:00 IST, during market hours, so a decision on day `d` is first usable for the forecast on day `d+1`.

---

## Step 12: GDELT news (secondary source)

**What GDELT is:** a free dataset, not a storage service. Every 15 minutes it reads news websites worldwide and publishes a file describing each article: time, source, URL, headline, organisations mentioned, tone and themes. It does **not** include the article text. Its full history is also hosted on Google BigQuery.

**Role in this project:** NSE announcements are the main evidence. GDELT adds headlines, news volume, tone, and sector and macro news that never appears in company filings.

**Example row (30 Oct 2026, 11:00 UTC)**

| Field | Value |
|---|---|
| Source | livemint.com |
| Title | SC appoints former judge to mediate $2.81 billion Reliance-Centre KG-D6 gas dispute |
| Organisations | reliance industries; supreme court; british petroleum; delhi high court |
| Tone | −1.46 (positive words 2.6%, negative words 4.0%) |
| Themes | TRIAL; JUDGE; EPU_POLICY_REGULATORY |

In those two 15-minute files there were 1,870 articles worldwide, 8 of them about Indian companies or markets.

### Three ways to get GDELT data

| Route | What it is | Good for | Limits |
|---|---|---|---|
| **Raw 15-minute files** | One zip per 15 minutes, listed in a master file with size and checksum | History, free, no account, no quota | ~141,000 files ≈ **761 GB** for Oct 2022 → today (avg ~4.5 MB each); stream, filter and delete |
| **BigQuery** | Google hosts all GDELT history; query with SQL | History; returns only matching rows | Free up to 1 TB scanned a month; always dry-run first |
| **DOC 2.0 API** | Search API | Last ~3 months; quick checks | 1 request every 5 seconds; rate-limited us twice during testing |

The route for history is chosen after the pilot. The raw-files route needs no Google account.

### Route A: raw 15-minute files

1. **List all files:** `https://data.gdeltproject.org/gdeltv2/masterfilelist.txt` (~129 MB). Each line is `size md5 url`; keep lines ending in `.gkg.csv.zip` from `20221001000000` onward.
2. **Plan one tracker task per file**, storing the expected size and MD5 so every download can be verified.
3. **Download:** `http://data.gdeltproject.org/gdeltv2/{YYYYMMDDHHMMSS}.gkg.csv.zip`. The newest file can return `404` for a few minutes after it is listed: a retryable error.
4. **Filter:** the file is tab-separated with no header. Keep rows where a company alias appears in the organisations column or the page title:

   | Column index | Field | Use |
   |---|---|---|
   | 1 | `DATE` (YYYYMMDDHHMMSS, UTC) | `published_at` (time GDELT saw the article) |
   | 3 | `SourceCommonName` | Source domain |
   | 4 | `DocumentIdentifier` | URL |
   | 8 | `V2Themes` | Themes |
   | 13 | `Organizations` | Company matching |
   | 15 | `V2Tone` | tone, positive %, negative %, … |
   | 26 | `Extras` | Contains `<PAGE_TITLE>…</PAGE_TITLE>` (headline) |

5. **Save** only matching rows → `data/raw/gdelt/{YYYY}/{MM}/{file}.parquet`, then **delete the zip**. This is the one exception to "keep every raw file": GDELT files are permanent and can be re-downloaded any time from the master list, and the tracker keeps each file's name and MD5.

At 5–10 MB/s the full history takes about 1–2 days of continuous downloading; it can run over several nights, since the tracker resumes where it stopped.

### Route B: BigQuery

- Table: `gdelt-bq.gdeltv2.gkg_partitioned`.
- **Always filter on `_PARTITIONTIME`** (date range) and select only the needed columns: `DATE, DocumentIdentifier, SourceCommonName, V2Tone, V2Themes, V2Organizations, Extras`.
- Filter companies with `LOWER(V2Organizations) LIKE '%infosys%'` (one alias list per company).
- **Dry-run first** to see bytes scanned; split history into monthly queries to stay within 1 TB a month. One tracker task per month.

### Route C: DOC 2.0 API (last ~3 months)

`https://api.gdeltproject.org/api/v2/doc/doc?query="Infosys" sourcecountry:IN&mode=artlist&maxrecords=250&format=json&startdatetime={YYYYMMDDHHMMSS}&enddatetime={YYYYMMDDHHMMSS}`

Returns up to 250 articles per request (`url, title, seendate, domain, language, sourcecountry`). Daily tone per company: same query with `mode=timelinetone`. One tracker task per stock per day.

### Company matching and relevance

Short or common names are ambiguous ("TCS", "Titan", "Reliance"). And many articles mention a company without being about it: market summaries mention ten companies, and a BSNL article may mention TCS in passing.

- Match with full names and alias lists from `universe.csv`.
- Store a `match_score` (how sure we are it's the right company) and a `relevance` score: high if the company is in the headline, lower for mentions in the body or long lists of companies.
- Hand-check 100 matched rows and record the precision.

---

## Step 13: Daily update job

After history is loaded, one command runs every evening (around 19:30 IST, after NSE posts its files). It adds today's tasks to the tracker and runs them:

1. Today's bhavcopy and index file
2. Announcements for the last 2 days (catches late posts)
3. FII/DII for today
4. Board meetings, insider trades
5. GDELT files for the day
6. FRED (weekly)

Then it rebuilds the coverage table and prints the review list. Schedule it with Windows Task Scheduler or cron on one team member's machine. Cloud runners (for example GitHub Actions) may be blocked by NSE; test before relying on one.

---

## T. Download tracker

Every download goes through one tracker, stored in SQLite (`data/ingest.sqlite`). SQLite suits the many small status updates; the market data itself stays in Parquet + DuckDB.

**Goal:** when something fails, skip that task, keep going, and check it later.

### Tables

```
sources ──┐
          ├──< tasks >──< attempts >── runs
error_types┘     │
                 └──< review_items
trading_calendar   (tells holidays from real gaps)
coverage           (per stock, per day, per source; built after parsing)
```

| Table | Job |
|---|---|
| `sources` | Each data source and its retry rules |
| `error_types` | Each kind of error, and whether retrying it can help |
| `runs` | Each time a collector runs (backfill, daily job, manual retry) |
| `tasks` | The current state of each unit of work |
| `attempts` | Every try, never overwritten: the full error history |
| `review_items` | The "verify later" list: failures and data-quality problems, and what a person decided |
| `trading_calendar` | NSE trading days |
| `coverage` | What we have for each stock on each trading day |

### Task statuses

```
                 ┌────────────── retryable error, attempts left ──────────────┐
                 ▼                                                              │
pending ──► running ──► success                                                 │
                    ├─► no_data        (holiday / nothing published)            │
                    ├─► retry_wait ─────────────────────────────────────────────┘
                    └─► needs_review   (retries used up, OR an error that retrying
                            │           can't fix, OR a serious data-quality problem)
                            │  person decides:
                            ├─► pending       (retry now)
                            ├─► verified_gap  (data really doesn't exist; accepted)
                            ├─► success       (filled from a backup source, e.g. yfinance)
                            └─► skipped       (deliberately ignored, with a note)
```

### Rules

1. **An error never stops the run.** Record the attempt, set the status, move on to the next task.
2. **If retrying can help** (timeout, rate limit, server error, temporary block), wait 5 min, then 10, then 20. After 3 attempts the task goes to `needs_review`.
3. **If retrying can't help** (file format changed, parser crashed, file missing on a trading day), the task goes straight to `needs_review`.
4. **A 404 on a holiday** becomes `no_data` automatically, using the trading calendar.
5. **Download and parse are separate stages.** If the parser had a bug, fix it and re-run only the parse step from the saved raw file. Nothing is downloaded again.
6. **Quality checks run after parsing.** A serious problem (for example, a Nifty stock missing from the day's file) → `needs_review`, and the data is not written to the lake yet. A minor problem (for example, a scanned PDF) → `success` plus a review item.
7. **After a crash,** tasks left in `running` are reset to `pending` when the next run starts.

### Schema

```sql
-- Data sources and their retry rules
CREATE TABLE sources (
    source          TEXT PRIMARY KEY,               -- 'nse_bhavcopy', 'gdelt_raw', ...
    task_unit       TEXT NOT NULL,                  -- 'day', 'symbol_year', 'symbol', 'document', 'file_15min', 'series'
    max_attempts    INTEGER NOT NULL DEFAULT 3,
    retry_base_sec  INTEGER NOT NULL DEFAULT 300,   -- 5 min, then 10, 20 ...
    description     TEXT
);

-- Error categories: retry automatically, or send for review
CREATE TABLE error_types (
    error_type   TEXT PRIMARY KEY,   -- 'timeout', 'http_403', 'http_429', 'http_5xx', 'http_404',
                                     -- 'empty_response', 'checksum_mismatch', 'parse_error',
                                     -- 'schema_changed', 'validation_failed'
    retryable    INTEGER NOT NULL,   -- 1 = retry automatically, 0 = straight to review
    description  TEXT
);

-- Each time a collector runs
CREATE TABLE runs (
    run_id        INTEGER PRIMARY KEY,
    source        TEXT NOT NULL REFERENCES sources(source),
    triggered_by  TEXT NOT NULL,      -- 'backfill', 'daily_job', 'manual_retry'
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL DEFAULT 'running',   -- running, completed, crashed
    machine       TEXT,
    code_version  TEXT,               -- git commit
    n_success     INTEGER NOT NULL DEFAULT 0,
    n_no_data     INTEGER NOT NULL DEFAULT 0,
    n_failed      INTEGER NOT NULL DEFAULT 0
);

-- One row per unit of work: its CURRENT state
CREATE TABLE tasks (
    task_id          INTEGER PRIMARY KEY,
    source           TEXT NOT NULL REFERENCES sources(source),
    task_key         TEXT NOT NULL,                   -- '2024-06-03', 'INFY|2024', 'nse:12345', '20221001000000'
    symbol           TEXT NOT NULL DEFAULT '*',       -- '*' = all stocks
    period_start     TEXT NOT NULL,                   -- date, or datetime for 15-minute GDELT files
    period_end       TEXT NOT NULL,
    url              TEXT,
    expected_bytes   INTEGER,                         -- from GDELT's master list
    expected_md5     TEXT,
    stage            TEXT NOT NULL DEFAULT 'download', -- download → parse → validate → done
    status           TEXT NOT NULL DEFAULT 'pending',  -- pending, running, retry_wait, needs_review,
                                                       -- success, no_data, verified_gap, skipped
    attempts         INTEGER NOT NULL DEFAULT 0,
    next_retry_at    TEXT,
    last_attempt_id  INTEGER,
    rows_saved       INTEGER,
    raw_path         TEXT,
    fallback_source  TEXT,             -- e.g. 'yfinance' if filled from a backup
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    UNIQUE (source, task_key)        -- one key per unit of work; a stock can have many documents on one day
);
CREATE INDEX idx_tasks_due ON tasks (source, status, next_retry_at);

-- Every try, never overwritten
CREATE TABLE attempts (
    attempt_id     INTEGER PRIMARY KEY,
    task_id        INTEGER NOT NULL REFERENCES tasks(task_id),
    run_id         INTEGER NOT NULL REFERENCES runs(run_id),
    stage          TEXT NOT NULL,       -- download, parse, validate
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    outcome        TEXT NOT NULL,       -- success, no_data, error
    http_status    INTEGER,
    error_type     TEXT REFERENCES error_types(error_type),
    error_message  TEXT,
    bytes          INTEGER,
    rows           INTEGER,
    duration_ms    INTEGER
);

-- The "verify later" list
CREATE TABLE review_items (
    review_id    INTEGER PRIMARY KEY,
    task_id      INTEGER NOT NULL REFERENCES tasks(task_id),
    reason       TEXT NOT NULL,       -- 'max_attempts', 'non_retryable_error', 'quality_check', 'coverage_gap'
    check_name   TEXT,                -- 'missing_symbol', 'price_jump_20pct', 'scanned_pdf', ...
    severity     TEXT NOT NULL,       -- info, warn, error
    details      TEXT,                -- JSON: which stock, which value
    status       TEXT NOT NULL DEFAULT 'open',   -- open, resolved
    decision     TEXT,                -- retry, accept_gap, use_fallback, fixed_parser, ignore
    reviewed_by  TEXT,
    reviewed_at  TEXT,
    note         TEXT,
    created_at   TEXT NOT NULL
);

-- NSE trading days
CREATE TABLE trading_calendar (
    trade_date  TEXT PRIMARY KEY,
    is_trading  INTEGER NOT NULL,
    note        TEXT                  -- 'Holi', 'Diwali Muhurat trading', ...
);

-- What we have for each stock on each trading day (rebuilt after parsing)
CREATE TABLE coverage (
    symbol      TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    source      TEXT NOT NULL,        -- 'prices', 'announcements', 'news', ...
    n_records   INTEGER NOT NULL,
    status      TEXT NOT NULL,        -- ok, empty (normal: no news that day), gap (expected but missing)
    PRIMARY KEY (symbol, trade_date, source)
);
```

Coverage rules: prices are expected on every trading day from a stock's `data_start` (or, if that is empty, from the first day it appears in the data), so a missing price is a `gap`. The bhavcopy quality check already puts each day's missing stocks on the review list (`missing_symbol`). Zero announcements or zero news on a day is normal (`empty`), not a gap.

### Example: three days in a bhavcopy backfill

| Day | What happened | Final status | Review list |
|---|---|---|---|
| 2019-03-04 | 404, but the calendar says it was a holiday (Mahashivratri) | `no_data` | — |
| 2020-06-15 | 403 blocked → retried at 5 / 10 / 20 min, still blocked. The run continued with 2020-06-16 in the meantime | `needs_review` | "max_attempts", with all 3 errors in `attempts` |
| 2024-07-08 | Download fine, parser crashed (new file format) | `needs_review` immediately | "non_retryable_error: schema_changed" |

Later, in review: 2020-06-15 → retry → `success`. 2024-07-08 → fix the parser, decision "fixed_parser" → only the parse step re-runs from the saved file → `success`.

### Commands

The code is in `src/ingest/` (one collector per source in `src/ingest/sources/`).

```
python -m src.ingest init                                               # tracker database + trading calendar
python -m src.ingest universe                                           # data/manual/universe.csv from NSE lists
python -m src.ingest sources                                            # list all sources
python -m src.ingest pilot                                              # every source, 5 stocks, June 2024
python -m src.ingest plan    --source nse_bhavcopy --from 2016-01-01   # create pending tasks
python -m src.ingest run     --source nse_bhavcopy                      # run; failures are skipped
python -m src.ingest status                                             # progress per source
python -m src.ingest review                                             # the "verify later" list
python -m src.ingest resolve 123 --decision accept_gap --note "NSE file missing; checked website"
python -m src.ingest reparse --source nse_bhavcopy --from 2024-07-08   # after fixing a parser
python -m src.ingest coverage --symbol INFY --from 2024-06-01           # per stock, per day
python -m src.ingest views                                              # DuckDB views over the lake
python -m src.ingest daily                                              # evening update job
```

Decisions for `resolve`: `retry` (download again), `fixed_parser` (re-parse the saved file), `accept_gap` (data really doesn't exist), `use_fallback` (fill from yfinance; prices only), `ignore`, and `acknowledge` (close a warning on a task that succeeded).

### Useful queries

Review list, with each task's last error:

```sql
SELECT t.source, t.symbol, t.period_start, t.stage, t.attempts,
       a.error_type, a.error_message, r.reason, r.check_name
FROM tasks t
JOIN attempts a          ON a.attempt_id = t.last_attempt_id
LEFT JOIN review_items r ON r.task_id = t.task_id AND r.status = 'open'
WHERE t.status = 'needs_review'
ORDER BY t.source, t.period_start;
```

Known gaps, so the backtest knows which days have missing data:

```sql
SELECT symbol, trade_date, source FROM coverage WHERE status = 'gap';
```

Progress per source:

```sql
SELECT source, status, COUNT(*) FROM tasks GROUP BY source, status;
```

---

## Checks after each step

| Data | Check |
|---|---|
| Prices | Coverage shows no `gap` after each stock's `data_start`; no duplicates; no zero volume; no unexplained jump above 20% after adjustment |
| Index | Nifty 50 and India VIX present on every trading day |
| Corporate actions | Every split, bonus and demerger checked by hand |
| Announcements | Count per stock per year looks reasonable; every `published_at` has a time, not just a date |
| PDFs | Share of scanned (no-text) PDFs recorded |
| FRED | `known_from` is never before the date of the value |
| GDELT | Every downloaded file matches its MD5; precision from the 100-row hand check recorded |
| Tracker | Review list is empty or every open item has a note |

---

## Open items (solve in the pilot)

1. **Results numbers:** the results list has no revenue or profit figures, and it stops at the Dec 2024 quarter. Find the detailed or XBRL endpoint and where 2025+ results are published.
2. **FII history:** check the NSDL report format. **DII history:** find a source or drop it.
3. **Bulk deals history:** blocked in testing; collect daily or drop.
4. **Symbol changes:** ZOMATO → ETERNAL, TATAMOTORS → TMPV (with demerger adjustment); look for others by checking every stock for gaps.
5. **GDELT history route:** raw files (~761 GB to stream) or BigQuery (measure bytes scanned per month with a dry run).
6. **Board meetings history:** check whether date parameters return meetings older than the latest 20.
7. **Nifty 200 prices:** collect or not, based on whether the LSTM needs more training data.
8. **News aliases:** the default aliases in `universe.csv` are formal company names (for example "adani ports and special economic zone"); GDELT often uses shorter names. Review them before the full GDELT run.

9. **Promoter block sales:** the insider (PIT) endpoint returned nothing for TCS in 2023–2024, although Tata Sons sold TCS shares in March 2024. Such sales are disclosed under SEBI's takeover regulations (SAST), which NSE lists separately; consider adding that endpoint.
10. **Very long PDFs:** important announcements include annual reports (one Infosys PDF in the pilot had 298 pages, ~1 million characters). Decide whether to cap pages or skip annual reports before the full PDF run.
11. **Pledging endpoint returned nothing:** on Sun 8 Nov 2026, `corporate-pledgedata` returned an empty list for all 50 stocks and for every parameter tried, although the pilot on Fri 30 Oct got data. Re-run on a weekday (`plan --source nse_pledges --refresh`, then `run`). If it is still empty, find the new endpoint.
12. **Insider data thins out after 2021:** rows per year drop from ~19,000 (2016) to ~1,500 (2025), and only 8 of 50 stocks have any 2026 rows. Check whether NSE publishes newer disclosures elsewhere (for example the system-driven disclosures) before relying on insider features in the test period.
13. **Rights issues and demergers:** 15 events in the Nifty 50 (for example Reliance 2020 rights and 2023 demerger, ITC 2025 demerger, HUL 2025 demerger, Tata Motors 2025 demerger) need a price adjustment factor worked out by hand. Splits and bonuses are adjusted automatically. See `review` (check `manual_adjustment_needed`).

Resolved: FRED `known_from` (all five series are market prices, usable the next day; see step 10).

**Backfill result (8 Nov 2026, Nifty 50):** prices 2,687 trading days (1 Jan 2016 → 6 Nov 2026), index 2,687 days, corporate actions 49 stocks, results 48, board meetings 50, announcements 524 stock-years with data, insider 346 stock-years, shareholding 50. Fixed during the run: one bhavcopy (13 Jul 2020) writes a 2-digit year; three index files write the date month-first; the 20 Jun 2016 index file is missing (filled from yfinance). JIOFIN has no EQ rows for its first 10 days (21 Aug – 1 Sep 2023) because it traded in the BE series.

**Pilot result (30 Oct 2026, 5 stocks, June 2024):** every source ran end to end. Prices 950/950 stock-days, index 19/19 days (17 Jun holiday → `no_data`), corporate actions parsed correctly (splits and bonuses; Reliance's 2020 rights issue and 2023 demerger flagged for manual adjustment), 707 announcements for Jan–Jun 2024 (144 important), 14 PDFs extracted (none scanned), 8 GDELT files filtered. FRED waits for an API key.
