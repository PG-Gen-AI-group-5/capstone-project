# Prompt templates

Copy-paste prompts for working on this project with Claude Code. Claude Code reads [CLAUDE.md](../CLAUDE.md) automatically, so these prompts leave out the project rules (the time rule, periods, prices, conventions). If you use another tool, paste CLAUDE.md at the start of the chat.

## How to work with the agent

1. **One task per session.** One work-plan task, or one notebook, per session. Start a fresh session for the next one.
2. **Fill every `[BRACKET]`** and delete lines that don't apply.
3. **Plan first.** Each template asks for a plan and waits. Correct the plan before any code is written; that's where mistakes are cheapest to fix.
4. **Check before the pull request.** Review the diff, run `pytest` and `ruff check src tests`, and run the notebook on a fresh kernel. Then use the [review template](#review-before-a-pull-request).
5. **Keep a record.** Paste the agent's final report into the pull request description.

Contents:

- [Any work-plan task](#any-work-plan-task)
- [EDA](#eda) (E1–E4 notebooks, then the report)
- [Model training](#model-training) (prerequisites, one model at a time, comparison)
- [Pipeline framework](#pipeline-framework) (core, point-in-time store, stage runner, walking skeleton)
- [Review before a pull request](#review-before-a-pull-request)

---

## Any work-plan task

Use this template for any task in WORK_PLAN.md that doesn't have its own template below.

```
Task: [ID(s) and name from WORK_PLAN.md, e.g. "2.4 Corporate actions: adjustment factors"]

Goal: [one sentence: what should exist when this is done]

Read first:
- WORK_PLAN.md, task [ID] and its "done when"
- docs/ARCHITECTURE.md, the section on [layer / interface]
- [existing files this builds on, e.g. src/ingest/sources/nse_corporate.py]

Inputs: [tables or files it reads]
Outputs: [tables, files or functions it creates, and where]
Already decided: [settled choices, e.g. "split and bonus adjustment only, no dividend adjustment"]
Scope: this task only; [later tasks that will build on it] come later.

First propose a plan: files to create or change, function signatures, output schema, tests,
and any decision you need from me. Wait for my OK, then implement it with tests.

Done when:
- [the task's "done when", made checkable, e.g. "no adjusted series moves more than 20% on an
  ex-date; the check prints the count of such days, and it is 0"]
- pytest and ruff pass

Report back: what was built, the evidence for each done-when item (command + output or number),
and anything left open.
```

---

## EDA

Run the four notebooks in order, then write the report.

- **E1** needs nothing new and can start today.
- **E2–E4** are best run on adjusted prices (task 2.4). Until those exist, each template says how to work around it.

### E1: Data inventory and quality

```
Task: EDA notebook 1, data inventory and quality: notebooks/eda/01_data_inventory.ipynb

Read first: docs/DATA_COLLECTION.md (lake tables, "Backfill result", "Open items"), the output of
`python -m src.ingest status` and `python -m src.ingest review`.
Data access: the DuckDB views in data/market.duckdb (run `python -m src.ingest views` if missing).
Period: the full range; this notebook checks coverage and quality.

Build:
1. Inventory: one row per table in data/lake: rows, stocks, first and last date, the date and
   time columns, share of nulls in each key column, source.
2. Coverage heat maps, stock × year, for prices, announcements, insider trades, shareholding and
   results; trading-day coverage for index_daily and each macro series.
3. Tracker summary: tasks per source and status from data/ingest.sqlite, and the open review items.
4. Timestamps: share of announcements and results that carry a time (not just a date); time of
   day in IST split into before 08:30, 08:30–15:30, after 15:30. Records before 08:30 IST can
   inform that day's trade; later ones inform the next trading day's.
5. Gap reconciliation: every gap or oddity found, each matched to a known item (open items,
   backfill notes, review list) or marked NEW.

Save figures to report/figures/eda/01_*.png. End with a "Findings" cell.
First propose the notebook outline and wait for my OK.

Done when: every lake table has an inventory row; every gap is matched to a known item or marked
NEW; the notebook runs top to bottom on a fresh kernel.
Report back: the findings list and every NEW gap.
```

### E2: Prices, returns and regimes

```
Task: EDA notebook 2, prices, returns and regimes: notebooks/eda/02_prices_returns.ipynb

Read first: notebooks/eda/01_data_inventory.ipynb findings, docs/ARCHITECTURE.md (analysis and
features layers).
Data access: DuckDB views. Prices: data/derived/prices_adj if it exists; otherwise raw closes with
each stock's returns dropped on every ex-date in corporate_actions. Say which in the first cell.
Period: 2016-01-01 to 2023-12-31 (train + validation).

Analyse the 50 stocks and the Nifty 50 index:
1. Return distributions at 1, 5 and 10 trading days: mean, std, skew, kurtosis, tails vs normal.
2. Base rates: share of up moves at 1, 5 and 10 days, per year and per stock. This is the class
   balance for the direction labels and the accuracy a constant guess achieves.
3. Autocorrelation of returns and of absolute returns, lags 1–20: linear predictability (what
   ARIMA can use) and volatility clustering.
4. Stationarity: ADF test on log prices vs log returns.
5. Overnight gap (open vs previous close) vs intraday move. Trades enter at the open, so gap size
   decides how often stops are skipped over.
6. Volatility: rolling 20-day realised volatility; how well the last 20 days predict the next 5
   and 10 days (the historical-volatility baseline, 16.5).
7. Market link: beta and correlation to the Nifty; correlation matrix by sector.
8. Regimes: India VIX level and Nifty trend (above or below its 200-day EMA); returns and
   volatility in each regime.
9. Liquidity: turnover, zero-volume days.
10. Train vs validation: does 2023 resemble 2016–2022 (volatility, base rates, regime mix)?

Functions that later tasks reuse go where docs/ARCHITECTURE.md puts them (regime labels in
src/features/regime.py, return helpers in src/analysis/), with tests.
Save figures to report/figures/eda/02_*.png. End with a "Findings" cell.
First propose the notebook outline and wait for my OK.

Done when: each of the 10 items has a figure or table and a finding with its number; the notebook
runs top to bottom on a fresh kernel; pytest passes.
Report back: the findings list, and what they imply for labels, features and baselines.
```

### E3: Events and announcements

```
Task: EDA notebook 3, events and announcements: notebooks/eda/03_events.ipynb

Read first: E1 and E2 findings, docs/DATA_COLLECTION.md steps 5–8.
Data access: DuckDB views; prices as in E2.
Period: counts over the full range; returns and event studies over 2016–2023.

Analyse:
1. Volume: announcements per year, per stock and per category; share tagged important; the
   most common important categories.
2. Event day: the first trading day each record can inform (published before 08:30 IST on a
   trading day → that day; otherwise the next trading day).
3. Event study: abnormal return = stock return − Nifty return, cumulative from day −5 to +10
   around the event day, for (a) results filings, (b) important announcements by category,
   (c) a random-day control sample of the same size per stock. Show the mean with a 95% interval
   and the share of positive outcomes.
4. Results: time of day they are released; size of the reaction (absolute abnormal return on
   days 0–1) vs non-event days.
5. Volatility: absolute return on event days vs other days.
6. Insider trades and shareholding: coverage per year (it thins after 2021); in 2016–2021,
   whether promoter buying or selling comes before abnormal returns.

Put the event-study function in src/analysis/event_study.py with tests; task 9.1 (event impact
memory) reuses it.
Save figures to report/figures/eda/03_*.png. End with a "Findings" cell.
First propose the notebook outline and wait for my OK.

Done when: items 1–6 each have a figure or table and a finding with its number; the event study
reports intervals and the control sample; the notebook runs top to bottom on a fresh kernel.
Report back: the findings, and which event types move prices beyond the control sample.
```

### E4: Macro, flows and news

```
Task: EDA notebook 4, macro, flows and news: notebooks/eda/04_macro_flows.ipynb

Read first: E1 findings, docs/DATA_COLLECTION.md steps 9–12.
Data access: DuckDB views. Align every macro value on known_from.
Period: 2016-01-01 to 2023-12-31.

Analyse:
1. FRED series (DGS10, VIXCLS, DEXINUS, DTWEXBGS, DCOILBRENTEU): plots, missing days, daily vs
   weekly publication.
2. Lead-lag: correlation of each series' change (known by day d) with the Nifty return on day d
   and over d..d+4.
3. US VIX vs India VIX.
4. RBI policy: if data/manual/rbi_policy.csv has rows, the reaction of the Nifty and bank stocks on
   policy days; otherwise record this item as blocked on task 4.3.
5. FII/DII: rows available, and the history plan (tasks 4.1–4.2).
6. News: if GDELT data covers more than the pilot files, volume and tone per stock; otherwise
   record this item as blocked on tasks 5.1–5.2.

Save figures to report/figures/eda/04_*.png. End with a "Findings" cell.
First propose the notebook outline and wait for my OK.

Done when: items 1–6 each have a figure, a table or a recorded block; the notebook runs top to
bottom on a fresh kernel.
Report back: the findings, and which macro inputs are worth passing to the models and agents.
```

### ER: EDA report

```
Task: write report/eda/EDA_REPORT.md from the EDA notebooks 01–04.

Read first: the "Findings" cell of each notebook and the figures in report/figures/eda/.
Readers: the project team and the capstone reviewers. Use plain language, and give every number
with its unit and period.

Sections:
1. Summary: the 5–8 findings that matter most for modelling, in half a page.
2. Data available: one row per dataset (rows, stocks, period, source, status), plus the coverage
   figure.
3. Data quality: issues found, fixes applied, open gaps and how each affects the project.
4. Price behaviour: returns, base rates, predictability, volatility, regimes.
5. Events: event-study results by type, with intervals.
6. Macro, flows and news: results, or what is blocked and on which task.
7. Implications for modelling: label balance and the accuracy bar to beat; inputs that look
   useful; horizons; whether 2023 is a fair validation year; data work to finish before training.
8. Limitations: survivorship bias (current constituents), the short clean test window, data gaps.

Each number cites its notebook; each figure is embedded with a relative link.
First propose the outline with the key numbers per section and wait for my OK.

Done when: every section is written, or marked blocked with the task that unblocks it; every
figure link resolves; every number traces to a notebook cell.
```

---

## Model training

Order of work:

1. Prerequisites: adjusted prices, features, labels and splits.
2. One session per model: baselines, ARIMA, LSTM, TFT.
3. The comparison notebook.

The notebooks tune and evaluate on validation (2023). Task 16.9 produces the test-period results once, after the models are final.

### T0: Prerequisites (2.4, 10.1, 16.1, 16.2)

```
Task: modelling prerequisites, tasks 2.4 (if not done), 10.1, 16.1, 16.2, and their pipeline
stages. Notebook: notebooks/models/10_labels_splits.ipynb

Read first: WORK_PLAN.md tasks 2.4, 10.1, 16.1, 16.2; docs/ARCHITECTURE.md (store, features,
labels, pipeline, walk-forward); report/eda/EDA_REPORT.md if it exists.
Check what already exists in src/store, src/features, src/labels and data/derived; build only
what is missing.

Label definition: [decide: open(d) → close(d+h−1), which matches the trade entry (recommended),
or close(d−1) → close(d+h−1), the usual time-series convention]. Horizons 1, 5, 10.
Volatility label: std of daily log returns over the horizon.

Build:
- src/store/adjust.py → derived/prices_adj (if missing)
- src/features/indicators.py (10.1) → derived/features_daily
- src/labels/labels.py (16.1) → derived/labels; definitions stored in config.yaml
- src/labels/splits.py (16.2): the walk-forward splitter with the gap rule from ARCHITECTURE.md
- pipeline stages prices_adj, features, labels
- the notebook: label distributions per horizon and period, feature summary, a picture of the
  splits

First propose the plan and wait for my OK.

Done when: tests show (a) features for as_of d use only data public before it, (b) no training
example's label window reaches into its validation or test window, (c) the gap equals the
horizon; the stages rebuild identical outputs on a re-run; the notebook runs top to bottom.
```

### T1: Train one model (repeat per model)

Fill in the model row from the table below the template.

```
Task: [16.x] train and evaluate [MODEL]. Code: src/models/[model].py.
Notebook: notebooks/models/[NN]_[model].ipynb

Read first: docs/ARCHITECTURE.md (model interface, forecast table, walk-forward); WORK_PLAN.md
task [16.x]; report/eda/EDA_REPORT.md section 7 if it exists.

Model notes: [paste this model's row from the table in docs/PROMPTS.md]

Build:
1. src/models/[model].py implementing ForecastModel (fit, predict, save, load), registered in
   src/models/registry.py, with tests on tests/fixtures.
2. The notebook:
   a. Load features and labels for train (2016–2022) and validation (2023) through src/.
   b. Fit with walk-forward refits on the schedule in config.yaml; tune hyperparameters on
      validation; record the search grid and the chosen values.
   c. Predict the validation period as forecast-table rows; save to
      data/derived/forecasts/[model]/.
   d. Evaluate on validation per horizon (1, 5, 10): accuracy, F1, Brier score, volatility MAE
      and RMSE; compare with the constant base rate and the historical-volatility baseline;
      calibration plot.
   e. Save the model and its manifest to data/models/[model]/.

First propose the plan (features used, architecture or order, search grid, refit schedule,
expected run time) and wait for my OK.

Done when: the notebook runs top to bottom on a fresh kernel with a fixed seed; the validation
metrics are saved to report/figures/models/[model]_validation.csv; the forecasts pass the
forecast-table schema check; pytest passes.
Report back: validation metrics vs the base rate and historical volatility, chosen settings,
training time, open issues.
```

| Model | Task | Notes |
|---|---|---|
| Baselines | 16.3–16.5 | Buy-and-hold (Nifty 50 index and each stock). Moving-average crossover (long while EMA 20 > EMA 50). Historical volatility (last 20-day realised volatility as the forecast). No training: they write forecast and signal tables directly. Notebook `11_baselines`. |
| ARIMA | 16.6 | One model per stock on daily log returns. Order chosen by AIC on train from a small grid. Refit monthly; update daily with new observations. h-day forecast = sum of the daily forecasts. `p_up` = P(sum > 0) from the forecast mean and variance. Notebook `12_arima`. |
| LSTM | 16.7 | One model shared across stocks (PyTorch). Input: 60-day window of scaled features (returns, indicators, volume z-score, Nifty return, India VIX); the stock as a learned embedding. Outputs: `p_up` at 1, 5 and 10 days, plus volatility. Scalers fitted on train only; early stopping on validation. Trains on the Nifty 200 if task 2.7 collected it; always evaluated on the Nifty 50. Notebook `13_lstm`. |
| TFT | 16.7b | One model shared across stocks. Static inputs: symbol, sector. Known future inputs: day of week, holidays, F&O expiry dates, results and RBI dates (calendar-only until the catalyst calendar 7.x exists). Observed inputs: as for the LSTM. Quantile outputs → `p_up` and volatility. Its forecasts also feed the Forecaster agent (12.6). Pick pytorch-forecasting or neuralforecast and record why. Notebook `14_tft`. |

### T2: Model comparison

```
Task: notebooks/models/15_model_comparison.ipynb: compare every model on validation (2023).

Read first: docs/ARCHITECTURE.md (forecast table); the manifests in data/models/.
Build: one table of model × horizon × metric (accuracy, F1, Brier, volatility MAE/RMSE) with block
bootstrap 95% intervals (src/evaluation/bootstrap.py, task 18.3: build it if missing); calibration
curves; results by regime (from src/features/regime.py); figures to report/figures/models/.
First propose the outline and wait for my OK.

Done when: every model with a forecasts folder appears in the table; every metric has an interval;
the notebook runs top to bottom on a fresh kernel.
Report back: the ranking per horizon, and which differences fall within the intervals.
```

---

## Pipeline framework

Run P1–P4 in order. P1 needs the whole team to approve the contracts (task 1.6). P4 also needs the LLM choice (1.10).

### P1: Core: config, contracts, logging (1.4, 1.6, 1.8)

```
Task: 1.4 config sections, 1.6 src/contracts.py, 1.8 src/logsetup.py.

Read first: WORK_PLAN.md 1.4, 1.6, 1.8; docs/ARCHITECTURE.md (interfaces); src/config.py; the lake
schemas (run `python -m src.ingest views`, then DESCRIBE each view in data/market.duckdb).

Build:
- config.yaml sections periods, labels, llm (provider, model, temperature 0, cutoff), costs,
  backtest, models (horizons, refit schedule, seeds), with accessors in src/config.py.
- src/contracts.py: the pydantic models listed in task 1.6. Record models mirror the lake columns;
  each record carries published_at or known_from as UTC. TradePlan, CriticVerdict and TraceStep
  follow the fields in WORK_PLAN.md 12.6, 12.8 and 14.1.
- src/logsetup.py: structured lines with a run_id, one file per run under data/logs/; switch
  src.ingest to it.

First propose the model list with fields and types, and wait for my OK.

Done when: every lake table's sample rows validate against their contract; config tests pass;
src.ingest logs through logsetup; pytest and ruff pass.
Report back: the model and field list, formatted for team review.
```

### P2: Point-in-time store (2.4, 6.6, 6.7)

```
Task: src/store: adjusted prices (2.4), as_of readers (6.6), leakage tests (6.7).

Read first: docs/ARCHITECTURE.md (store, point-in-time readers, when data becomes usable);
docs/DATA_COLLECTION.md open item 13; `python -m src.ingest review` (manual_adjustment_needed
items); src/ingest/sources/nse_corporate.py.

Build:
- data/manual/ca_adjustments.csv: one row per rights issue or demerger, with factor, method and
  source note. Rights factor = theoretical ex-rights price / pre-ex close; demerger factor from the
  company's cost split or the price discovered on the ex-date.
- src/store/adjust.py → derived/prices_adj (cumulative factors on OHLC, inverse on volume).
- src/store/db.py and src/store/pit.py with the readers listed in ARCHITECTURE.md.
- tests/fixtures planted with future-dated records; one leakage test per reader.

First propose the plan, and the factor for each of the 15 events with how you derived it; wait
for my OK.

Done when: the jump check reports 0 unexplained moves above 20% on ex-dates; each reader has a
passing leakage test; pytest and ruff pass.
```

### P3: Stage runner

```
Task: src/pipeline: the stage registry and command line from docs/ARCHITECTURE.md.

Build: the Stage definition (name, reads, writes, run function), the registry, manifests,
`python -m src.pipeline run <stage>|all --from --to` and `status`; register every stage whose
code exists, running in the order the stage table gives.
First propose the plan and wait for my OK.

Done when: `python -m src.pipeline run all --from 2023-01-01 --to 2023-03-31` builds every
registered stage; a second run produces identical outputs; `status` lists each stage's last build,
date range and inputs; pytest passes.
```

### P4: Walking skeleton (M3)

```
Task: milestone M3, walking skeleton: 11.1–11.4, 13.1–13.2, 13.4, 14.1–14.2, 19.1–19.2.

Read first: WORK_PLAN.md M3 tasks; docs/ARCHITECTURE.md (agents, interfaces); src/contracts.py.
Build: src/llm (client, cache, budget, versioned prompt files); src/graph with all 7 agents as
stubs that return fixed, valid contract objects; src/run.py; src/tracing/writer.py; a Streamlit
page that lists the day's plans and shows a plan's trace.
First propose the plan and wait for my OK.

Done when: `python -m src.run --date 2024-06-14` on the fixtures writes plans and a trace file
under data/runs/; the dashboard shows both; a test runs the graph end to end on the fixtures.
```

---

## Review before a pull request

```
Review the changes on this branch against main for task [ID]. Read the task's "done when" in
WORK_PLAN.md and the rules in CLAUDE.md.

Check:
- the time rule: every read that feeds a decision goes through an as_of reader and filters on
  published_at or known_from
- the test period stays unused outside the freeze tasks
- adjusted prices for features and models, raw prices for plans and fills
- settings come from config.yaml; data passed between modules uses contracts
- tests cover each done-when item and include a leakage case
- notebooks run top to bottom on a fresh kernel

List the findings, most severe first, with file:line. Then go through each done-when item with
the evidence that it is met, or say why it isn't.
```

Claude Code's `/code-review` command gives a second, general bug-finding pass.
