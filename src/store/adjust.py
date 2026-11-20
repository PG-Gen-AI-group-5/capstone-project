"""Corporate-action adjustment (task 2.4).

An event with factor f on ex-date e means: multiply every price before e by f and divide every
volume before e by f, so the series has no jump on e. The adjusted price on day t is the raw
price times the product of the factors of all events after t.

Where each factor comes from:
- split, bonus: the ratios in NSE's subject text (both are multiplied when one subject has both)
- rights: theoretical ex-rights price / close before the ex-date; issue price = face value + premium
- demerger, scheme of arrangement: open on the ex-date / close before it (NSE discovers the
  ex-demerger price in a special pre-open session); moves below `demerger_min_move` count as 1
- data/manual/ca_adjustments.csv: hand-checked factors; they replace computed ones on the same
  symbol and ex-date and add events NSE's list is missing
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import Config
from src.ingest.sources.nse_corporate import _BONUS, _SPLIT
from src.store.db import read_table, write_manifest, write_table

_RIGHTS = re.compile(r"rights\D*?(\d+)\s*:\s*(\d+).*?premium\s*(?:of\s*)?(?:rs\.?|re\.?|inr)?\s*(\d+(?:\.\d+)?)",
                     re.IGNORECASE)
MANUAL_FILE = "ca_adjustments.csv"
MANUAL_COLUMNS = ["symbol", "ex_date", "factor", "kind", "note"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


def split_bonus_parts(subject: str) -> list[tuple]:
    """Split and bonus components in one subject, e.g. 'Bonus 1:1/Face Value Split From Rs 10 To Rs 2'."""
    s = subject or ""
    low = s.lower()
    parts = []
    if "split" in low or "sub-division" in low or "subdivision" in low:
        m = _SPLIT.search(s)
        if m and float(m[1]) > 0 and float(m[2]) > 0:
            parts.append(("split", float(m[1]), float(m[2])))
    if "bonus" in low:
        m = _BONUS.search(s)
        if m and int(m[2]) > 0:
            parts.append(("bonus", int(m[1]), int(m[2])))
    return parts


def part_factor(part: tuple) -> float:
    kind, a, b = part
    return b / a if kind == "split" else b / (a + b)       # split: to/from; bonus new:held -> held/(new+held)


def rights_terms(subject: str) -> tuple[int, int, float] | None:
    """(new shares, shares held, premium) from 'Rights 1:15 @ Premium Rs 1247'."""
    m = _RIGHTS.search(subject or "")
    return (int(m[1]), int(m[2]), float(m[3])) if m and int(m[2]) > 0 else None


def _first_trading_index(dates: np.ndarray, ex: np.datetime64) -> int | None:
    i = int(np.searchsorted(dates, ex, side="left"))
    return i if i < len(dates) else None


def build_events(prices: pd.DataFrame, actions: pd.DataFrame, manual: pd.DataFrame,
                 demerger_min_move: float = 0.02) -> pd.DataFrame:
    """One row per adjustment event: symbol, ex_date, trade_date (first trading day >= ex_date),
    kind, factor, method, detail. Events without prices get factor NaN and method 'no_prices'."""
    by_symbol = {s: g.sort_values("trade_date") for s, g in prices.groupby("symbol")}
    rows: list[dict] = []

    def locate(symbol, ex_date):
        g = by_symbol.get(symbol)
        if g is None:
            return None, None
        dates = g["trade_date"].to_numpy(dtype="datetime64[ns]")
        i = _first_trading_index(dates, np.datetime64(pd.Timestamp(ex_date), "ns"))
        return g, i

    def add(symbol, ex_date, kind, factor, method, detail, g, i):
        trade_date = g["trade_date"].iloc[i] if g is not None and i is not None else pd.NaT
        if g is None or i is None or (method not in ("manual",) and i == 0):
            factor, method = np.nan, "no_prices"
        rows.append({"symbol": symbol, "ex_date": pd.Timestamp(ex_date), "trade_date": trade_date,
                     "kind": kind, "factor": factor, "method": method, "detail": detail})

    acts = actions.copy()
    acts["ex_date"] = pd.to_datetime(acts["ex_date"])
    acts = acts.dropna(subset=["ex_date"])

    # splits and bonuses: unique components per symbol and ex-date (NSE sometimes repeats them)
    parts: dict[tuple, set] = {}
    for r in acts.itertuples():
        for p in split_bonus_parts(r.subject):
            parts.setdefault((r.symbol, r.ex_date), set()).add(p)
    for (symbol, ex_date), ps in sorted(parts.items()):
        g, i = locate(symbol, ex_date)
        factor = float(np.prod([part_factor(p) for p in ps]))
        kind = "+".join(sorted({p[0] for p in ps}))
        add(symbol, ex_date, kind, factor, "subject_ratio", "; ".join(map(str, sorted(ps))), g, i)

    for r in acts[acts["ca_type"] == "rights"].itertuples():
        g, i = locate(r.symbol, r.ex_date)
        terms = rights_terms(r.subject)
        if terms is None:
            add(r.symbol, r.ex_date, "rights", np.nan, "unparsed", r.subject, g, i)
            continue
        new, held, premium = terms
        face = r.face_value if pd.notna(r.face_value) else 0.0
        if g is None or i is None or i == 0:
            add(r.symbol, r.ex_date, "rights", np.nan, "no_prices", r.subject, g, i)
            continue
        cum = float(g["close"].iloc[i - 1])
        issue = face + premium
        terp = (held * cum + new * issue) / (held + new)
        add(r.symbol, r.ex_date, "rights", terp / cum, "terp",
            f"{new}:{held} at Rs {issue:g} (face {face:g} + premium {premium:g}); close before {cum:g}", g, i)

    for r in acts[acts["ca_type"] == "demerger"].itertuples():
        g, i = locate(r.symbol, r.ex_date)
        if g is None or i is None or i == 0:
            add(r.symbol, r.ex_date, "demerger", np.nan, "no_prices", r.subject, g, i)
            continue
        ratio = float(g["open"].iloc[i]) / float(g["close"].iloc[i - 1])
        if abs(ratio - 1) < demerger_min_move:
            add(r.symbol, r.ex_date, "demerger", 1.0, "no_price_effect", f"open/close before = {ratio:.4f}", g, i)
        else:
            add(r.symbol, r.ex_date, "demerger", ratio, "ex_date_open", f"open/close before = {ratio:.4f}", g, i)

    events = pd.DataFrame(rows, columns=["symbol", "ex_date", "trade_date", "kind", "factor", "method", "detail"])

    if not manual.empty:
        man = manual.copy()
        man["ex_date"] = pd.to_datetime(man["ex_date"])
        keys = set(zip(man["symbol"], man["ex_date"]))
        events = events[[k not in keys for k in zip(events["symbol"], events["ex_date"])]]
        manual_rows = []
        for r in man.itertuples():
            g, i = locate(r.symbol, r.ex_date)
            manual_rows.append({"symbol": r.symbol, "ex_date": r.ex_date,
                                "trade_date": g["trade_date"].iloc[i] if g is not None and i is not None else pd.NaT,
                                "kind": r.kind, "factor": float(r.factor), "method": "manual",
                                "detail": r.note})
        events = pd.concat([events, pd.DataFrame(manual_rows)], ignore_index=True)

    return events.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def adjust_prices(prices: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Adjusted OHLC and volume. Keeps raw_open/raw_close and the cumulative adj_factor."""
    out = prices.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    factor = np.ones(len(out))
    usable = events.dropna(subset=["factor", "trade_date"])
    for symbol, ev in usable.groupby("symbol"):
        idx = np.flatnonzero(out["symbol"].to_numpy() == symbol)
        dates = out["trade_date"].to_numpy()[idx]
        for d, f in zip(ev["trade_date"], ev["factor"]):
            factor[idx[dates < np.datetime64(d)]] *= f
    out["adj_factor"] = factor
    out["raw_open"] = out["open"]
    out["raw_close"] = out["close"]
    for c in PRICE_COLUMNS:
        out[c] = out[c] * factor
    out["volume"] = out["volume"] / factor
    return out


def jump_report(adjusted: pd.DataFrame, events: pd.DataFrame, index_close: pd.Series,
                threshold: float = 0.20) -> pd.DataFrame:
    """Adjusted daily moves above `threshold`; `on_event_day` rows mean an adjustment is wrong or missing."""
    df = adjusted[["symbol", "trade_date", "close"]].copy()
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    big = df[df["ret"].abs() > threshold].copy()
    ev_days = set(zip(events["symbol"], pd.to_datetime(events["trade_date"])))
    big["on_event_day"] = [(s, d) in ev_days for s, d in zip(big["symbol"], big["trade_date"])]
    nifty = index_close.sort_index().pct_change()
    big["nifty_ret"] = big["trade_date"].map(nifty)
    return big.drop(columns="close").reset_index(drop=True)


def load_manual(cfg: Config) -> pd.DataFrame:
    path = cfg.manual_dir / MANUAL_FILE
    if not path.exists():
        return pd.DataFrame(columns=MANUAL_COLUMNS)
    return pd.read_csv(path, comment="#")


def load_raw_prices(cfg: Config) -> pd.DataFrame:
    prices = read_table(cfg, "prices_daily", where="series = 'EQ'",
                        columns=["symbol", "trade_date", "open", "high", "low", "close", "volume", "turnover"])
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    return prices.drop_duplicates(["symbol", "trade_date"], keep="last")


def build(cfg: Config) -> dict:
    """Write derived/prices_adj, derived/ca_events and derived/jump_report. Returns a summary."""
    a = cfg.get("adjustment", {})
    prices = load_raw_prices(cfg)
    actions = read_table(cfg, "corporate_actions")
    events = build_events(prices, actions, load_manual(cfg), a.get("demerger_min_move", 0.02))
    adjusted = adjust_prices(prices, events)
    index = read_table(cfg, "index_daily", where="index_name = 'Nifty 50'", columns=["trade_date", "close"])
    index_close = index.assign(trade_date=pd.to_datetime(index["trade_date"])).set_index("trade_date")["close"]
    jumps = jump_report(adjusted, events, index_close, a.get("jump_threshold", 0.20))

    write_table(cfg, "prices_adj", adjusted)
    write_table(cfg, "ca_events", events)
    write_table(cfg, "jump_report", jumps)
    summary = {
        "rows": len(adjusted), "symbols": adjusted["symbol"].nunique(),
        "events": len(events), "events_by_method": events["method"].value_counts().to_dict(),
        "jumps_over_threshold": len(jumps), "jumps_on_event_days": int(jumps["on_event_day"].sum()),
    }
    write_manifest(cfg.derived_dir / "prices_adj", stage="prices_adj", **summary)
    return summary
