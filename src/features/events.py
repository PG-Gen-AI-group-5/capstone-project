"""Company-event features (EDA 03): results releases and important announcements.

Each record counts from its day 0, the first trading day whose 08:30 IST forecast may use it
(src.store.pit.first_usable_day). EDA 03 found that events change how much a price moves, not which
way: on a results day 0 the absolute move is 1.5-2x that of a random day.

- results_day0: 1 when a results release reaches its day 0 on d
- days_since_results: trading days since the last release's day 0 (0 on day 0), capped
- ann_important_day0: log(1 + important announcements whose day 0 is d)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.store.pit import first_usable_day

EVENT_FEATURES = ["results_day0", "days_since_results", "ann_important_day0"]
BOARD_OUTCOME = "Outcome of Board Meeting"


def results_records(results: pd.DataFrame, announcements: pd.DataFrame, categories) -> pd.DataFrame:
    """Every record that announces results: symbol, published_at.

    results: financial_results (symbol, broadcast_at). announcements: symbol, category, summary,
    published_at. Board-meeting outcomes count only when the summary mentions results.
    """
    ann = announcements
    board = (ann["category"] == BOARD_OUTCOME) & ann["summary"].fillna("").str.contains("result", case=False)
    parts = [results[["symbol", "broadcast_at"]].rename(columns={"broadcast_at": "published_at"}),
             ann.loc[ann["category"].isin(list(categories)) | board, ["symbol", "published_at"]]]
    out = pd.concat(parts, ignore_index=True).dropna(subset=["published_at"])
    out["published_at"] = pd.to_datetime(out["published_at"], utc=True)
    return out


def results_releases(records: pd.DataFrame, days, merge_days: int = 10, hhmm: str = "08:30") -> pd.DataFrame:
    """One row per results release: symbol, day0.

    A stock's records are walked in time order; a record more than `merge_days` trading days after the
    current release's day 0 starts a new release. Each decision uses only earlier records.
    """
    days = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(days)).unique()))
    rec = records.assign(day0=first_usable_day(records["published_at"], days, hhmm)).dropna(subset=["day0"])
    rec["pos"] = days.get_indexer(rec["day0"])
    rows = []
    for symbol, g in rec.groupby("symbol", sort=True):
        start = None
        for p in np.unique(g["pos"].to_numpy()):
            if start is None or p > start + merge_days:
                rows.append((symbol, days[p]))
                start = p
    return pd.DataFrame(rows, columns=["symbol", "day0"])


def event_features(panel_keys: pd.DataFrame, releases: pd.DataFrame, announcements: pd.DataFrame,
                   days, cap: int = 90, hhmm: str = "08:30") -> pd.DataFrame:
    """EVENT_FEATURES for each (symbol, trade_date) in panel_keys.

    days: the trading calendar the panel uses. announcements: symbol, published_at, is_important.
    """
    days = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(days)).unique()))
    pos = pd.Series(np.arange(len(days), dtype=float), index=days)
    out = pd.DataFrame({"symbol": pd.Series(panel_keys["symbol"].to_numpy(), dtype=object).astype(str),
                        "trade_date": pd.to_datetime(panel_keys["trade_date"]).to_numpy()})
    out["_row"] = np.arange(len(out))
    out["_pos"] = out["trade_date"].map(pos)

    rel = pd.DataFrame({"symbol": pd.Series(releases["symbol"].to_numpy(), dtype=object).astype(str),
                        "trade_date": pd.to_datetime(releases["day0"]).to_numpy()}).drop_duplicates()
    rel["_rel_pos"] = rel["trade_date"].map(pos)
    rel = rel.dropna(subset=["_rel_pos"])
    out = out.merge(rel[["symbol", "trade_date"]].assign(results_day0=1.0), on=["symbol", "trade_date"], how="left")
    out["results_day0"] = out["results_day0"].fillna(0.0)

    last = pd.merge_asof(out[["_row", "symbol", "_pos"]].dropna().sort_values("_pos"),
                         rel[["symbol", "_rel_pos"]].sort_values("_rel_pos"),
                         left_on="_pos", right_on="_rel_pos", by="symbol", direction="backward")
    since = pd.Series((last["_pos"] - last["_rel_pos"]).to_numpy(), index=last["_row"].to_numpy())
    out["days_since_results"] = out["_row"].map(since).clip(upper=cap).fillna(cap)

    imp = announcements[announcements["is_important"].fillna(False).astype(bool)]
    counts = pd.DataFrame({"symbol": imp["symbol"].astype(str).to_numpy(),
                           "trade_date": first_usable_day(imp["published_at"], days, hhmm).to_numpy()}).dropna()
    counts = counts.groupby(["symbol", "trade_date"]).size().rename("n").reset_index()
    out = out.merge(counts, on=["symbol", "trade_date"], how="left").sort_values("_row")
    out["ann_important_day0"] = np.log1p(out["n"].fillna(0.0))
    return out[["symbol", "trade_date"] + EVENT_FEATURES].reset_index(drop=True)


def empty_event_features(panel_keys: pd.DataFrame, cap: int = 90) -> pd.DataFrame:
    """Event features for a panel without event data: no releases, no announcements."""
    out = panel_keys[["symbol", "trade_date"]].copy()
    out["results_day0"] = 0.0
    out["days_since_results"] = float(cap)
    out["ann_important_day0"] = 0.0
    return out
