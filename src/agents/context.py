"""What the agents can see on a forecast day, loaded once per run.

`RunData.load` reads the built stages (features, calendar, results releases, adjusted prices, base model
forecasts) and the announcements; `RunData.day(d)` gives the view as of 08:30 IST on d. Every dated
record passes through src.store.pit, so a record published at or after as_of is never visible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from src.catalysts.calendar import CatalystCalendar
from src.config import Config
from src.contracts import BaseForecast, Catalyst, Evidence, Setup
from src.features.setups import detect, make_setup
from src.store.db import read_table, universe
from src.store.pit import AnnouncementIndex, as_of_utc

CONTEXT_COLUMNS = ["ret_5", "ret_20", "ema200_gap", "ema20_50", "rsi14", "atr14_pct", "rvol_20", "dist_52w_high",
                   "rs_nifty_20", "beta_nifty", "vix", "nifty_ema200_gap", "nifty_ret_20", "usvix",
                   "results_day0", "days_since_results", "ann_important_day0"]


@dataclass
class RunData:
    cfg: Config
    features: pd.DataFrame                    # ready Nifty 50 rows, with setup_type and ref_close
    calendar: pd.DataFrame
    announcements: AnnouncementIndex
    catalysts: CatalystCalendar
    forecasts: pd.DataFrame                   # base model, one row per symbol and day
    base_model: str
    days: pd.DatetimeIndex = field(init=False)

    def __post_init__(self):
        self.days = pd.DatetimeIndex(sorted(self.features["trade_date"].unique()))
        self._by_day = {d: g.set_index("symbol") for d, g in self.features.groupby("trade_date")}
        fc = self.forecasts
        self._fc = {k: g for k, g in fc.groupby(["symbol", "trade_date"])} if len(fc) else {}

    @classmethod
    def load(cls, cfg: Config, start=None, end=None, symbols: list[str] | None = None) -> RunData:
        a = cfg.get("agents") or {}
        feats = read_table(cfg, "features_daily", layer="derived")
        feats["trade_date"] = pd.to_datetime(feats["trade_date"])
        nifty50 = set(universe(cfg)["symbol"])
        keep = nifty50 if symbols is None else nifty50 & set(symbols)
        feats = feats[feats["ready"] & feats["symbol"].isin(keep)]
        prices = read_table(cfg, "prices_adj", layer="derived", columns=["symbol", "trade_date", "raw_close"])
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        feats = with_ref_close(feats, prices)
        if start is not None:
            lo = pd.Timestamp(start) - pd.Timedelta(days=5)
            feats = feats[feats["trade_date"] >= lo]
        if end is not None:
            feats = feats[feats["trade_date"] <= pd.Timestamp(end)]
        feats = feats.assign(setup_type=detect(feats, cfg.get("setups"))).reset_index(drop=True)

        calendar = read_table(cfg, "calendar", layer="derived")
        releases = read_table(cfg, "results_releases", layer="derived")
        board = read_table(cfg, "board_meetings", columns=["symbol", "meeting_date", "purpose", "published_at"])
        overrides = pd.read_csv(cfg.manual_dir / "calendar_overrides.csv", dtype=str)
        ann = read_table(cfg, "announcements", columns=["doc_id", "symbol", "category", "summary", "published_at",
                                                         "is_important", "attachment_url"])
        base = a.get("base_model", "tft")
        return cls(cfg, feats, calendar, AnnouncementIndex(ann), CatalystCalendar(calendar, releases, board, overrides),
                   load_base_forecasts(cfg, base), base)

    def calendar_days_after(self, d: pd.Timestamp, n: int) -> pd.DatetimeIndex:
        """Trading days d .. d+n-1 (the calendar includes future weekdays)."""
        cal = pd.DatetimeIndex(pd.to_datetime(self.calendar["trade_date"]).sort_values())
        i = cal.searchsorted(d)
        return cal[i:i + n]

    def day(self, trade_date) -> DayView:
        d = pd.Timestamp(trade_date)
        rows = self._by_day.get(d)
        if rows is None:
            raise KeyError(f"{d.date()} is not a trading day with features")
        return DayView(self, d, rows)


def with_ref_close(feats: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """ref_close: the stock's last raw close before d (its previous trading day). Also defined for a live day
    whose own prices are not out yet."""
    f = feats.assign(_i=range(len(feats)), trade_date=feats["trade_date"].astype("datetime64[ns]")).sort_values("trade_date")
    p = prices[["symbol", "trade_date", "raw_close"]].dropna()
    p = p.assign(trade_date=p["trade_date"].astype("datetime64[ns]")).sort_values("trade_date")
    f["symbol"] = f["symbol"].astype(str)
    p = p.assign(symbol=p["symbol"].astype(str))
    out = pd.merge_asof(f, p.rename(columns={"raw_close": "ref_close"}), on="trade_date", by="symbol",
                        allow_exact_matches=False)
    return out.sort_values("_i").drop(columns="_i").reset_index(drop=True)


def load_base_forecasts(cfg: Config, model: str) -> pd.DataFrame:
    """Walk-forward forecasts of the base model from every saved period (validation, inner, test)."""
    folder = cfg.derived_dir / "forecasts" / model
    frames = [pd.read_parquet(p) for p in sorted(folder.glob("*.parquet"))] if folder.exists() else []
    if not frames:
        return pd.DataFrame(columns=["symbol", "trade_date", "horizon_days", "p_up", "exp_return", "vol_forecast",
                                     "model_version"])
    fc = pd.concat(frames, ignore_index=True)
    fc["trade_date"] = pd.to_datetime(fc["trade_date"])
    return fc.drop_duplicates(["symbol", "trade_date", "horizon_days"], keep="last")


@dataclass
class DayView:
    data: RunData
    trade_date: pd.Timestamp
    rows: pd.DataFrame                        # indexed by symbol

    @property
    def as_of(self) -> datetime:
        return as_of_utc(pd.Series([self.trade_date]), self.data.cfg.get("forecast_time_ist", "08:30")).iloc[0] \
            .to_pydatetime()

    @property
    def symbols(self) -> list[str]:
        return sorted(self.rows.index)

    def setup(self, symbol: str) -> Setup | None:
        r = self.rows.loc[symbol]
        if pd.isna(r["ref_close"]) or pd.isna(r["atr14_pct"]):
            return None
        return make_setup(symbol, self.trade_date.date(), r["setup_type"], float(r["ref_close"]), float(r["atr14_pct"]),
                          self.data.cfg.get("setups"))

    def base_forecast(self, symbol: str) -> BaseForecast | None:
        g = self.data._fc.get((symbol, self.trade_date))
        if g is None or g.empty:
            return None
        h = g.set_index("horizon_days")
        num = lambda col: {int(k): float(v) for k, v in h[col].items() if pd.notna(v)}
        return BaseForecast(model=self.data.base_model, model_version=str(g["model_version"].iloc[0]),
                            p_up=num("p_up"), exp_return=num("exp_return"), vol_forecast=num("vol_forecast"))

    def context(self, symbol: str) -> dict:
        r = self.rows.loc[symbol]
        return {c: (None if pd.isna(r[c]) else round(float(r[c]), 4)) for c in CONTEXT_COLUMNS if c in r.index}

    def evidence(self, symbol: str, lookback_days: int = 30, limit: int = 15) -> list[Evidence]:
        since = pd.Timestamp(self.as_of) - pd.Timedelta(days=lookback_days)
        ann = self.data.announcements.as_of(symbol, self.as_of, since)
        ann = ann.sort_values(["is_important", "published_at"], ascending=[False, False]).head(limit)
        out = [Evidence(evidence_id=f"ann:{r.doc_id}", kind="announcement", symbol=symbol,
                        published_at=r.published_at.to_pydatetime(), title=str(r.category),
                        text=str(r.summary or "")[:600], url=None if pd.isna(r.attachment_url) else r.attachment_url,
                        important=bool(r.is_important)) for r in ann.itertuples()]
        fc = self.base_forecast(symbol)
        if fc is not None:
            out.append(Evidence(evidence_id=f"forecast:{fc.model}:{self.trade_date.date()}", kind="forecast",
                                symbol=symbol, published_at=self.as_of - pd.Timedelta(hours=1),
                                title=f"{fc.model} base forecast",
                                text=f"p_up {fc.p_up}; expected return {fc.exp_return}; volatility {fc.vol_forecast}"))
        # the indicator values and code-computed levels, citable like any other record (from the last close)
        closed = self.last_close_time()
        out.append(Evidence(evidence_id=context_id(symbol, self.trade_date), kind="price", symbol=symbol,
                            published_at=closed, title="indicators at the last close",
                            text=json.dumps(self.context(symbol))))
        setup = self.setup(symbol)
        if setup is not None:
            out.append(Evidence(evidence_id=setup_id(symbol, self.trade_date), kind="price", symbol=symbol,
                                published_at=closed, title=f"code-computed {setup.setup_type} levels",
                                text=f"entry {setup.entry}, stop {setup.stop}, target {setup.target}, "
                                     f"risk-to-reward {setup.risk_reward}, ATR {setup.atr_pct:.2%} of price"))
        return out

    def last_close_time(self) -> datetime:
        """18:00 IST on the previous trading day, when its bhavcopy (and so its indicators) became public."""
        i = self.data.days.searchsorted(self.trade_date)
        prev = self.data.days[i - 1] if i > 0 else self.trade_date - pd.Timedelta(days=1)
        return (prev + pd.Timedelta(hours=18)).tz_localize("Asia/Kolkata").tz_convert("UTC").to_pydatetime()

    def catalysts(self, symbol: str, hold_days: int = 10) -> list[Catalyst]:
        days = self.data.calendar_days_after(self.trade_date, hold_days)
        return self.data.catalysts.query(symbol, self.trade_date, days[-1] if len(days) else self.trade_date,
                                         self.as_of)


def context_id(symbol: str, d) -> str:
    return f"ctx:{symbol}:{pd.Timestamp(d).date()}"


def setup_id(symbol: str, d) -> str:
    return f"setup:{symbol}:{pd.Timestamp(d).date()}"


def citable(evidence: list[Evidence]) -> list[dict]:
    """The IDs an agent may cite, with a title, for the prompts."""
    return [{"evidence_id": e.evidence_id, "title": e.title, "published_at": e.published_at.isoformat()}
            for e in evidence]


def trading_dates(data: RunData, start, end) -> list[date]:
    d = data.days
    return [x.date() for x in d[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]]


def safe_float(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
