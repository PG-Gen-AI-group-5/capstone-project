"""The stock universe: data/manual/universe.csv.

Built from NSE's official index lists, then edited by hand (aliases, data_start, old_symbols).
Rebuilding keeps every hand edit.
"""
from __future__ import annotations

import io
import re
from datetime import date
from pathlib import Path

import pandas as pd

from src.timeutil import parse_date

LIST_URLS = {
    "nifty50": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "nifty100": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "nifty200": "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
}
COLUMNS = ["symbol", "company_name", "industry", "isin", "tier", "aliases", "data_start", "old_symbols", "notes"]
HAND_EDITED = ["aliases", "data_start", "old_symbols", "notes"]

# Known special cases (see docs/DATA_COLLECTION.md, step 1). Applied only where the field is still empty.
KNOWN = {
    "JIOFIN": {"data_start": "2023-08-21", "notes": "Listed Aug 2023 after demerger from Reliance"},
    "ETERNAL": {"aliases": "eternal ltd; zomato", "old_symbols": "ZOMATO", "notes": "Renamed from Zomato"},
    "TMPV": {"old_symbols": "TATAMOTORS",
             "notes": "Tata Motors passenger vehicles after the 2025 demerger; check the demerger price adjustment"},
    "TATACONSUM": {"old_symbols": "TATAGLOBAL", "notes": "Tata Global Beverages until 26 Feb 2020"},
    "SHRIRAMFIN": {"old_symbols": "SRTRANSFIN", "notes": "Shriram Transport Finance until 19 Dec 2022"},
}

TIER_ORDER = {"nifty50": 0, "nifty100": 1, "nifty200": 2}


def default_aliases(company_name: str) -> str:
    """Starting aliases for news matching, e.g. 'Larsen & Toubro Ltd.' -> 'larsen & toubro; larsen and toubro'."""
    name = company_name.lower().strip()
    name = re.sub(r"\s*\((india|i)\)\s*", " ", name)
    name = re.sub(r"[\s.,]*(limited|ltd)\.?$", "", name).strip(" .,")
    aliases = [name]
    if "&" in name:
        aliases.append(re.sub(r"\s*&\s*", " and ", name))
    return "; ".join(dict.fromkeys(a for a in aliases if a))


def build_universe(http, path: Path, tiers: tuple[str, ...] = ("nifty50", "nifty200")) -> pd.DataFrame:
    """Download the index lists and write universe.csv, keeping existing hand edits."""
    frames = []
    for tier in tiers:
        raw = http.get_bytes(LIST_URLS[tier], kind="csv")
        df = pd.read_csv(io.BytesIO(raw))
        df = df.rename(columns={"Company Name": "company_name", "Industry": "industry", "Symbol": "symbol",
                                "ISIN Code": "isin"})
        df["tier"] = tier
        frames.append(df[["symbol", "company_name", "industry", "isin", "tier"]])
    new = pd.concat(frames, ignore_index=True)
    new["_order"] = new["tier"].map(TIER_ORDER)
    new = new.sort_values(["_order", "symbol"]).drop_duplicates("symbol", keep="first").drop(columns="_order")

    if path.exists():
        old = pd.read_csv(path, dtype=str).fillna("")
        new = new.merge(old[["symbol", *HAND_EDITED]], on="symbol", how="left")
    for col in HAND_EDITED:
        if col not in new.columns:
            new[col] = ""
    new[HAND_EDITED] = new[HAND_EDITED].fillna("")

    for i, row in new.iterrows():
        for col, value in KNOWN.get(row["symbol"], {}).items():
            if not new.at[i, col]:
                new.at[i, col] = value
        if not new.at[i, "aliases"] and row["tier"] == "nifty50":
            new.at[i, "aliases"] = default_aliases(row["company_name"])

    new = new[COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


class Universe:
    def __init__(self, df: pd.DataFrame):
        self.df = df.fillna("")

    @classmethod
    def load(cls, path: Path) -> Universe:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `python -m src.ingest universe` first")
        return cls(pd.read_csv(path, dtype=str))

    def symbols(self, tier: str = "nifty50") -> list[str]:
        """Stocks in a universe. 'nifty200' includes the Nifty 50."""
        limit = TIER_ORDER[tier]
        return sorted(self.df.loc[self.df["tier"].map(TIER_ORDER) <= limit, "symbol"])

    def old_symbol_map(self) -> dict[str, str]:
        """Old symbol -> current symbol, e.g. {'ZOMATO': 'ETERNAL'}."""
        out = {}
        for _, row in self.df.iterrows():
            for old in filter(None, (s.strip() for s in row["old_symbols"].split(";"))):
                out[old] = row["symbol"]
        return out

    def data_start(self, symbol: str) -> date | None:
        rows = self.df.loc[self.df["symbol"] == symbol, "data_start"]
        return parse_date(rows.iloc[0]) if len(rows) else None

    def aliases(self, tier: str = "nifty50") -> dict[str, list[str]]:
        keep = set(self.symbols(tier))
        out = {}
        for _, row in self.df.iterrows():
            if row["symbol"] in keep:
                names = [a.strip().lower() for a in row["aliases"].split(";") if a.strip()]
                if names:
                    out[row["symbol"]] = names
        return out
