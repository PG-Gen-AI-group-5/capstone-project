from __future__ import annotations

import pandas as pd
import pytest
import yaml

from src.config import ROOT, Config
from src.ingest.runner import make_context, source_registrations

UNIVERSE = [
    # symbol, company_name, industry, isin, tier, aliases, data_start, old_symbols, notes
    ("AAA", "Aaa Industries Ltd.", "Energy", "INE000A01", "nifty50", "aaa industries", "", "", ""),
    ("BBB", "Bbb Bank Ltd.", "Financial Services", "INE000B01", "nifty50", "bbb bank", "", "", ""),
    ("INFY", "Infosys Ltd.", "Information Technology", "INE009A01021", "nifty50", "infosys", "", "", ""),
    ("ETERNAL", "Eternal Ltd.", "Consumer Services", "INE758T01015", "nifty50", "eternal; zomato", "",
     "ZOMATO", ""),
    ("CCC", "Ccc Motors Ltd.", "Automobile", "INE000C01", "nifty200", "", "", "", ""),
]


@pytest.fixture
def cfg(tmp_path) -> Config:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["paths"]["data"] = "data"
    return Config(raw, root=tmp_path)


@pytest.fixture
def ctx(cfg):
    context = make_context(cfg)
    context.tracker.register_sources(source_registrations(cfg) + [
        {"source": "fake", "task_unit": "day", "max_attempts": 3, "retry_base_sec": 300, "description": "test"},
    ])
    cfg.manual_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(UNIVERSE, columns=["symbol", "company_name", "industry", "isin", "tier", "aliases",
                                    "data_start", "old_symbols", "notes"]).to_csv(
        cfg.manual_dir / "universe.csv", index=False)
    yield context
    context.tracker.close()
