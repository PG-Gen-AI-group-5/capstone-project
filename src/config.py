"""Settings from config.yaml and secrets from .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict
    root: Path = field(default=ROOT)

    @classmethod
    def load(cls, path: Path | None = None, root: Path | None = None) -> Config:
        root = root or ROOT
        load_dotenv(root / ".env")
        with open(path or root / "config.yaml", encoding="utf-8") as f:
            return cls(yaml.safe_load(f), root)

    def __getitem__(self, key: str):
        return self.raw[key]

    def get(self, key: str, default=None):
        return self.raw.get(key, default)

    # ---- paths ----
    @property
    def data_dir(self) -> Path:
        return self.root / self.raw["paths"]["data"]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def lake_dir(self) -> Path:
        return self.data_dir / "lake"

    @property
    def manual_dir(self) -> Path:
        return self.data_dir / "manual"

    @property
    def tracker_db(self) -> Path:
        return self.data_dir / "ingest.sqlite"

    @property
    def duckdb_path(self) -> Path:
        return self.data_dir / "market.duckdb"

    # ---- helpers ----
    def start_date(self, name: str) -> date:
        value = self.raw["start_dates"][name]
        return value if isinstance(value, date) else date.fromisoformat(str(value))

    @staticmethod
    def secret(name: str) -> str | None:
        return os.environ.get(name) or None
