"""Read lake and derived tables as DataFrames, write derived tables, and record build manifests."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import ROOT, Config


def read_table(cfg: Config, table: str, layer: str = "lake", where: str | None = None,
               columns: list[str] | None = None) -> pd.DataFrame:
    """All Parquet files of one table, read through DuckDB (tolerates small schema differences)."""
    import duckdb

    folder = (cfg.lake_dir if layer == "lake" else cfg.derived_dir) / table
    if not any(folder.glob("*.parquet")):
        hint = "" if layer == "lake" else " Build it first: python -m src.models prepare"
        raise FileNotFoundError(f"No Parquet files in {folder}.{hint}")
    cols = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    sql = f"SELECT {cols} FROM read_parquet('{(folder / '*.parquet').as_posix()}', union_by_name = true)"
    if where:
        sql += f" WHERE {where}"
    con = duckdb.connect()
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def write_table(cfg: Config, table: str, df: pd.DataFrame, name: str | None = None) -> Path:
    """Write a derived table to data/derived/<table>/<name>.parquet (atomic replace)."""
    folder = cfg.derived_dir / table
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name or table}.parquet"
    tmp = path.with_name(path.name + ".part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def universe(cfg: Config, tier: str | None = "nifty50") -> pd.DataFrame:
    df = pd.read_csv(cfg.manual_dir / "universe.csv")
    return df if tier is None else df[df["tier"] == tier].reset_index(drop=True)


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=10, check=False)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True, timeout=10, check=False).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def config_hash(*parts) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def write_manifest(folder: Path, **info) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    info = {"git_commit": git_commit(), "built_at": datetime.now(timezone.utc).isoformat(), **info}
    path = folder / "_manifest.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path
