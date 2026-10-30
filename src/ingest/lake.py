"""Parquet lake: one folder per table, one file per partition, upserts by key.

Every table has a fixed schema (column -> type) so all partitions have the same types,
which keeps DuckDB queries across partitions simple.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PA_TYPES = {
    "str": pa.string(),
    "float": pa.float64(),
    "int": pa.int64(),
    "date": pa.date32(),
    "ts": pa.timestamp("us", tz="UTC"),
    "bool": pa.bool_(),
}


def conform(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Return a frame with exactly the schema's columns, in order, with consistent types."""
    out = {}
    for col, typ in schema.items():
        s = df[col] if col in df.columns else pd.Series([None] * len(df), index=df.index, dtype=object)
        if typ == "str":
            out[col] = s.astype("string")
        elif typ == "float":
            out[col] = pd.to_numeric(s, errors="coerce").astype("float64")
        elif typ == "int":
            out[col] = pd.to_numeric(s, errors="coerce").round().astype("Int64")
        elif typ == "date":
            d = pd.to_datetime(s, errors="coerce")
            out[col] = d.dt.date.astype(object).where(d.notna(), None)
        elif typ == "ts":
            out[col] = pd.to_datetime(s, errors="coerce", utc=True)
        elif typ == "bool":
            out[col] = s.astype("boolean")
        else:
            raise ValueError(f"Unknown type {typ!r} for column {col!r}")
    return pd.DataFrame(out, index=df.index).reset_index(drop=True)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9&._=-]", "_", str(name))


class Lake:
    def __init__(self, root: Path):
        self.root = Path(root)

    def table_dir(self, table: str) -> Path:
        return self.root / table

    def upsert(self, table: str, df: pd.DataFrame, key: list[str], partition: str,
               schema: dict[str, str]) -> int:
        """Insert new rows and replace rows with the same key. Returns the number of rows written."""
        if df.empty:
            return 0
        df = conform(df, schema)
        for part, chunk in df.groupby(partition, dropna=False, sort=False):
            path = self.table_dir(table) / f"{_safe(part)}.parquet"
            if path.exists():
                chunk = pd.concat([conform(pd.read_parquet(path), schema), chunk], ignore_index=True)
            chunk = chunk.drop_duplicates(subset=key, keep="last")
            self._write(path, chunk, schema)
        return len(df)

    def _write(self, path: Path, df: pd.DataFrame, schema: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pa_schema = pa.schema([(c, PA_TYPES[t]) for c, t in schema.items()])
        table = pa.Table.from_pandas(df, schema=pa_schema, preserve_index=False, safe=False)
        tmp = path.with_name(path.name + ".part")
        pq.write_table(table, tmp)
        os.replace(tmp, path)

    def read(self, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        files = sorted(self.table_dir(table).glob("*.parquet"))
        if not files:
            return pd.DataFrame(columns=columns or [])
        return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)

    def tables(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and any(p.glob("*.parquet")))
