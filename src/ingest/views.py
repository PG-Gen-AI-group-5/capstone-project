"""DuckDB views over the Parquet lake: one view per table, queryable with SQL.

The view paths are absolute, so run `python -m src.ingest views` again on each machine after syncing data.
"""
from __future__ import annotations

from src.config import Config
from src.ingest.lake import Lake


def create_views(cfg: Config, lake: Lake) -> list[str]:
    import duckdb

    tables = lake.tables()
    cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(cfg.duckdb_path))
    try:
        for table in tables:
            pattern = (lake.table_dir(table) / "*.parquet").as_posix()
            con.execute(f"CREATE OR REPLACE VIEW {table} AS "
                        f"SELECT * FROM read_parquet('{pattern}', union_by_name = true)")
    finally:
        con.close()
    return tables
