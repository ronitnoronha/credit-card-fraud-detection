"""Run sql/fraud_analysis.sql against the local Parquet cache with DuckDB.

Usage: python -m src.run_sql
Each query (delimited by "-- name: <id>") is executed and saved to reports/sql/<id>.csv.
"""
from __future__ import annotations

import logging
import re

import duckdb
import pandas as pd

from src import config
from src.data_loader import load_splits

logger = logging.getLogger(__name__)
SQL_FILE = config.SQL_DIR / "fraud_analysis.sql"
OUT_DIR = config.REPORTS_DIR / "sql"


def parse_queries(text: str) -> dict[str, str]:
    """Split the SQL file into {name: query} using '-- name:' markers."""
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.M)
    return {name: body.strip().rstrip(";") for name, body in zip(parts[1::2], parts[2::2])}


def connect() -> duckdb.DuckDBPyConnection:
    if not (config.RAW_DATA_DIR / "train.parquet").exists():
        load_splits()  # downloads + caches on first run
    con = duckdb.connect()
    # Views cannot take prepared parameters; paths come from config (not user input),
    # single quotes are escaped anyway.
    train, test = (str(config.RAW_DATA_DIR / f"{s}.parquet").replace("'", "''") for s in ("train", "test"))
    con.execute(
        f"""
        CREATE VIEW transactions AS
        SELECT *, CAST(trans_date_trans_time AS TIMESTAMP) AS ts
        FROM (
            SELECT *, 'train' AS split FROM read_parquet('{train}')
            UNION ALL BY NAME
            SELECT *, 'test'  AS split FROM read_parquet('{test}')
        )
        """
    )
    return con


def run_all() -> dict[str, pd.DataFrame]:
    con = connect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, query in parse_queries(SQL_FILE.read_text()).items():
        df = con.execute(query).df()
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        results[name] = df
        logger.info("%-34s -> %d rows", name, len(df))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pd.set_option("display.width", 160)
    for name, df in run_all().items():
        print(f"\n=== {name} ===")
        print(df.head(15).to_string(index=False))
