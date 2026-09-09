"""Assign each eval query to a train/val/test split by time.

Older questions train, 2021 validates, 2022+ tests. Splitting by date
(not randomly) keeps near-duplicate questions from leaking across splits
and mirrors serving future queries.

    python scripts/split_queries.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb

PROCESSED = Path("data/processed")

VAL_START_YEAR = 2021    # train < this
TEST_START_YEAR = 2022   # val < this, test >= this

BUILD = f"""
COPY (
    SELECT
        id AS query_id,
        title AS query_text,
        creation_date,
        CASE
            WHEN year(creation_date) < {VAL_START_YEAR}  THEN 'train'
            WHEN year(creation_date) < {TEST_START_YEAR} THEN 'val'
            ELSE 'test'
        END AS split
    FROM read_parquet('{{src}}')
    WHERE is_eval_query
) TO '{{out}}' (FORMAT parquet)
"""


def main() -> None:
    src = (PROCESSED / "questions_clean.parquet").as_posix()
    out = (PROCESSED / "splits.parquet").as_posix()
    if not Path(src).exists():
        raise SystemExit("missing questions_clean.parquet - run scripts/clean_text.py first")

    con = duckdb.connect()
    con.execute(BUILD.format(src=src, out=out))

    rows = con.execute(f"""
        SELECT split, count(*) n, min(creation_date)::DATE lo, max(creation_date)::DATE hi
        FROM read_parquet('{out}')
        GROUP BY split
        ORDER BY lo
    """).fetchall()
    total = sum(r[1] for r in rows)

    print("=" * 56)
    print("QUERY SPLITS")
    print("=" * 56)
    for split, n, lo, hi in rows:
        print(f"  {split:<6} {n:>8,}  ({n/total*100:5.1f}%)   {lo} -> {hi}")
    print(f"  {'total':<6} {total:>8,}")

    mb = Path(out).stat().st_size / 1e6
    print(f"\n  data/processed/splits.parquet  ({mb:,.1f} MB)")


if __name__ == "__main__":
    main()
