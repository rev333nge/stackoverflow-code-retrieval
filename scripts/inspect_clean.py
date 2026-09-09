"""Sanity-check the cleaned parquet: row counts, empty/length stats,
leftover markup, and a couple of example bodies. Read-only.

    python scripts/inspect_clean.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb

PROCESSED = Path("data/processed")


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def check_counts(con, raw: str, clean: str, label: str) -> None:
    rule(f"{label}: ROW COUNT (clean must equal raw)")
    r = con.execute(f"SELECT count(*) FROM read_parquet('{raw}')").fetchone()[0]
    c = con.execute(f"SELECT count(*) FROM read_parquet('{clean}')").fetchone()[0]
    dup = con.execute(
        f"SELECT count(*) - count(DISTINCT id) FROM read_parquet('{clean}')"
    ).fetchone()[0]
    print(f"  raw rows        : {r:,}")
    print(f"  clean rows      : {c:,}   {'OK' if c == r else '!! MISMATCH'}")
    print(f"  duplicate ids   : {dup:,}   {'OK' if dup == 0 else '!! DUPES'}")


def check_bodies(con, clean: str, label: str) -> None:
    rule(f"{label}: body_clean STATS")
    row = con.execute(
        f"""
        SELECT
          count(*)                                             AS n,
          count(*) FILTER (WHERE body_clean = '')              AS empty,
          count(*) FILTER (WHERE length(body_clean) < 15)      AS very_short,
          round(avg(length(body_clean)))                       AS avg_len,
          max(length(body_clean))                              AS max_len,
          count(*) FILTER (WHERE body_clean LIKE '%<p>%'
                        OR body_clean LIKE '%</%>%'
                        OR body_clean LIKE '%&lt;%'
                        OR body_clean LIKE '%&amp;%')           AS markup_leak
        FROM read_parquet('{clean}')
        """
    ).fetchone()
    n, empty, short, avg_len, max_len, leak = row
    print(f"  rows            : {n:,}")
    print(f"  empty body      : {empty:,}   ({empty / n * 100:.2f}%)")
    print(f"  under 15 chars  : {short:,}   ({short / n * 100:.2f}%)")
    print(f"  avg length      : {avg_len:,.0f} chars")
    print(f"  max length      : {max_len:,} chars")
    print(f"  markup leaked   : {leak:,}   {'OK' if leak == 0 else '<- inspect these'}")


def show_examples(con, clean: str, label: str, body_col: str = "body_clean") -> None:
    rule(f"{label}: EXAMPLES (cleaned output)")
    # one with a code-ish body, one ordinary
    for cond, tag in [
        ("body_clean LIKE '%def %' OR body_clean LIKE '%import %'", "has code"),
        ("length(body_clean) BETWEEN 200 AND 600", "ordinary"),
    ]:
        row = con.execute(
            f"SELECT id, {body_col} FROM read_parquet('{clean}') "
            f"WHERE {cond} LIMIT 1"
        ).fetchone()
        if not row:
            continue
        print(f"\n  --- [{tag}] id={row[0]} ---")
        text = row[1][:700]
        for line in text.splitlines():
            print(f"  | {line}")
        if len(row[1]) > 700:
            print(f"  | ... [{len(row[1]) - 700} more chars]")


def main() -> None:
    con = duckdb.connect()
    pairs = [
        ("questions.parquet", "questions_clean.parquet", "QUESTIONS"),
        ("answers.parquet", "answers_clean.parquet", "ANSWERS"),
    ]
    for raw_name, clean_name, label in pairs:
        raw = (PROCESSED / raw_name).as_posix()
        clean = (PROCESSED / clean_name).as_posix()
        if not (PROCESSED / clean_name).exists():
            print(f"missing {clean_name} - run scripts/clean_text.py first")
            continue
        check_counts(con, raw, clean, label)
        check_bodies(con, clean, label)
        show_examples(con, clean, label)


if __name__ == "__main__":
    main()
