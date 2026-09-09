"""Dump one raw parquet file's schema, row counts, sample rows and tag counts.

Read-only. Used to check the real Tags/Body format before writing the filter.

    python scripts/inspect_parquet.py [data/raw/posts/2020.parquet]
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import duckdb

DEFAULT_FILE = Path("data/raw/posts/2024.parquet")
BODY_PREVIEW_CHARS = 700


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def show_schema(con: duckdb.DuckDBPyConnection, src: str) -> None:
    rule("SCHEMA  (column name -> type)")
    for name, dtype, *_ in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall():
        print(f"  {name:<24} {dtype}")


def show_counts(con: duckdb.DuckDBPyConnection, src: str) -> None:
    rule("ROW COUNTS BY PostTypeId  (1=question, 2=answer, 3+=wiki/other)")
    rows = con.execute(
        f"SELECT PostTypeId, count(*) AS n FROM {src} "
        f"GROUP BY PostTypeId ORDER BY PostTypeId"
    ).fetchall()
    for ptype, n in rows:
        print(f"  PostTypeId = {ptype!s:<3} {n:>12,}")


def show_row(con: duckdb.DuckDBPyConnection, src: str, where: str, title: str) -> None:
    rule(f"SAMPLE {title}")
    con.execute(f"SELECT * FROM {src} WHERE {where} LIMIT 1")
    row = con.fetchone()
    if row is None:
        print("  (no row matched)")
        return
    columns = [d[0] for d in con.description]
    for name, value in zip(columns, row):
        # string columns come back as BLOB
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", errors="replace")
        if name == "Body" and value is not None:
            body = str(value)
            clipped = body[:BODY_PREVIEW_CHARS]
            if len(body) > BODY_PREVIEW_CHARS:
                clipped += f"\n      ... [{len(body) - BODY_PREVIEW_CHARS} more chars]"
            print(f"  {name:<24}|")
            print(textwrap.indent(clipped, "      | "))
        else:
            print(f"  {name:<24} {value!r}")


def show_tag_counts(con: duckdb.DuckDBPyConnection, src: str) -> None:
    rule("QUESTIONS TAGGED pandas / numpy IN THIS FILE")
    row = con.execute(
        f"""
        WITH q AS (
          SELECT lower(TRY_CAST(Tags AS VARCHAR)) AS tags
          FROM {src}
          WHERE PostTypeId = 1
        )
        SELECT
          count(*) FILTER (WHERE tags LIKE '%|pandas|%')                         AS pandas,
          count(*) FILTER (WHERE tags LIKE '%|numpy|%')                          AS numpy,
          count(*) FILTER (WHERE tags LIKE '%|pandas|%' OR tags LIKE '%|numpy|%') AS either
        FROM q
        """
    ).fetchone()
    print("  Tags are pipe-delimited, e.g. |c++|stdvector|libc++|")
    print(f"  pandas: {row[0]:,}")
    print(f"  numpy : {row[1]:,}")
    print(f"  either: {row[2]:,}")


def show_sentinels(con: duckdb.DuckDBPyConnection, src: str) -> None:
    rule("HOW 'MISSING' IS ENCODED  (sentinels, not NULL)")
    row = con.execute(
        f"""
        SELECT
          count(*)                                                   AS questions,
          count(*) FILTER (WHERE AcceptedAnswerId IS NULL)            AS acc_is_null,
          count(*) FILTER (WHERE AcceptedAnswerId = 0)               AS acc_is_zero,
          count(*) FILTER (WHERE AcceptedAnswerId > 0)               AS acc_real,
          count(*) FILTER (WHERE year(ClosedDate) = 1970)            AS closed_epoch,
          count(*) FILTER (WHERE ClosedDate IS NULL)                 AS closed_is_null,
          count(*) FILTER (WHERE TRY_CAST(Tags AS VARCHAR) = '')     AS tags_empty,
          count(*) FILTER (WHERE TRY_CAST(Tags AS VARCHAR) IS NULL)  AS tags_null
        FROM {src}
        WHERE PostTypeId = 1
        """
    ).fetchone()
    labels = [
        "questions total",
        "AcceptedAnswerId IS NULL",
        "AcceptedAnswerId = 0      (== 'no accepted answer')",
        "AcceptedAnswerId > 0      (== 'has accepted answer')",
        "year(ClosedDate) = 1970   (== 'not closed')",
        "ClosedDate IS NULL",
        "Tags = ''",
        "Tags IS NULL",
    ]
    for label, value in zip(labels, row):
        print(f"  {label:<52} {value:>10,}")

    rule("CAN WE CAST ParentId (BLOB) TO A NUMBER?  (answers only)")
    row = con.execute(
        f"""
        SELECT
          count(*)                                                    AS answers,
          count(*) FILTER (WHERE TRY_CAST(TRY_CAST(ParentId AS VARCHAR) AS BIGINT) > 0) AS parent_ok,
          count(*) FILTER (WHERE TRY_CAST(TRY_CAST(ParentId AS VARCHAR) AS BIGINT) IS NULL) AS parent_bad
        FROM {src}
        WHERE PostTypeId = 2
        """
    ).fetchone()
    for label, value in zip(
        ["answers total", "ParentId casts to a positive int", "ParentId fails to cast"], row
    ):
        print(f"  {label:<52} {value:>10,}")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    if not path.exists():
        sys.exit(f"File not found: {path}\n(has the download reached this file?)")

    # DuckDB wants forward slashes even on Windows.
    src = f"read_parquet('{path.as_posix()}')"  # forward slashes on Windows
    con = duckdb.connect()

    print(f"Inspecting: {path}")
    show_schema(con, src)
    show_counts(con, src)
    show_row(con, src, "PostTypeId = 1 AND Score > 20", "QUESTION")
    show_row(con, src, "PostTypeId = 2 AND Score > 20", "ANSWER")
    show_tag_counts(con, src)
    show_sentinels(con, src)


if __name__ == "__main__":
    main()
