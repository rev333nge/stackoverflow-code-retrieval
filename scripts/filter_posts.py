"""Filter the raw dump to python Q&A.

  questions.parquet - every python question. is_eval_query flags the
    ones usable as test queries (accepted answer present, score >= 1, open).
  answers.parquet   - all their answers, weak ones included (needed as
    distractors at eval time).

Two passes over data/raw/posts/*.parquet (~36 GB). Run once.

    python scripts/filter_posts.py
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb

RAW_GLOB = "data/raw/posts/*.parquet"
OUT_DIR = Path("data/processed")
TMP_DIR = Path("data/tmp")

# "never closed" is stored as the 1970 epoch, so year < 2005 means open.
NOT_CLOSED = "year(ClosedDate) < 2005"

# Pass 1: questions. is_eval_query is finalised in pass 3 (needs answers).
BUILD_QUESTIONS = f"""
CREATE OR REPLACE TABLE questions AS
SELECT
    Id                                       AS id,
    CAST(Title AS VARCHAR)                    AS title,
    CAST(Body  AS VARCHAR)                    AS body,
    lower(CAST(Tags AS VARCHAR))              AS tags,
    AcceptedAnswerId                          AS accepted_answer_id,
    Score                                    AS score,
    ViewCount                                AS view_count,
    AnswerCount                              AS answer_count,
    CreationDate                             AS creation_date,
    NOT ({NOT_CLOSED})                       AS is_closed,
    (AcceptedAnswerId > 0 AND Score >= 1 AND {NOT_CLOSED}) AS is_eval_query
FROM read_parquet('{RAW_GLOB}')
WHERE PostTypeId = 1
  AND lower(CAST(Tags AS VARCHAR)) LIKE '%|python|%'
"""

# Pass 2: answers of those questions. ParentId is a BLOB numeric string.
BUILD_ANSWERS = f"""
CREATE OR REPLACE TABLE answers AS
WITH parsed AS (
    SELECT
        Id                                               AS id,
        TRY_CAST(TRY_CAST(ParentId AS VARCHAR) AS BIGINT) AS question_id,
        CAST(Body AS VARCHAR)                             AS body,
        Score                                            AS score,
        CreationDate                                     AS creation_date
    FROM read_parquet('{RAW_GLOB}')
    WHERE PostTypeId = 2
)
SELECT parsed.*
FROM parsed
WHERE question_id IN (SELECT id FROM questions)
"""

# Pass 3: drop the eval flag if the accepted answer isn't in the dump.
FINALISE_EVAL_FLAG = """
UPDATE questions
SET is_eval_query = FALSE
WHERE is_eval_query
  AND accepted_answer_id NOT IN (SELECT id FROM answers)
"""

SUMMARY = """
SELECT
    (SELECT count(*) FROM questions)                                   AS questions,
    (SELECT count(*) FROM questions WHERE is_eval_query)               AS eval_queries,
    (SELECT count(*) FROM questions WHERE is_closed)                   AS closed,
    (SELECT count(*) FROM answers)                                     AS answers,
    (SELECT min(creation_date)::DATE FROM questions)                   AS earliest,
    (SELECT max(creation_date)::DATE FROM questions)                   AS latest
"""


def scalar(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET enable_progress_bar = true")
    con.execute(f"SET temp_directory = '{TMP_DIR.as_posix()}'")  # spill here, not C:

    t0 = time.time()

    print("[1/3] scanning all years for python questions ...")
    con.execute(BUILD_QUESTIONS)
    print(f"      {scalar(con, 'SELECT count(*) FROM questions'):,} questions kept")

    print("[2/3] scanning all years for their answers (slow pass) ...")
    con.execute(BUILD_ANSWERS)
    print(f"      {scalar(con, 'SELECT count(*) FROM answers'):,} answers kept")

    print("[3/3] finalising is_eval_query ...")
    before = scalar(con, "SELECT count(*) FROM questions WHERE is_eval_query")
    con.execute(FINALISE_EVAL_FLAG)
    after = scalar(con, "SELECT count(*) FROM questions WHERE is_eval_query")
    print(f"      demoted {before - after:,} (accepted answer not in dump)")

    row = con.execute(SUMMARY).fetchone()
    cols = [d[0] for d in con.description]
    s = dict(zip(cols, row))
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  questions (corpus side) : {s['questions']:,}")
    print(f"    closed                : {s['closed']:,}")
    print(f"  eval queries            : {s['eval_queries']:,}")
    print(f"  answers (corpus docs)   : {s['answers']:,}")
    print(f"  answers per question    : {s['answers'] / max(s['questions'], 1):.1f}")
    print(f"  date range              : {s['earliest']} -> {s['latest']}")

    print("\nwriting parquet ...")
    con.execute(f"COPY questions TO '{OUT_DIR.as_posix()}/questions.parquet' (FORMAT parquet)")
    con.execute(f"COPY answers   TO '{OUT_DIR.as_posix()}/answers.parquet' (FORMAT parquet)")

    for name in ("questions.parquet", "answers.parquet"):
        mb = (OUT_DIR / name).stat().st_size / 1e6
        print(f"  data/processed/{name}  ({mb:,.0f} MB)")

    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
