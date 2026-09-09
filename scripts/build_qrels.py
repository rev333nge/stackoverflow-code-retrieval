"""Build qrels (query_id, doc_id, grade) from the corpus.

For each eval query, grade its own answers:
    3 = accepted, 2 = score >= 4, 1 = score 1-3, 0 = score <= 0
Acceptance overrides score. Unjudged docs are grade 0 by convention.

    python scripts/build_qrels.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb

PROCESSED = Path("data/processed")

GRADE2_MIN = 4   # score >= this -> grade 2
GRADE1_MIN = 1   # score >= this (and < GRADE2_MIN) -> grade 1

BUILD = f"""
COPY (
    SELECT
        question_id AS query_id,
        doc_id,
        CASE
            WHEN is_accepted                 THEN 3
            WHEN answer_score >= {GRADE2_MIN} THEN 2
            WHEN answer_score >= {GRADE1_MIN} THEN 1
            ELSE 0
        END AS grade
    FROM read_parquet('{{corpus}}')
    WHERE from_eval_query
) TO '{{out}}' (FORMAT parquet)
"""


def main() -> None:
    corpus = (PROCESSED / "corpus.parquet").as_posix()
    out = (PROCESSED / "qrels.parquet").as_posix()
    if not Path(corpus).exists():
        raise SystemExit("missing corpus.parquet - run scripts/build_corpus.py first")

    con = duckdb.connect()
    print("building qrels ...")
    con.execute(BUILD.format(corpus=corpus, out=out))

    # ---- summary + sanity checks ----
    g = con.execute(f"""
        SELECT grade, count(*) n FROM read_parquet('{out}')
        GROUP BY grade ORDER BY grade DESC
    """).fetchall()
    total = sum(n for _, n in g)

    n_queries = con.execute(
        f"SELECT count(DISTINCT query_id) FROM read_parquet('{out}')"
    ).fetchone()[0]

    # A query must have exactly one grade-3 (its single accepted answer).
    multi_accepted = con.execute(f"""
        SELECT count(*) FROM (
            SELECT query_id, count(*) c FROM read_parquet('{out}')
            WHERE grade = 3 GROUP BY query_id HAVING c > 1
        )
    """).fetchone()[0]

    no_accepted = con.execute(f"""
        SELECT count(*) FROM (
            SELECT query_id FROM read_parquet('{out}')
            GROUP BY query_id HAVING count(*) FILTER (WHERE grade = 3) = 0
        )
    """).fetchone()[0]

    avg_judged = total / n_queries if n_queries else 0

    print("\n" + "=" * 56)
    print("QRELS")
    print("=" * 56)
    print(f"  eval queries            : {n_queries:,}")
    print(f"  judged (query,doc) pairs: {total:,}")
    print(f"  avg judged docs / query : {avg_judged:.2f}")
    print("\n  grade distribution:")
    labels = {3: "3 accepted", 2: f"2 score>={GRADE2_MIN}", 1: "1 score 1-3", 0: "0 score<=0"}
    for grade, n in g:
        print(f"    grade {labels[grade]:<14}: {n:>9,}  ({n / total * 100:5.1f}%)")

    print("\n  sanity checks:")
    print(f"    queries with >1 accepted (want 0) : {multi_accepted}")
    print(f"    queries with 0 accepted  (want 0) : {no_accepted}")

    mb = Path(out).stat().st_size / 1e6
    print(f"\n  data/processed/qrels.parquet  ({mb:,.1f} MB)")


if __name__ == "__main__":
    main()
