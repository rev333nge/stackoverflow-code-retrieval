"""Flatten the cleaned Q&A into a search corpus, one doc per answer.

doc text = title + answer body. Title, question body and answer body are
also kept separately so the indexer can build title+question+answer too.
answer_score and is_accepted ride along for the qrels step.

    python scripts/build_corpus.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb

PROCESSED = Path("data/processed")

BUILD = """
COPY (
    SELECT
        a.id                                   AS doc_id,
        a.question_id                          AS question_id,
        q.title                                AS title,
        q.body_clean                           AS question_body,
        a.body_clean                           AS answer_body,
        q.title || '\n\n' || a.body_clean      AS text,          -- variant A
        a.score                                AS answer_score,
        (a.id = q.accepted_answer_id)          AS is_accepted,
        q.is_eval_query                        AS from_eval_query
    FROM read_parquet('{answers}') a
    JOIN read_parquet('{questions}') q
      ON a.question_id = q.id
) TO '{out}' (FORMAT parquet)
"""

SUMMARY = """
SELECT
    count(*)                                    AS docs,
    count(DISTINCT question_id)                 AS questions_covered,
    count(*) FILTER (WHERE is_accepted)         AS accepted_docs,
    round(avg(length(text)))                    AS avg_text_len,
    max(length(text))                           AS max_text_len,
    count(*) FILTER (WHERE from_eval_query)      AS docs_of_eval_qs
FROM read_parquet('{out}')
"""


def main() -> None:
    q = (PROCESSED / "questions_clean.parquet").as_posix()
    a = (PROCESSED / "answers_clean.parquet").as_posix()
    out = (PROCESSED / "corpus.parquet").as_posix()

    for path, name in [(q, "questions_clean.parquet"), (a, "answers_clean.parquet")]:
        if not Path(path).exists():
            raise SystemExit(f"missing {name} - run scripts/clean_text.py first")

    con = duckdb.connect()
    print("building corpus ...")
    con.execute(BUILD.format(answers=a, questions=q, out=out))

    row = con.execute(SUMMARY.format(out=out)).fetchone()
    cols = [d[0] for d in con.description]
    s = dict(zip(cols, row))
    print("\n" + "=" * 56)
    print("CORPUS")
    print("=" * 56)
    print(f"  documents (= answers)   : {s['docs']:,}")
    print(f"  questions covered       : {s['questions_covered']:,}")
    print(f"  accepted-answer docs    : {s['accepted_docs']:,}")
    print(f"  docs from eval queries  : {s['docs_of_eval_qs']:,}")
    print(f"  avg text length         : {s['avg_text_len']:,.0f} chars")
    print(f"  max text length         : {s['max_text_len']:,} chars")

    mb = Path(out).stat().st_size / 1e6
    print(f"\n  data/processed/corpus.parquet  ({mb:,.0f} MB)")

    print("\n  --- sample document (text column) ---")
    sample = con.execute(
        f"SELECT doc_id, text FROM read_parquet('{out}') "
        f"WHERE length(text) BETWEEN 300 AND 500 LIMIT 1"
    ).fetchone()
    if sample:
        print(f"  doc_id={sample[0]}")
        for line in sample[1].splitlines():
            print(f"  | {line}")


if __name__ == "__main__":
    main()
