"""Before/after RAG comparison: does retrieval improve gemma's answers? (Phase 7)

    python scripts/compare_rag_answers.py [--test-one]

For each sampled query, generate a natural-prose answer TWICE:
  before = gemma alone (no context, its own parametric knowledge)
  after  = gemma given our top-k retrieved docs as context
Both use a natural-answer prompt (NOT the claims/citations eval prompt), so the
output reads like a real answer. This isolates what the retrieval pipeline adds.

    --test-one   only the first query, print both, don't save

Output: data/processed/rag_comparison.parquet
    (query_id, bucket, query_text, answer_before, answer_after)
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import requests

from generate_rag_answers import load_docs, MODEL, OLLAMA_URL

PROCESSED = Path("data/processed")
OPTIONS = {"num_ctx": 8192}

PROMPT_BEFORE = "Answer this Python programming question clearly and concisely, with a short code example.\n\nQuestion: {q}"

PROMPT_AFTER = """You are answering a Python question. Use the reference answers below as your source. Give ONE clear, focused answer - lead with the best approach and a short code example. Do not list every reference separately.

Question: {q}

{ctx}

Answer:"""


def ask(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False, "options": OPTIONS},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def main() -> None:
    test_one = "--test-one" in sys.argv

    queries = pd.read_parquet(PROCESSED / "rag_eval_queries.parquet")
    if test_one:
        queries = queries.iloc[:1]

    con = duckdb.connect()
    all_ids = sorted({int(d) for ids in queries["top_k_doc_ids"] for d in ids})
    docs = load_docs(con, all_ids)

    out_path = PROCESSED / "rag_comparison.parquet"
    columns = ["query_id", "bucket", "query_text", "answer_before", "answer_after"]

    rows: list[tuple] = []
    done: set = set()
    if out_path.exists() and not test_one:
        prev = pd.read_parquet(out_path)
        rows = list(prev.itertuples(index=False, name=None))
        done = set(prev["query_id"].unique())
        print(f"resuming: {len(done)} already done, skipping them")

    for i, row in enumerate(queries.itertuples(), 1):
        if row.query_id in done:
            continue
        dids = [int(d) for d in row.top_k_doc_ids]
        ctx = "\n\n".join(f"[{j}] {docs[d][0]}\n{docs[d][1]}" for j, d in enumerate(dids, 1))
        print(f"[{i}/{len(queries)}] query_id={row.query_id} bucket={row.bucket}")
        before = ask(PROMPT_BEFORE.format(q=row.query_text))
        after = ask(PROMPT_AFTER.format(q=row.query_text, ctx=ctx))

        if test_one:
            print("\n===== BEFORE (no retrieval) =====\n" + before)
            print("\n===== AFTER (with retrieval) =====\n" + after)
            return

        rows.append((row.query_id, row.bucket, row.query_text, before, after))
        pd.DataFrame(rows, columns=columns).to_parquet(out_path, index=False)

    if test_one:
        return
    print(f"\nsaved {len(rows)} before/after pairs -> {out_path}")


if __name__ == "__main__":
    main()
