"""Sample RAG-eval queries stratified by real top-k context quality (Phase 7).

    python scripts/sample_rag_queries.py [variant] [k_context] [n_per_bucket] [seed]

Buckets are based on the actual top-k reranked context the RAG generator will
see (LTR output, same model/features evaluate.py's `ltr` method already uses)
-- NOT the raw BM25-vs-dense comparison from Phase 4, which answers a
different question (which retriever wins) than the one Phase 7 needs (is the
context handed to the LLM actually any good).

Buckets, by the best qrels grade among the top-k docs:
    good_context  - at least one grade>=2 (accepted / score>=4) doc in top-k
    weak_context  - best grade is 1 (some upvotes, not accepted/strong)
    no_context    - all top-k docs are irrelevant (grade 0) -- retrieval
                    pipeline gave the generator nothing to work with

Output: data/processed/rag_eval_queries.parquet
    (query_id, query_text, top_k_doc_ids, best_grade, bucket)
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from evaluate import load_qrels, load_split, retrieve_ltr

PROCESSED = Path("data/processed")


def main() -> None:
    variant = sys.argv[1].lower() if len(sys.argv) > 1 else "a"
    k_context = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    n_per_bucket = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    con = duckdb.connect()
    qrels = load_qrels(con)
    qids_all, texts_all = load_split(con, "test")
    text_by_qid = dict(zip(qids_all, texts_all))

    print(f"variant {variant}, split test, top-{k_context} context")
    print("scoring LTR-reranked candidates ...")
    qids, doc_id_lists = retrieve_ltr(variant, "test")

    rows = []
    for qid, doc_ids in zip(qids, doc_id_lists):
        judged = qrels.get(qid)
        if not judged:
            continue
        top_k = doc_ids[:k_context]
        grades = [judged.get(did, 0) for did in top_k]
        rows.append((qid, text_by_qid[qid], top_k, max(grades)))

    df = pd.DataFrame(rows, columns=["query_id", "query_text", "top_k_doc_ids", "best_grade"])

    good = df["best_grade"] >= 2
    weak = df["best_grade"] == 1
    none_ = df["best_grade"] == 0

    df["bucket"] = np.select(
        [good, weak, none_],
        ["good_context", "weak_context", "no_context"],
        default="unknown",
    )

    print(f"\nbucket sizes (full test set, {len(df):,} judged queries):")
    print(df["bucket"].value_counts().to_string())

    rng = np.random.default_rng(seed)
    sampled = []
    for bucket in ["good_context", "weak_context", "no_context"]:
        pool = df[df["bucket"] == bucket]
        n = min(n_per_bucket, len(pool))
        if n < n_per_bucket:
            print(f"WARNING: bucket '{bucket}' only has {len(pool)} queries, using all of them")
        idx = rng.choice(pool.index.to_numpy(), size=n, replace=False)
        sampled.append(pool.loc[idx])

    out = pd.concat(sampled).sample(frac=1, random_state=seed).reset_index(drop=True)
    out_path = PROCESSED / "rag_eval_queries.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nsaved {len(out)} queries -> {out_path}")
    print(out["bucket"].value_counts().to_string())


if __name__ == "__main__":
    main()
