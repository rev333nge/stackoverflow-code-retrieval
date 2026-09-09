"""Evaluate a BM25 index against the qrels: NDCG@10, MRR@10, Recall@100.

    python scripts/evaluate.py <variant> <split>
      variant = a | b     (which index)
      split   = val | test

Compare variants on val, then report the winner on test.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bm25s
import duckdb
import numpy as np

from tokenizer import tokenize

PROCESSED = Path("data/processed")
INDEX_ROOT = Path("data/index")

NDCG_K = 10
MRR_K = 10
RECALL_K = 100
RETRIEVE_K = 100   # must be >= max(cutoffs)


def load_qrels(con) -> dict[int, dict[int, int]]:
    src = (PROCESSED / "qrels.parquet").as_posix()
    qrels: dict[int, dict[int, int]] = {}
    for qid, did, grade in con.execute(
        f"SELECT query_id, doc_id, grade FROM read_parquet('{src}')"
    ).fetchall():
        qrels.setdefault(qid, {})[did] = grade
    return qrels


def load_split(con, split: str):
    src = (PROCESSED / "splits.parquet").as_posix()
    rows = con.execute(
        f"SELECT query_id, query_text FROM read_parquet('{src}') WHERE split = ?",
        [split],
    ).fetchall()
    return [r[0] for r in rows], [r[1] for r in rows]


def dcg(grades) -> float:
    # gain = 2^grade - 1, discount = log2(rank+1), rank 1-based
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def evaluate(variant: str, split: str) -> None:
    index_dir = INDEX_ROOT / f"bm25_{variant}"
    if not index_dir.exists():
        raise SystemExit(f"no index at {index_dir} - run: python scripts/build_index.py {variant}")

    con = duckdb.connect()
    qrels = load_qrels(con)
    qids, texts = load_split(con, split)
    print(f"variant {variant}, split {split}: {len(qids):,} queries")

    retriever = bm25s.BM25.load(str(index_dir), load_corpus=False)
    doc_ids = np.load(index_dir / "doc_ids.npy")

    # empty token lists would break retrieve; a placeholder token matches nothing
    q_tokens = [tokenize(t) or ["\0"] for t in texts]
    print("retrieving ...")
    results, _ = retriever.retrieve(q_tokens, k=RETRIEVE_K)

    ndcg_sum = mrr_sum = recall_sum = 0.0
    n = 0
    for qi, qid in enumerate(qids):
        judged = qrels.get(qid)
        if not judged:
            continue
        n += 1

        # grade of each retrieved doc, in rank order
        graded = [judged.get(int(doc_ids[int(r)]), 0) for r in results[qi]]

        # NDCG@10
        ideal = sorted(judged.values(), reverse=True)[:NDCG_K]
        idcg = dcg(ideal)
        ndcg_sum += (dcg(graded[:NDCG_K]) / idcg) if idcg > 0 else 0.0

        # MRR@10
        for rank, g in enumerate(graded[:MRR_K], 1):
            if g >= 1:
                mrr_sum += 1.0 / rank
                break

        # Recall@100
        total_rel = sum(1 for g in judged.values() if g >= 1)
        got_rel = sum(1 for g in graded[:RECALL_K] if g >= 1)
        recall_sum += (got_rel / total_rel) if total_rel > 0 else 0.0

    print("\n" + "=" * 40)
    print(f"BM25  variant={variant}  split={split}")
    print("=" * 40)
    print(f"  queries scored : {n:,}")
    print(f"  NDCG@{NDCG_K}       : {ndcg_sum / n:.4f}")
    print(f"  MRR@{MRR_K}        : {mrr_sum / n:.4f}")
    print(f"  Recall@{RECALL_K}    : {recall_sum / n:.4f}")


def main() -> None:
    variant = sys.argv[1].lower() if len(sys.argv) > 1 else "a"
    split = sys.argv[2].lower() if len(sys.argv) > 2 else "val"
    if split not in ("val", "test"):
        raise SystemExit("split must be 'val' or 'test'")
    evaluate(variant, split)


if __name__ == "__main__":
    main()
