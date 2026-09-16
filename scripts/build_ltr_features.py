"""Build (query, doc) feature rows for the LambdaMART reranker.

    python scripts/build_ltr_features.py <variant> <split> [sample_size]
      variant     = a | b
      split       = train | val | test
      sample_size = optional cap on the number of queries (random subset,
                    fixed seed). Retrieval (BM25 + dense) is run per query,
                    so this is the knob for keeping train affordable on a
                    huge corpus -- LambdaMART doesn't need every training
                    query to learn a 5-feature combination well.

Candidate pool per query = BM25 top-100 union dense top-100 (the same pool
the hybrid method fuses in evaluate.py). A candidate missing from one
method's list gets NaN for that method's score/rank - LightGBM treats NaN
as its own branch during training, so it can learn "not retrieved by X" as
a signal instead of us having to fake a sentinel value.

Writes data/processed/ltr_<split>_<variant>.parquet:
    query_id, doc_id, bm25_score, bm25_rank, dense_score, dense_rank,
    rrf_score, label
"""

from __future__ import annotations

import sys

import duckdb
import numpy as np
import pandas as pd

from evaluate import PROCESSED, load_qrels, load_split, retrieve_bm25, retrieve_dense, to_doc_ids

RRF_K = 5   # chosen in the Phase 5 val sweep
SAMPLE_SEED = 0


def build_features(variant: str, split: str, sample_size: int | None = None) -> pd.DataFrame:
    con = duckdb.connect()
    qrels = load_qrels(con)
    qids, texts = load_split(con, split)
    if sample_size is not None and sample_size < len(qids):
        rng = np.random.default_rng(SAMPLE_SEED)
        idx = rng.choice(len(qids), size=sample_size, replace=False)
        qids = [qids[i] for i in idx]
        texts = [texts[i] for i in idx]
        print(f"variant {variant}, split {split}: sampled {len(qids):,} of the full split")
    else:
        print(f"variant {variant}, split {split}: {len(qids):,} queries")

    print("retrieving ...")
    bm25_idx, bm25_scores, bm25_docids_arr = retrieve_bm25(variant, texts)
    dense_idx, dense_scores, dense_docids_arr = retrieve_dense(variant, texts)
    bm25_lists = to_doc_ids(bm25_idx, bm25_docids_arr)
    dense_lists = to_doc_ids(dense_idx, dense_docids_arr)

    print("building feature rows ...")
    rows = []
    for qi, qid in enumerate(qids):
        judged = qrels.get(qid, {})

        bm25_by_doc = {
            did: (float(bm25_scores[qi][rank - 1]), rank)
            for rank, did in enumerate(bm25_lists[qi], 1)
        }
        dense_by_doc = {
            did: (float(dense_scores[qi][rank - 1]), rank)
            for rank, did in enumerate(dense_lists[qi], 1)
        }

        for did in bm25_by_doc.keys() | dense_by_doc.keys():
            bscore, brank = bm25_by_doc.get(did, (np.nan, np.nan))
            dscore, drank = dense_by_doc.get(did, (np.nan, np.nan))
            rrf = 0.0
            if did in bm25_by_doc:
                rrf += 1.0 / (RRF_K + brank)
            if did in dense_by_doc:
                rrf += 1.0 / (RRF_K + drank)
            rows.append((qid, did, bscore, brank, dscore, drank, rrf, judged.get(did, 0)))

    return pd.DataFrame(rows, columns=[
        "query_id", "doc_id", "bm25_score", "bm25_rank",
        "dense_score", "dense_rank", "rrf_score", "label",
    ])


def main() -> None:
    args = sys.argv[1:]
    variant = args[0].lower() if len(args) > 0 else "a"
    split = args[1].lower() if len(args) > 1 else "train"
    sample_size = int(args[2]) if len(args) > 2 else None
    if split not in ("train", "val", "test"):
        raise SystemExit("split must be 'train', 'val' or 'test'")

    df = build_features(variant, split, sample_size)

    out = PROCESSED / f"ltr_{split}_{variant}.parquet"
    df.to_parquet(out, index=False)

    n_queries = df["query_id"].nunique()
    n_pos = int((df["label"] > 0).sum())
    print(f"\n{len(df):,} rows, {n_queries:,} queries, "
          f"{n_pos:,} positive-label rows ({n_pos / len(df):.1%})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
