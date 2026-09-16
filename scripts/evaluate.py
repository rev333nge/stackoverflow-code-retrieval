"""Evaluate a retriever against the qrels: NDCG@10, MRR@10, Recall@100.

    python scripts/evaluate.py [bm25|dense|hybrid|ltr] <variant> <split> [k]
      method  = bm25 (default) | dense | hybrid | ltr
      variant = a | b       (which index)
      split   = val | test
      k       = RRF constant for hybrid only (default 60)

    python scripts/evaluate.py hybrid <variant> <split> --sweep[=k1,k2,...]
      Runs BM25 + dense retrieval once, then fuses with each k in the list
      (default 10,20,40,60,100,200) and prints a comparison table.

    ltr requires data/processed/ltr_<split>_<variant>.parquet (build_ltr_features.py)
    and data/index/ltr_<variant>/model.txt (train_ltr.py) to already exist.

Compare on val, then report the winner on test.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
INDEX_ROOT = Path("data/index")

LTR_FEATURES = ["bm25_score", "bm25_rank", "dense_score", "dense_rank", "rrf_score"]

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


def retrieve_bm25(variant: str, texts: list[str]):
    import bm25s
    from tokenizer import tokenize

    index_dir = INDEX_ROOT / f"bm25_{variant}"
    if not index_dir.exists():
        raise SystemExit(f"no index at {index_dir} - run: python scripts/build_index.py {variant}")
    retriever = bm25s.BM25.load(str(index_dir), load_corpus=False)
    doc_ids = np.load(index_dir / "doc_ids.npy")
    q_tokens = [tokenize(t) or ["\0"] for t in texts]   # placeholder for empty
    results, scores = retriever.retrieve(q_tokens, k=RETRIEVE_K)
    return results, scores, doc_ids


def retrieve_dense(variant: str, texts: list[str]):
    """Brute-force cosine search on GPU via torch, not faiss-cpu.

    Mathematically identical to the saved IndexFlatIP (exact, unit vectors,
    inner product = cosine) -- faiss-gpu isn't installable on this platform
    (no conda, no Windows/Python-3.14 wheel), but the search itself is just
    a matmul + top-k, which torch already does on this machine's CUDA
    device (same GPU the embedder already runs on). Batched over queries so
    the similarity matrix (queries x corpus) never fully materializes.
    """
    import faiss
    import torch
    from embedder import load_model, encode_queries

    index_dir = INDEX_ROOT / f"dense_{variant}"
    if not (index_dir / "dense.faiss").exists():
        raise SystemExit(f"no index at {index_dir} - run: python scripts/build_embeddings.py {variant}")
    index = faiss.read_index(str(index_dir / "dense.faiss"))
    doc_ids = np.load(index_dir / "doc_ids.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    doc_vecs = index.reconstruct_n(0, index.ntotal)  # (N, dim) float32, already unit-normalized
    doc_t = torch.from_numpy(doc_vecs).to(device=device, dtype=torch.float16)

    model = load_model(device)
    qvecs = encode_queries(model, texts)
    q_t = torch.from_numpy(qvecs).to(device=device, dtype=torch.float16)

    batch = 1024
    all_scores, all_idx = [], []
    with torch.no_grad():
        for start in range(0, q_t.shape[0], batch):
            sims = q_t[start:start + batch] @ doc_t.T          # (b, N) cosine scores
            top_scores, top_idx = torch.topk(sims, k=RETRIEVE_K, dim=1)
            all_scores.append(top_scores.float().cpu().numpy())
            all_idx.append(top_idx.cpu().numpy())
    scores = np.concatenate(all_scores, axis=0)
    results = np.concatenate(all_idx, axis=0)
    return results, scores, doc_ids


def retrieve_ltr(variant: str, split: str):
    """Score the precomputed LTR candidate pool with the trained model.

    Unlike bm25/dense/hybrid, this doesn't retrieve anything itself - it
    loads the (query, doc) feature rows build_ltr_features.py already built,
    runs the trained LightGBM model over them, and ranks each query's
    candidates by predicted score. Returns its own (qids, doc_id_lists),
    since the feature file already carries a fixed candidate pool per query.
    """
    import lightgbm as lgb

    feat_path = PROCESSED / f"ltr_{split}_{variant}.parquet"
    if not feat_path.exists():
        raise SystemExit(f"no features at {feat_path} - run: python scripts/build_ltr_features.py {variant} {split}")
    model_path = INDEX_ROOT / f"ltr_{variant}" / "model.txt"
    if not model_path.exists():
        raise SystemExit(f"no model at {model_path} - run: python scripts/train_ltr.py {variant}")

    df = pd.read_parquet(feat_path)
    booster = lgb.Booster(model_file=str(model_path))
    df["pred"] = booster.predict(df[LTR_FEATURES])
    df = df.sort_values(["query_id", "pred"], ascending=[True, False])

    qids, doc_id_lists = [], []
    for qid, g in df.groupby("query_id", sort=False):
        qids.append(qid)
        doc_id_lists.append(g["doc_id"].tolist()[:RETRIEVE_K])
    return qids, doc_id_lists


def dcg(grades) -> float:
    # gain = 2^grade - 1, discount = log2(rank+1), rank 1-based
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def to_doc_ids(index_results, doc_ids_arr):
    """Translate a retriever's per-query row of internal indices into real doc_ids."""
    return [[int(doc_ids_arr[r]) for r in row] for row in index_results]


def fuse_rrf(bm25_lists, dense_lists, k: int, top: int = RETRIEVE_K):
    """Reciprocal Rank Fusion: merge two per-query doc_id rankings into one.

    RRF(d) = sum over methods of 1 / (k + rank_method(d)); a doc missing from
    a method's list contributes 0 for that method. Ranked descending, top-`top` kept.
    """
    fused = []
    for bm25_row, dense_row in zip(bm25_lists, dense_lists):
        scores: dict[int, float] = {}
        for rank, did in enumerate(bm25_row, 1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
        for rank, did in enumerate(dense_row, 1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
        ranked = sorted(scores, key=scores.get, reverse=True)[:top]
        fused.append(ranked)
    return fused


def score(qids, doc_id_lists, qrels):
    """NDCG@10 / MRR@10 / Recall@100 given, per query, a ranked list of real doc_ids."""
    ndcg_sum = mrr_sum = recall_sum = 0.0
    n = 0
    for qi, qid in enumerate(qids):
        judged = qrels.get(qid)
        if not judged:
            continue
        n += 1
        graded = [judged.get(did, 0) for did in doc_id_lists[qi]]

        ideal = sorted(judged.values(), reverse=True)[:NDCG_K]
        idcg = dcg(ideal)
        ndcg_sum += (dcg(graded[:NDCG_K]) / idcg) if idcg > 0 else 0.0

        for rank, g in enumerate(graded[:MRR_K], 1):
            if g >= 1:
                mrr_sum += 1.0 / rank
                break

        total_rel = sum(1 for g in judged.values() if g >= 1)
        got_rel = sum(1 for g in graded[:RECALL_K] if g >= 1)
        recall_sum += (got_rel / total_rel) if total_rel > 0 else 0.0
    return ndcg_sum / n, mrr_sum / n, recall_sum / n, n


def evaluate(method: str, variant: str, split: str, k_rrf: int = 60, sweep=None) -> None:
    con = duckdb.connect()
    qrels = load_qrels(con)

    if method == "ltr":
        print(f"ltr, variant {variant}, split {split}")
        print("scoring precomputed candidates ...")
        qids, doc_id_lists = retrieve_ltr(variant, split)
    else:
        qids, texts = load_split(con, split)
        print(f"{method}, variant {variant}, split {split}: {len(qids):,} queries")
        print("retrieving ...")
        if method == "bm25":
            idx_results, _, doc_ids_arr = retrieve_bm25(variant, texts)
            doc_id_lists = to_doc_ids(idx_results, doc_ids_arr)
        elif method == "dense":
            idx_results, _, doc_ids_arr = retrieve_dense(variant, texts)
            doc_id_lists = to_doc_ids(idx_results, doc_ids_arr)
        elif method == "hybrid":
            bm25_idx, _, bm25_docids_arr = retrieve_bm25(variant, texts)
            dense_idx, _, dense_docids_arr = retrieve_dense(variant, texts)
            bm25_lists = to_doc_ids(bm25_idx, bm25_docids_arr)
            dense_lists = to_doc_ids(dense_idx, dense_docids_arr)

            if sweep:
                print("\n" + "=" * 52)
                print(f"HYBRID (RRF) sweep  variant={variant}  split={split}")
                print("=" * 52)
                print(f"  {'k':>6}  {'NDCG@10':>8}  {'MRR@10':>8}  {'Recall@100':>10}")
                for kv in sweep:
                    fused = fuse_rrf(bm25_lists, dense_lists, kv)
                    ndcg, mrr, recall, n = score(qids, fused, qrels)
                    print(f"  {kv:>6}  {ndcg:>8.4f}  {mrr:>8.4f}  {recall:>10.4f}")
                print(f"\n  ({n:,} queries scored)")
                return
            doc_id_lists = fuse_rrf(bm25_lists, dense_lists, k_rrf)
        else:
            raise SystemExit(f"unknown method: {method}")

    ndcg, mrr, recall, n = score(qids, doc_id_lists, qrels)
    header = f"{method.upper()}  variant={variant}  split={split}"
    if method == "hybrid":
        header += f"  k={k_rrf}"
    print("\n" + "=" * 44)
    print(header)
    print("=" * 44)
    print(f"  queries scored : {n:,}")
    print(f"  NDCG@{NDCG_K}       : {ndcg:.4f}")
    print(f"  MRR@{MRR_K}        : {mrr:.4f}")
    print(f"  Recall@{RECALL_K}    : {recall:.4f}")


def main() -> None:
    args = sys.argv[1:]
    method = "bm25"
    if args and args[0] in ("bm25", "dense", "hybrid", "ltr"):
        method = args.pop(0)

    sweep = None
    remaining = []
    for a in args:
        if a == "--sweep":
            sweep = [10, 20, 40, 60, 100, 200]
        elif a.startswith("--sweep="):
            sweep = [int(x) for x in a.split("=", 1)[1].split(",")]
        else:
            remaining.append(a)
    args = remaining

    variant = args[0].lower() if len(args) > 0 else "a"
    split = args[1].lower() if len(args) > 1 else "val"
    k_rrf = int(args[2]) if len(args) > 2 else 60
    if split not in ("val", "test"):
        raise SystemExit("split must be 'val' or 'test'")
    evaluate(method, variant, split, k_rrf=k_rrf, sweep=sweep)


if __name__ == "__main__":
    main()
