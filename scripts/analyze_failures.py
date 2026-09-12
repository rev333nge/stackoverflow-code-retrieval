"""Compare BM25 vs dense per query: where each wins, why, and how much
headroom a hybrid has.

    python scripts/analyze_failures.py [variant] [split]   (default a, val)

Prints: mean NDCG@10 per method, the oracle bound (best-of-both per query),
their correlation, win/loss buckets, and real example queries per bucket.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np

from evaluate import (
    NDCG_K, dcg, load_qrels, load_split, retrieve_bm25, retrieve_dense,
)

PROCESSED = Path("data/processed")
WIN = 0.10        # NDCG gap that counts as a "win"
FAIL = 0.10       # below this, a method basically failed the query
N_EXAMPLES = 10


def per_query_ndcg(results_row, doc_ids, judged) -> float:
    graded = [judged.get(int(doc_ids[int(r)]), 0) for r in results_row]
    ideal = sorted(judged.values(), reverse=True)[:NDCG_K]
    idcg = dcg(ideal)
    return (dcg(graded[:NDCG_K]) / idcg) if idcg > 0 else 0.0


def top1_title(con, doc_id) -> str:
    corpus = (PROCESSED / "corpus.parquet").as_posix()
    row = con.execute(
        f"SELECT title FROM read_parquet('{corpus}') WHERE doc_id = {int(doc_id)} LIMIT 1"
    ).fetchone()
    return row[0] if row else "?"


def main() -> None:
    variant = sys.argv[1].lower() if len(sys.argv) > 1 else "a"
    split = sys.argv[2].lower() if len(sys.argv) > 2 else "val"

    con = duckdb.connect()
    qrels = load_qrels(con)
    qids, texts = load_split(con, split)
    print(f"variant {variant}, split {split}: {len(qids):,} queries")

    print("retrieving BM25 ...")
    bm_res, _, bm_ids = retrieve_bm25(variant, texts)
    print("retrieving dense ...")
    de_res, _, de_ids = retrieve_dense(variant, texts)

    # per-query NDCG for both, keeping only queries we have judgments for
    rows = []  # (qi, qid, bm, de)
    for qi, qid in enumerate(qids):
        judged = qrels.get(qid)
        if not judged:
            continue
        bm = per_query_ndcg(bm_res[qi], bm_ids, judged)
        de = per_query_ndcg(de_res[qi], de_ids, judged)
        rows.append((qi, qid, bm, de))

    bm = np.array([r[2] for r in rows])
    de = np.array([r[3] for r in rows])
    n = len(rows)

    # ---- headline numbers ----
    oracle = np.maximum(bm, de)
    corr = float(np.corrcoef(bm, de)[0, 1])
    print("\n" + "=" * 52)
    print(f"PER-QUERY NDCG@10  ({n:,} queries)")
    print("=" * 52)
    print(f"  BM25 mean            : {bm.mean():.4f}")
    print(f"  dense mean           : {de.mean():.4f}")
    print(f"  oracle (best-of-both): {oracle.mean():.4f}   <- hybrid ceiling")
    print(f"  correlation(bm,de)   : {corr:.3f}   (lower = more complementary)")

    # ---- buckets ----
    delta = de - bm
    both_fail = np.maximum(bm, de) < FAIL
    dense_win = (delta > WIN) & ~both_fail
    bm25_win = (delta < -WIN) & ~both_fail
    tie = ~dense_win & ~bm25_win & ~both_fail
    print("\n  buckets:")
    for label, mask in [
        ("dense wins", dense_win), ("BM25 wins", bm25_win),
        ("tie / both ok", tie), ("both fail", both_fail),
    ]:
        c = int(mask.sum())
        print(f"    {label:<16}: {c:>6,}  ({c / n * 100:5.1f}%)")

    # ---- examples ----
    def show(mask, order_desc: bool, title: str):
        idx = [i for i in range(n) if mask[i]]
        idx.sort(key=lambda i: delta[i], reverse=order_desc)
        print(f"\n  --- {title} ---")
        for i in idx[:N_EXAMPLES]:
            qi, qid, b, d = rows[i]
            bm_t = top1_title(con, bm_ids[int(bm_res[qi][0])])
            de_t = top1_title(con, de_ids[int(de_res[qi][0])])
            print(f'  Q: "{texts[qi][:70]}"   bm={b:.2f} de={d:.2f}')
            print(f'       BM25 #1 : {bm_t[:70]}')
            print(f'       dense #1: {de_t[:70]}')

    show(dense_win, True, f"top {N_EXAMPLES} DENSE wins (dense >> BM25)")
    show(bm25_win, False, f"top {N_EXAMPLES} BM25 wins (BM25 >> dense)")

    hard = [i for i in range(n) if both_fail[i]]
    print(f"\n  --- {N_EXAMPLES} BOTH-FAIL queries (hard for both) ---")
    for i in hard[:N_EXAMPLES]:
        qi = rows[i][0]
        print(f'  Q: "{texts[qi][:80]}"')


if __name__ == "__main__":
    main()
