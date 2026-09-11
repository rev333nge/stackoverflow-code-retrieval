"""Interactive LTR search: BM25 + dense retrieve, LambdaMART rerank, top-k.

    python scripts/search_demo_ltr.py [a|b] [k]

For the typed query: retrieve BM25 top-100 and dense top-100, build the same
5 features build_ltr_features.py uses (bm25/dense score+rank, rrf_score),
score the merged candidate pool with the trained model, print the reranked
top-k. Empty line quits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bm25s
import duckdb
import faiss
import lightgbm as lgb
import numpy as np

from build_ltr_features import RRF_K
from embedder import load_model, encode_queries
from evaluate import RETRIEVE_K
from tokenizer import tokenize

PROCESSED = Path("data/processed")
INDEX_ROOT = Path("data/index")


def rank_lookup(doc_ids_list, scores_row):
    """doc_id -> (score, 1-based rank) for one query's top-RETRIEVE_K results."""
    return {did: (float(scores_row[r - 1]), r) for r, did in enumerate(doc_ids_list, 1)}


def build_features(bm25_by_doc, dense_by_doc):
    # row order must match train_ltr.py's FEATURES: bm25_score, bm25_rank,
    # dense_score, dense_rank, rrf_score - booster.predict() takes a plain
    # array, so a wrong order here would silently mispredict, not error.
    doc_ids, rows = [], []
    for did in bm25_by_doc.keys() | dense_by_doc.keys():
        bscore, brank = bm25_by_doc.get(did, (np.nan, np.nan))
        dscore, drank = dense_by_doc.get(did, (np.nan, np.nan))
        rrf = 0.0
        if did in bm25_by_doc:
            rrf += 1.0 / (RRF_K + brank)
        if did in dense_by_doc:
            rrf += 1.0 / (RRF_K + drank)
        doc_ids.append(did)
        rows.append([bscore, brank, dscore, drank, rrf])
    return doc_ids, np.array(rows, dtype=np.float64)


def main() -> None:
    variant = sys.argv[1].lower() if len(sys.argv) > 1 else "a"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    bm25_dir = INDEX_ROOT / f"bm25_{variant}"
    dense_dir = INDEX_ROOT / f"dense_{variant}"
    model_path = INDEX_ROOT / f"ltr_{variant}" / "model.txt"
    for p, hint in [(bm25_dir, f"build_index.py {variant}"),
                    (dense_dir, f"build_embeddings.py {variant}"),
                    (model_path, f"train_ltr.py {variant}")]:
        if not p.exists():
            raise SystemExit(f"missing {p} - run: python scripts/{hint}")

    print("loading BM25 + dense indexes + model ...")
    bm25_retriever = bm25s.BM25.load(str(bm25_dir), load_corpus=False)
    bm25_doc_ids = np.load(bm25_dir / "doc_ids.npy")
    dense_index = faiss.read_index(str(dense_dir / "dense.faiss"))
    dense_doc_ids = np.load(dense_dir / "doc_ids.npy")
    embed_model = load_model("cuda")
    booster = lgb.Booster(model_file=str(model_path))
    con = duckdb.connect()
    corpus = (PROCESSED / "corpus.parquet").as_posix()

    print(f"ready (ltr, variant {variant}, top {k}). Type a question, empty line to quit.\n")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break

        tokens = tokenize(query) or ["\0"]
        bm25_idx, bm25_scores = bm25_retriever.retrieve([tokens], k=RETRIEVE_K)
        bm25_ids = [int(bm25_doc_ids[r]) for r in bm25_idx[0]]
        bm25_by_doc = rank_lookup(bm25_ids, bm25_scores[0])

        qvec = encode_queries(embed_model, [query])
        dense_scores, dense_idx = dense_index.search(qvec, RETRIEVE_K)
        dense_ids = [int(dense_doc_ids[r]) for r in dense_idx[0]]
        dense_by_doc = rank_lookup(dense_ids, dense_scores[0])

        doc_ids, feats = build_features(bm25_by_doc, dense_by_doc)
        preds = booster.predict(feats)
        order = np.argsort(-preds)[:k]
        hits = [(doc_ids[i], float(preds[i])) for i in order]

        ids = ",".join(str(d) for d, _ in hits)
        rows = con.execute(
            f"SELECT doc_id, title, answer_body FROM read_parquet('{corpus}') "
            f"WHERE doc_id IN ({ids})"
        ).fetchall()
        text_by_id = {r[0]: (r[1], r[2]) for r in rows}

        for rank, (doc_id, pred) in enumerate(hits, 1):
            title, body = text_by_id.get(doc_id, ("?", ""))
            snippet = " ".join(body.split())[:180]
            print(f"  #{rank}  score={pred:.3f}  doc={doc_id}")
            print(f"      Q: {title}")
            print(f"      A: {snippet}")
        print()


if __name__ == "__main__":
    main()
