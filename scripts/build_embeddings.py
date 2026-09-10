"""Encode the corpus into vectors and build a FAISS index (Phase 3).

Runs every document through the bge embedding model on the GPU, then stores
the normalized vectors in an exact flat FAISS index (inner product = cosine).
One-time; saves to data/index/dense_<variant>/.

    python scripts/build_embeddings.py [a|b]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import faiss
import numpy as np

from embedder import load_model, encode_documents

PROCESSED = Path("data/processed")
INDEX_ROOT = Path("data/index")


def load_docs(variant: str):
    src = (PROCESSED / "corpus.parquet").as_posix()
    con = duckdb.connect()
    if variant == "a":
        rows = con.execute(f"SELECT doc_id, text FROM read_parquet('{src}')").fetchall()
    elif variant == "b":
        rows = con.execute(
            f"SELECT doc_id, title || '\n\n' || question_body || '\n\n' || answer_body "
            f"FROM read_parquet('{src}')"
        ).fetchall()
    else:
        raise SystemExit("variant must be 'a' or 'b'")
    return np.array([r[0] for r in rows], dtype=np.int64), [r[1] for r in rows]


def main() -> None:
    variant = sys.argv[1].lower() if len(sys.argv) > 1 else "a"
    if not (PROCESSED / "corpus.parquet").exists():
        raise SystemExit("missing corpus.parquet - run scripts/build_corpus.py first")

    print(f"variant {variant}: loading corpus ...")
    doc_ids, texts = load_docs(variant)
    print(f"  {len(texts):,} documents")

    print("loading model (first run downloads bge-base ~400 MB) ...")
    model = load_model("cuda")

    print("encoding on GPU ...")
    t0 = time.time()
    emb = encode_documents(model, texts)
    print(f"  encoded {emb.shape[0]:,} x {emb.shape[1]} in {time.time() - t0:.0f}s")

    print("building FAISS flat index ...")
    index = faiss.IndexFlatIP(emb.shape[1])   # inner product on unit vectors = cosine
    index.add(emb)

    out = INDEX_ROOT / f"dense_{variant}"
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "dense.faiss"))
    np.save(out / "doc_ids.npy", doc_ids)
    print(f"\nsaved index to {out}/  ({index.ntotal:,} vectors, dim {emb.shape[1]})")


if __name__ == "__main__":
    main()
