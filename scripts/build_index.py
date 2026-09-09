"""Build a BM25 index over the corpus.

Tokenizes every document with the Phase 1 tokenizer, indexes with bm25s
(k1=1.2, b=0.75), and saves to data/index/bm25_<variant>/.

variant a = title + answer body        (corpus.text, default)
variant b = title + question + answer

    python scripts/build_index.py [a|b]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import bm25s
import duckdb
import numpy as np

from tokenizer import tokenize

PROCESSED = Path("data/processed")
INDEX_ROOT = Path("data/index")
K1, B = 1.2, 0.75


def load_docs(variant: str):
    src = (PROCESSED / "corpus.parquet").as_posix()
    con = duckdb.connect()
    if variant == "a":
        rows = con.execute(f"SELECT doc_id, text FROM read_parquet('{src}')").fetchall()
    elif variant == "b":
        rows = con.execute(
            f"SELECT doc_id, "
            f"title || '\n\n' || question_body || '\n\n' || answer_body "
            f"FROM read_parquet('{src}')"
        ).fetchall()
    else:
        raise SystemExit("variant must be 'a' or 'b'")
    doc_ids = np.array([r[0] for r in rows], dtype=np.int64)
    texts = [r[1] for r in rows]
    return doc_ids, texts


def main() -> None:
    variant = sys.argv[1].lower() if len(sys.argv) > 1 else "a"
    if not (PROCESSED / "corpus.parquet").exists():
        raise SystemExit("missing corpus.parquet - run scripts/build_corpus.py first")

    print(f"variant {variant}: loading corpus ...")
    doc_ids, texts = load_docs(variant)
    print(f"  {len(texts):,} documents")

    print("tokenizing ...")
    t0 = time.time()
    corpus_tokens = [tokenize(t) for t in texts]
    print(f"  done in {time.time() - t0:.0f}s")

    print("indexing ...")
    t0 = time.time()
    retriever = bm25s.BM25(k1=K1, b=B)
    retriever.index(corpus_tokens)
    print(f"  done in {time.time() - t0:.0f}s")

    out = INDEX_ROOT / f"bm25_{variant}"
    out.mkdir(parents=True, exist_ok=True)
    retriever.save(str(out))
    np.save(out / "doc_ids.npy", doc_ids)
    print(f"\nsaved index to {out}/  (k1={K1}, b={B})")


if __name__ == "__main__":
    main()
