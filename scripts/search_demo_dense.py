"""Interactive dense (embedding) search. Type a question, see the top answers.

    python scripts/search_demo_dense.py [a|b] [k]

Loads the FAISS index from build_embeddings.py, embeds the query with the same
model, and returns the nearest documents by cosine similarity. Empty line quits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import faiss
import numpy as np

from embedder import load_model, encode_queries

PROCESSED = Path("data/processed")
INDEX_ROOT = Path("data/index")


def main() -> None:
    variant = sys.argv[1].lower() if len(sys.argv) > 1 else "a"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    index_dir = INDEX_ROOT / f"dense_{variant}"
    if not (index_dir / "dense.faiss").exists():
        raise SystemExit(f"no index at {index_dir} - run: python scripts/build_embeddings.py {variant}")

    print("loading FAISS index + model ...")
    index = faiss.read_index(str(index_dir / "dense.faiss"))
    doc_ids = np.load(index_dir / "doc_ids.npy")
    model = load_model("cuda")
    con = duckdb.connect()
    corpus = (PROCESSED / "corpus.parquet").as_posix()

    print(f"ready (dense, variant {variant}, top {k}). Type a question, empty line to quit.\n")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break

        qvec = encode_queries(model, [query])          # (1, 768) float32, normalized
        scores, idx = index.search(qvec, k)            # cosine sims (desc), row indices
        hits = [(int(doc_ids[idx[0, j]]), float(scores[0, j])) for j in range(idx.shape[1])]

        ids = ",".join(str(d) for d, _ in hits)
        rows = con.execute(
            f"SELECT doc_id, title, answer_body FROM read_parquet('{corpus}') "
            f"WHERE doc_id IN ({ids})"
        ).fetchall()
        text_by_id = {r[0]: (r[1], r[2]) for r in rows}

        for rank, (doc_id, score) in enumerate(hits, 1):
            title, body = text_by_id.get(doc_id, ("?", ""))
            snippet = " ".join(body.split())[:180]
            print(f"  #{rank}  cos={score:.3f}  doc={doc_id}")
            print(f"      Q: {title}")
            print(f"      A: {snippet}")
        print()


if __name__ == "__main__":
    main()
