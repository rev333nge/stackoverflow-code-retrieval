"""Interactive BM25 search. Type a question, see the top answers.

    python scripts/search_demo.py [a|b] [k]

Loads the index built by build_index.py and looks up result text from
corpus.parquet. Ctrl+C / empty line to quit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bm25s
import duckdb
import numpy as np

from tokenizer import tokenize

PROCESSED = Path("data/processed")
INDEX_ROOT = Path("data/index")


def search(retriever, doc_ids, query: str, k: int):
    tokens = tokenize(query)
    if not tokens:
        return []
    idx, scores = retriever.retrieve([tokens], k=k)
    out = []
    for j in range(idx.shape[1]):
        row = int(idx[0, j])
        out.append((int(doc_ids[row]), float(scores[0, j])))
    return out


def main() -> None:
    variant = sys.argv[1].lower() if len(sys.argv) > 1 else "a"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    index_dir = INDEX_ROOT / f"bm25_{variant}"
    if not index_dir.exists():
        raise SystemExit(f"no index at {index_dir} - run: python scripts/build_index.py {variant}")

    print("loading index ...")
    retriever = bm25s.BM25.load(str(index_dir), load_corpus=False)
    doc_ids = np.load(index_dir / "doc_ids.npy")
    con = duckdb.connect()
    corpus = (PROCESSED / "corpus.parquet").as_posix()

    print(f"ready (variant {variant}, top {k}). Type a question, empty line to quit.\n")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break

        hits = search(retriever, doc_ids, query, k)
        if not hits:
            print("  no matching tokens\n")
            continue

        ids = ",".join(str(d) for d, _ in hits)
        rows = con.execute(
            f"SELECT doc_id, title, answer_body FROM read_parquet('{corpus}') "
            f"WHERE doc_id IN ({ids})"
        ).fetchall()
        text_by_id = {r[0]: (r[1], r[2]) for r in rows}

        for rank, (doc_id, score) in enumerate(hits, 1):
            title, body = text_by_id.get(doc_id, ("?", ""))
            snippet = " ".join(body.split())[:180]
            print(f"  #{rank}  score={score:.2f}  doc={doc_id}")
            print(f"      Q: {title}")
            print(f"      A: {snippet}")
        print()


if __name__ == "__main__":
    main()
