"""Generate a synthesized answer per sampled query using the top-k context (Phase 7).

    python scripts/generate_rag_answers.py [--test-one]

Reads data/processed/rag_eval_queries.parquet (query_id, query_text,
top_k_doc_ids, best_grade, bucket), builds a numbered-context prompt from
each query's top-k docs (title + answer body, from corpus.parquet), and calls
the local Ollama model with a CONSTRAINED JSON schema so the answer comes back
as a list of {text, citations} claims. Citations live in a structured field,
never inside prose -- so the grounding judge never has to parse [1] out of
code like arr[1]; there is nothing to disambiguate.

    --test-one   only run the first query, print prompt + parsed claims, don't save

Output: data/processed/rag_generations.parquet
    (query_id, query_text, bucket, best_grade, prompt, claims_json, answer)
    claims_json = canonical [{"text":..,"citations":[..]}, ..]
    answer      = human-readable reconstruction (for the manual spot-check)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import requests

PROCESSED = Path("data/processed")
MODEL = "gemma4-e4b-unsloth-q4kxl"
OLLAMA_URL = "http://localhost:11434/api/generate"

# constrained-decoding schema: the model is forced to emit exactly this shape
CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "citations"],
            },
        }
    },
    "required": ["claims"],
}

PROMPT_TEMPLATE = """You are answering a programming question using ONLY the reference answers below. Do not use outside knowledge.

Question: {query_text}

{context}

Break your answer into individual factual claims. For each claim, give the reference number(s) from 1 to {k} whose content directly supports it. If no reference supports a claim, use an empty list. Use only the references above."""


def load_docs(con, doc_ids: list[int]) -> dict[int, tuple[str, str]]:
    corpus = (PROCESSED / "corpus.parquet").as_posix()
    ids_str = ",".join(str(int(d)) for d in doc_ids)
    rows = con.execute(
        f"SELECT doc_id, title, answer_body FROM read_parquet('{corpus}') "
        f"WHERE doc_id IN ({ids_str})"
    ).fetchall()
    return {doc_id: (title, body) for doc_id, title, body in rows}


def build_prompt(query_text: str, doc_ids: list[int], docs: dict[int, tuple[str, str]]) -> str:
    parts = []
    for i, doc_id in enumerate(doc_ids, 1):
        title, body = docs[doc_id]
        parts.append(f"[{i}] {title}\n\n{body}")
    context = "\n\n".join(parts)
    return PROMPT_TEMPLATE.format(query_text=query_text, context=context, k=len(doc_ids))


# num_ctx must fit the largest prompt (~5k tokens for a 4-doc context) PLUS the
# output; the Ollama runtime default (~4096) overflows on big prompts and the
# model then produces runaway/truncated JSON. num_predict caps pathological runs.
GEN_OPTIONS = {"num_ctx": 8192, "num_predict": 2048}


def generate_claims(prompt: str, k: int, attempts: int = 3) -> list[dict]:
    """Return a list of {'text': str, 'citations': [int]} from the model.

    Constrained decoding usually yields valid JSON, but a truncated response can
    still be unparseable; retry a few times, and if all fail return [] so one bad
    query is recorded as claimless (surfaced later) rather than killing the run.
    """
    data = None
    for _ in range(attempts):
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False,
                  "format": CLAIMS_SCHEMA, "options": GEN_OPTIONS},
            timeout=300,
        )
        resp.raise_for_status()
        try:
            data = json.loads(resp.json()["response"])
            break
        except json.JSONDecodeError:
            continue
    if data is None:
        print("  WARNING: model returned unparseable JSON after retries - recording 0 claims")
        return []
    claims = []
    for c in data.get("claims", []):
        text = (c.get("text") or "").strip()
        if not text:
            continue
        cites = sorted({int(n) for n in c.get("citations", []) if 1 <= int(n) <= k})
        claims.append({"text": text, "citations": cites})
    return claims


def readable(claims: list[dict]) -> str:
    lines = []
    for c in claims:
        cite = "".join(f"[{n}]" for n in c["citations"]) or "[-]"
        lines.append(f"{c['text']} {cite}")
    return "\n".join(lines)


def main() -> None:
    test_one = "--test-one" in sys.argv

    queries_path = PROCESSED / "rag_eval_queries.parquet"
    if not queries_path.exists():
        raise SystemExit(f"missing {queries_path} - run scripts/sample_rag_queries.py first")

    queries = pd.read_parquet(queries_path)
    if test_one:
        queries = queries.iloc[:1]

    con = duckdb.connect()
    all_doc_ids = sorted({int(d) for ids in queries["top_k_doc_ids"] for d in ids})
    docs = load_docs(con, all_doc_ids)
    missing = set(all_doc_ids) - set(docs)
    if missing:
        raise SystemExit(f"{len(missing)} doc_ids not found in corpus: {sorted(missing)[:10]}")

    out_path = PROCESSED / "rag_generations.parquet"
    columns = ["query_id", "query_text", "bucket", "best_grade", "prompt", "claims_json", "answer"]

    # resume: keep whatever already generated so a re-run continues, not restarts
    rows: list[tuple] = []
    done_qids: set = set()
    if out_path.exists() and not test_one:
        prev = pd.read_parquet(out_path)
        rows = list(prev.itertuples(index=False, name=None))
        done_qids = set(prev["query_id"].unique())
        print(f"resuming: {len(done_qids)} queries already generated, skipping them")

    for i, row in enumerate(queries.itertuples(), 1):
        if row.query_id in done_qids:
            continue
        doc_ids = [int(d) for d in row.top_k_doc_ids]
        prompt = build_prompt(row.query_text, doc_ids, docs)
        print(f"[{i}/{len(queries)}] query_id={row.query_id} bucket={row.bucket} generating ...")
        claims = generate_claims(prompt, len(doc_ids))

        if test_one:
            print("\n" + "=" * 60 + "\nPROMPT\n" + "=" * 60)
            print(prompt)
            print("\n" + "=" * 60 + f"\nPARSED CLAIMS ({len(claims)})\n" + "=" * 60)
            for c in claims:
                print(f"  citations={c['citations']}  {c['text']}")
            return

        rows.append(
            (row.query_id, row.query_text, row.bucket, row.best_grade,
             prompt, json.dumps(claims), readable(claims))
        )
        pd.DataFrame(rows, columns=columns).to_parquet(out_path, index=False)  # checkpoint per query

    if test_one:
        return

    print(f"\nsaved {len(rows)} generations -> {out_path}")


if __name__ == "__main__":
    main()
