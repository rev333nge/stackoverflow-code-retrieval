"""Judge whether each generated claim is grounded in its cited context (Phase 7).

    python scripts/judge_grounding.py [--test-one]

Reads data/processed/rag_generations.parquet (claims_json: a list of
{text, citations} the generator produced under a constrained JSON schema) and
data/processed/rag_eval_queries.parquet (the top-k doc_ids). For each claim it
asks the judge model (qwen3.5 -- deliberately different from the gemma4
generator, to avoid self-grading) whether the claim's CITED reference(s)
support it -- or all k references if the claim cites none.

Because citations arrive as structured data, there is no prose parsing here:
no splitting answers into claims, no telling a citation [1] from code arr[1].

    --test-one   only judge the first query's claims, print everything, don't save

Output:
    data/processed/rag_claim_judgments.parquet   (one row per claim)
    data/processed/rag_faithfulness.parquet      (one row per query, aggregated)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd
import requests

from generate_rag_answers import OLLAMA_URL, load_docs

PROCESSED = Path("data/processed")
JUDGE_MODEL = "qwen3.5"

JUDGE_PROMPT = """You are checking whether a claim is supported by the reference document(s) below. Verify the specifics -- exact function names, arguments, data structures, values -- not just the general topic.

{context}

Claim: "{claim}"

First quote the exact words or code from the reference(s) that support the claim. If nothing in the references supports it, write NONE.
Then give your verdict on the final line as exactly `VERDICT: YES`, `VERDICT: PARTIAL`, or `VERDICT: NO`:
- YES only if the references directly support everything the claim asserts.
- PARTIAL if only part is supported, or it is a loose/approximate match with a wrong or unverifiable detail.
- NO if the references do not support it or contradict it.

EVIDENCE:"""


def build_context(refs: list[int], doc_ids: list[int], docs: dict[int, tuple[str, str]]) -> str:
    parts = []
    for i in refs:
        title, body = docs[doc_ids[i - 1]]
        parts.append(f"[{i}] {title}\n\n{body}")
    return "\n\n".join(parts)


# qwen3.5 (a reasoning model) thinks for thousands of tokens before answering;
# num_predict must NOT cap below that or the thinking is truncated and the final
# answer never emitted (empty response). Give it ample room; retry covers the
# rare residual empty. num_ctx holds a big 4-doc prompt + long thinking + answer.
JUDGE_OPTIONS = {"num_ctx": 16384, "num_predict": 8192}


def judge(prompt: str, attempts: int = 3) -> str:
    last = ""
    for _ in range(attempts):
        resp = requests.post(
            OLLAMA_URL,
            json={"model": JUDGE_MODEL, "prompt": prompt, "stream": False,
                  "options": JUDGE_OPTIONS},
            timeout=120,
        )
        resp.raise_for_status()
        last = resp.json()["response"]
        if parse_label(last) != "unparsed":
            return last
    return last


def parse_label(raw: str) -> str:
    upper = raw.strip().upper()
    # the judge ends with an explicit "VERDICT: <label>" line
    verdicts = re.findall(r"VERDICT:\s*(YES|PARTIAL|NO)", upper)
    if verdicts:
        return {"YES": "yes", "PARTIAL": "partial", "NO": "no"}[verdicts[-1]]
    return "unparsed"


LABEL_SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def main() -> None:
    test_one = "--test-one" in sys.argv

    gens_path = PROCESSED / "rag_generations.parquet"
    queries_path = PROCESSED / "rag_eval_queries.parquet"
    if not gens_path.exists():
        raise SystemExit(f"missing {gens_path} - run scripts/generate_rag_answers.py first")

    gens = pd.read_parquet(gens_path)
    queries = pd.read_parquet(queries_path)[["query_id", "top_k_doc_ids"]]
    df = gens.merge(queries, on="query_id", how="left")
    if df["top_k_doc_ids"].isna().any():
        bad = df.loc[df["top_k_doc_ids"].isna(), "query_id"].tolist()
        raise SystemExit(f"{len(bad)} generations have no matching sampled query: {bad[:10]}")

    if test_one:
        df = df.iloc[:1]

    con = duckdb.connect()
    all_doc_ids = sorted({int(d) for ids in df["top_k_doc_ids"] for d in ids})
    docs = load_docs(con, all_doc_ids)
    missing = set(all_doc_ids) - set(docs)
    if missing:
        raise SystemExit(f"{len(missing)} doc_ids not found in corpus: {sorted(missing)[:10]}")

    claims_out = PROCESSED / "rag_claim_judgments.parquet"
    columns = ["query_id", "bucket", "claim_idx", "claim", "cited_refs", "uncited", "label", "raw_response"]

    # resume: reuse judgments already on disk so a re-run continues, not restarts
    claim_rows: list[tuple] = []
    done_qids: set = set()
    if claims_out.exists() and not test_one:
        prev = pd.read_parquet(claims_out)
        claim_rows = list(prev.itertuples(index=False, name=None))
        done_qids = set(prev["query_id"].unique())
        print(f"resuming: {len(done_qids)} queries already judged, skipping them")

    for qi, row in enumerate(df.itertuples(), 1):
        if row.query_id in done_qids:
            continue
        doc_ids = [int(d) for d in row.top_k_doc_ids]
        k = len(doc_ids)
        claims = json.loads(row.claims_json)
        print(f"[{qi}/{len(df)}] query_id={row.query_id} bucket={row.bucket} {len(claims)} claims")
        for ci, claim in enumerate(claims, 1):
            text = claim["text"]
            # cited refs come straight from the generator's structured output;
            # Option A: empty citations -> judge against all k (a claim supported
            # by some doc but left uncited is not a hallucination). `uncited`
            # flags these so the count is visible (they face an easier, all-k bar).
            valid = [n for n in claim["citations"] if 1 <= n <= k]
            uncited = not valid
            refs = valid or list(range(1, k + 1))
            context = build_context(refs, doc_ids, docs)
            prompt = JUDGE_PROMPT.format(context=context, claim=text)
            raw = judge(prompt)
            label = parse_label(raw)

            if test_one:
                print(f"\n--- claim {ci} (cites {refs}{' UNCITED' if uncited else ''}) ---")
                print(text)
                print(f"judge raw: {raw!r} -> {label}")

            claim_rows.append((row.query_id, row.bucket, ci, text, refs, uncited, label, raw))

        if not test_one:
            # checkpoint after every query so a crash/stop loses at most one query
            pd.DataFrame(claim_rows, columns=columns).to_parquet(claims_out, index=False)

    if test_one:
        return

    claims_df = pd.DataFrame(claim_rows, columns=columns)
    claims_df.to_parquet(claims_out, index=False)
    print(f"\nsaved {len(claims_df)} claim judgments -> {claims_out}")

    # unparsed = judge gave no readable verdict; exclude from the score rather
    # than silently counting it as an unsupported (0.0) claim
    n_unparsed = int((claims_df["label"] == "unparsed").sum())
    if n_unparsed:
        print(f"WARNING: {n_unparsed} claims had unparsable verdicts (excluded from faithfulness)")
    n_uncited = int(claims_df["uncited"].sum())
    print(f"uncited claims (judged against all k, Option A): {n_uncited}/{len(claims_df)}")
    scored = claims_df[claims_df["label"] != "unparsed"].copy()
    scored["score"] = scored["label"].map(LABEL_SCORE)
    faith = scored.groupby(["query_id", "bucket"], as_index=False)["score"].mean()
    faith = faith.rename(columns={"score": "faithfulness"})
    faith_out = PROCESSED / "rag_faithfulness.parquet"
    faith.to_parquet(faith_out, index=False)
    print(f"saved {len(faith)} per-query faithfulness scores -> {faith_out}")

    # surface queries with no scorable claim (empty generation or all-unparsed)
    # instead of dropping them -- otherwise a bucket's denominator silently
    # shrinks and it looks more faithful than it is
    all_q = df[["query_id", "bucket"]].drop_duplicates()
    dropped = all_q[~all_q["query_id"].isin(set(faith["query_id"]))]
    if len(dropped):
        print(f"\nWARNING: {len(dropped)} queries had no scorable claims (excluded from means):")
        print(dropped.groupby("bucket")["query_id"].count().to_string())

    print("\nmean faithfulness by bucket (n = scored / total):")
    scored_stats = faith.groupby("bucket")["faithfulness"].agg(["mean", "count"])
    total_by_bucket = all_q.groupby("bucket")["query_id"].count()
    for bucket in total_by_bucket.index:
        if bucket in scored_stats.index:
            m = scored_stats.loc[bucket, "mean"]
            sc = int(scored_stats.loc[bucket, "count"])
            print(f"  {bucket:<14} {m:.3f}   (n={sc}/{int(total_by_bucket[bucket])})")
        else:
            print(f"  {bucket:<14}  --     (n=0/{int(total_by_bucket[bucket])}, all dropped)")


if __name__ == "__main__":
    main()
