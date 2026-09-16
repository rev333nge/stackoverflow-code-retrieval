"""Adaptive RAG: the model reads StackOverflow only when it decides it needs to.

    query
     └─ router (gemma: ANSWER | SEARCH)
         ├─ ANSWER  -> answer from model knowledge
         └─ SEARCH  -> retrieve top-4 (BM25 + dense -> LambdaMART rerank)
             └─ confidence gate (top dense cosine >= 0.70?)
                 ├─ OPEN   -> answer grounded in the retrieved docs
                 └─ CLOSED -> drop the (off-topic) docs, answer from model knowledge

Everything is local (llama.cpp, via llama-cpp-python), one .gguf model loaded
at a time. Import AdaptiveRAG and call .answer(query) -> dict with the final
answer and a decision trace, or run this file for an interactive prompt.

    python scripts/adaptive_rag.py [variant] <path-to-model.gguf>

Notes on the constants:
  - GATE_COSINE = 0.70 tuned on 99 in-domain + 8 off-domain queries: 0 false-closes
    on in-domain, ~88% of off-domain correctly closed. The gate is an OUT-OF-SCOPE
    detector (is the corpus's best match even close?), not a retrieval-quality one.
  - K = 4 context docs; MRR@10 says the relevant answer is at rank 1-2 anyway.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import bm25s
import duckdb
import faiss
import lightgbm as lgb
import numpy as np
import torch

if sys.platform == "win32":
    # llama-cpp-python's CUDA build (ggml-cuda.dll) dynamically links against
    # CUDA 12.x runtime DLLs (cudart64_12.dll, cublas64_12.dll,
    # cublasLt64_12.dll). A machine can easily have a *different* CUDA
    # Toolkit version installed system-wide (its DLLs are named
    # differently -- e.g. v13 ships cudart64_13.dll, not _12), in which case
    # Windows can't resolve ggml-cuda.dll's imports and it fails to load.
    # torch's own CUDA wheel already bundles exactly the v12 DLLs it needs
    # (that's how torch itself works without a matching system CUDA install)
    # -- point the loader at torch's lib/ before llama_cpp imports its DLLs.
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if torch_lib.is_dir():
        os.add_dll_directory(str(torch_lib))

from llama_cpp import Llama

from build_ltr_features import RRF_K
from embedder import load_model, encode_queries
from evaluate import RETRIEVE_K
from tokenizer import tokenize

PROCESSED = Path("data/processed")
INDEX_ROOT = Path("data/index")

N_CTX = 8192  # fixed at load time -- llama.cpp can't flex context per call like Ollama could
K = 4
GATE_COSINE = 0.70


ROUTER_PROMPT = """You are a helpful assistant with a Python StackOverflow search tool available.
Decide whether to use it before replying to the user's latest message.

Searching is cheap; a wrong or oversimplified answer is costly. So WHEN IN DOUBT, SEARCH.
Only choose ANSWER if the message is simple (including plain conversation, like a greeting)
and, when it is a technical question, you are certain your unaided answer is correct and complete.
Choose SEARCH for anything Python related that is tricky, niche, performance-sensitive,
easy to get subtly wrong, or where a specific idiom/technique matters.

Reply with ONLY one word: ANSWER or SEARCH.
{history}
Message: {q}

One-word decision:"""

ANSWER_ALONE_PROMPT = """You are a helpful assistant, especially knowledgeable about Python.
Reply to the user's latest message directly and naturally. If it is a Python programming
question, explain clearly and include a short code example. If it is not a programming question
(a greeting, a follow-up, small talk), just respond to it normally -- do not ask for a question.
{history}
Message: {q}

Reply:"""

ANSWER_WITH_DOCS_PROMPT = """You are answering a Python question. Use the reference answers below as your source. Give ONE clear, focused answer - lead with the best approach and a short code example. Do not list every reference separately.
{history}
Question: {q}

{context}

Answer:"""


def _format_history(history: list[tuple[str, str]] | None) -> str:
    """Render recent turns as a transcript block, or "" if there is none.

    Wrapped in explicit start/end markers (rather than bare "User:"/"Message:"
    labels) so a pasted code snippet or error log in the user's own message
    that happens to contain those words is less likely to be misread as a
    turn boundary.
    """
    if not history:
        return ""
    lines = [f"{'User' if role == 'user' else 'Assistant'}: {content}" for role, content in history]
    return "--- Conversation so far (for context only) ---\n" + "\n".join(lines) + "\n--- End of conversation so far ---\n"


class AdaptiveRAG:
    def __init__(self, variant: str = "a", device: str = "cuda", model: str | None = None) -> None:
        self.model = model  # absolute path to a .gguf file, chosen by the user
        self._llm: Llama | None = None
        self._llm_path: str | None = None
        # Path of the model currently being (re)loaded into self._llm, or
        # None the rest of the time -- polled by /api/status so the UI can
        # show real loading progress instead of guessing client-side whether
        # a reload is needed (it can't know things like "the backend process
        # itself just restarted and lost its cached model").
        self._loading_path: str | None = None
        bm25_dir = INDEX_ROOT / f"bm25_{variant}"
        dense_dir = INDEX_ROOT / f"dense_{variant}"
        model_path = INDEX_ROOT / f"ltr_{variant}" / "model.txt"
        for p, hint in [(bm25_dir, f"build_index.py {variant}"),
                        (dense_dir, f"build_embeddings.py {variant}"),
                        (model_path, f"train_ltr.py {variant}")]:
            if not p.exists():
                raise SystemExit(f"missing {p} - run: python scripts/{hint}")

        self.bm25 = bm25s.BM25.load(str(bm25_dir), load_corpus=False)
        self.bm25_ids = np.load(bm25_dir / "doc_ids.npy")
        self.dense = faiss.read_index(str(dense_dir / "dense.faiss"))
        self.dense_ids = np.load(dense_dir / "doc_ids.npy")
        self.embedder = load_model(device)
        self.booster = lgb.Booster(model_file=str(model_path))
        self.con = duckdb.connect()
        self.corpus = (PROCESSED / "corpus.parquet").as_posix()
        # This instance is shared across FastAPI's threadpool (see rag_service.py).
        # retrieve() touches state that isn't thread-safe (the single DuckDB
        # connection, the torch embedder's forward pass, the FAISS index) --
        # serialize just that section. _llm_lock below is separate: it guards
        # the loaded llama.cpp model itself.
        self._retrieve_lock = threading.Lock()
        # Unlike Ollama (a server that can run several models via HTTP calls
        # in parallel), a llama.cpp model loaded in-process can only run one
        # generation at a time, and swapping which .gguf is loaded must not
        # race a generation already in flight. One lock covers both.
        self._llm_lock = threading.Lock()

    # ---- LLM calls (all gemma, local via llama.cpp) ----
    def _load_llm(self, model_path: str) -> Llama:
        """Return the cached Llama instance for model_path, (re)loading if needed.

        Must be called with _llm_lock held.
        """
        if self._llm is not None and self._llm_path == model_path:
            return self._llm
        path = Path(model_path)
        if not path.is_file():
            raise RuntimeError(f"model file not found: {model_path}")
        self._loading_path = model_path
        try:
            self._llm = Llama(model_path=str(path), n_ctx=N_CTX, n_gpu_layers=-1, verbose=False)
        finally:
            self._loading_path = None
        self._llm_path = model_path
        return self._llm

    def loading_status(self) -> str | None:
        """Path of the model currently loading, or None. Safe to call from
        another thread while _generate() holds _llm_lock -- this is a plain
        attribute read, and the GIL makes that atomic."""
        return self._loading_path

    def _generate(self, prompt: str, max_tokens: int = 1024, temperature: float | None = None,
                  model: str | None = None) -> str:
        model_path = model or self.model
        if not model_path:
            raise RuntimeError("no model selected -- pick a .gguf file first")
        with self._llm_lock:
            try:
                llm = self._load_llm(model_path)
                out = llm.create_completion(
                    prompt, max_tokens=max_tokens,
                    temperature=0.7 if temperature is None else temperature,
                )
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"local generation failed ({Path(model_path).name}): {e}") from e
        return out["choices"][0]["text"].strip()

    def set_model(self, model: str) -> None:
        self.model = model

    def route(self, query: str, history: list[tuple[str, str]] | None = None,
              model: str | None = None) -> str:
        """Return 'SEARCH' or 'ANSWER' (defaults to ANSWER if unclear)."""
        prompt = ROUTER_PROMPT.format(history=_format_history(history), q=query)
        raw = self._generate(prompt, max_tokens=10, temperature=0, model=model).upper()
        return "SEARCH" if "SEARCH" in raw else "ANSWER"

    # ---- retrieval (BM25 + dense -> LTR rerank) ----
    @staticmethod
    def _rank_lookup(ids, scores):
        return {d: (float(scores[r - 1]), r) for r, d in enumerate(ids, 1)}

    def _build_features(self, bm25_by_doc, dense_by_doc):
        # order must match train_ltr.py: bm25_score, bm25_rank, dense_score, dense_rank, rrf_score
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

    def retrieve(self, query: str):
        """Return (top_cosine, [(doc_id, title, body), ...]) for the top-K reranked docs.

        top_cosine is the best dense cosine over the whole pool -- the confidence
        signal the gate uses (how close is the nearest doc to the query, at all).
        """
        with self._retrieve_lock:
            tokens = tokenize(query) or ["\0"]
            bm_idx, bm_scores = self.bm25.retrieve([tokens], k=RETRIEVE_K)
            bm_ids = [int(self.bm25_ids[r]) for r in bm_idx[0]]

            qvec = encode_queries(self.embedder, [query])
            dense_scores, dense_idx = self.dense.search(qvec, RETRIEVE_K)
            dense_ids = [int(self.dense_ids[r]) for r in dense_idx[0]]
            top_cosine = float(dense_scores[0][0])  # best dense cosine available

            doc_ids, feats = self._build_features(
                self._rank_lookup(bm_ids, bm_scores[0]),
                self._rank_lookup(dense_ids, dense_scores[0]),
            )
            order = np.argsort(-self.booster.predict(feats))[:K]
            top = [doc_ids[i] for i in order]
            rows = self.con.execute(
                f"SELECT doc_id, title, answer_body FROM read_parquet('{self.corpus}') "
                f"WHERE doc_id IN ({','.join(map(str, top))})"
            ).fetchall()
        by_id = {r[0]: (r[1], r[2]) for r in rows}
        docs = [(d, by_id[d][0], by_id[d][1]) for d in top if d in by_id]
        return top_cosine, docs

    # ---- the full adaptive flow ----
    def answer(self, query: str, history: list[tuple[str, str]] | None = None,
               model: str | None = None) -> dict:
        """Run router -> (retrieve -> gate) -> generate. Returns answer + trace.

        history is the recent conversation as [(role, content), ...] ("user"/
        "assistant"), oldest first -- used only to keep replies coherent
        across turns. Retrieval itself still searches on the bare query text
        (no query rewriting yet), so a context-dependent follow-up like
        "give me another example" will retrieve poorly; known limitation.

        model overrides self.model for this call only (does not mutate shared
        state) -- lets one AdaptiveRAG instance safely serve concurrent
        requests for different conversations/models without a race.
        """
        hist = _format_history(history)
        route = self.route(query, history, model=model)
        if route == "ANSWER":
            text = self._generate(ANSWER_ALONE_PROMPT.format(history=hist, q=query), model=model)
            return {"route": "ANSWER", "searched": False, "top_cosine": None,
                    "gate": None, "used_docs": False, "retrieved": [], "answer": text}

        top_cosine, docs = self.retrieve(query)
        gate_open = top_cosine >= GATE_COSINE
        # doc_id is the real Stack Exchange answer Post.Id (corpus is built
        # straight off the official data dump -- see build_corpus.py), so
        # stackoverflow.com/a/{id} is a live, working short link to it.
        retrieved = [{"title": t, "url": f"https://stackoverflow.com/a/{d}"} for d, t, _ in docs]
        if gate_open:
            context = "\n\n".join(f"[{i}] {t}\n{b}" for i, (_, t, b) in enumerate(docs, 1))
            text = self._generate(ANSWER_WITH_DOCS_PROMPT.format(history=hist, q=query, context=context), model=model)
            return {"route": "SEARCH", "searched": True, "top_cosine": top_cosine,
                    "gate": "open", "used_docs": True, "retrieved": retrieved, "answer": text}
        # gate closed: retrieval too weak -> answer from model knowledge instead of junk
        text = self._generate(ANSWER_ALONE_PROMPT.format(history=hist, q=query), model=model)
        return {"route": "SEARCH", "searched": True, "top_cosine": top_cosine,
                "gate": "closed", "used_docs": False, "retrieved": retrieved, "answer": text}


def main() -> None:
    variant = sys.argv[1].lower() if len(sys.argv) > 1 else "a"
    if len(sys.argv) < 3:
        raise SystemExit(f"usage: python {sys.argv[0]} [variant] <path-to-model.gguf>")
    model = sys.argv[2]
    print("loading indexes + model ...")
    rag = AdaptiveRAG(variant, model=model)
    print(f"ready (variant {variant}, gate>={GATE_COSINE}). Type a question, empty line to quit.\n")
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        r = rag.answer(q)
        trace = f"[route={r['route']}"
        if r["searched"]:
            trace += f"  top_cos={r['top_cosine']:.3f}  gate={r['gate']}  used_docs={r['used_docs']}"
        trace += "]"
        print(trace)
        if r["retrieved"] and r["used_docs"]:
            for i, s in enumerate(r["retrieved"], 1):
                print(f"  [{i}] {s['title'][:70]}  {s['url']}")
        print("\n" + r["answer"] + "\n")


if __name__ == "__main__":
    main()
