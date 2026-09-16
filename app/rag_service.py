"""Thin service layer between the desktop UI and AdaptiveRAG.

Loads the (expensive: FAISS index, BM25 index, embedder, LTR model)
AdaptiveRAG exactly once, then exposes the one thing the UI needs:

    answer(query, model=) -> run the full adaptive RAG flow with the given
                              model (an absolute path to a .gguf file) for
                              this call only, same trace dict
                              AdaptiveRAG.answer() returns

The model is passed per-call rather than set on shared state: this one
RagService/AdaptiveRAG instance is shared across all HTTP requests (FastAPI
dispatches them to a threadpool), so mutating a shared "current model" would
race between concurrent conversations using different models. The list of
available models (which .gguf files the user has actually picked before)
lives in the sqlite history, not here -- see app/db.py's list_recent_models.

Deliberately has no GUI imports, so it can be reused from a different
frontend, or exercised in a script/test without starting the app.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from adaptive_rag import AdaptiveRAG  # noqa: E402


class RagService:
    def __init__(self, variant: str = "a", device: str = "cuda") -> None:
        self.rag = AdaptiveRAG(variant=variant, device=device)

    def answer(self, query: str, history: list[tuple[str, str]] | None = None,
               model: str | None = None) -> dict:
        return self.rag.answer(query, history, model=model)

    def loading_status(self) -> str | None:
        """Path of the .gguf currently being loaded, or None -- polled by
        /api/status while a message is in flight, for the UI's loading bar."""
        return self.rag.loading_status()
