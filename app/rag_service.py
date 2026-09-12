"""Thin service layer between the desktop UI and AdaptiveRAG.

Loads the (expensive: FAISS index, BM25 index, embedder, LTR model)
AdaptiveRAG exactly once, then exposes the three things the UI needs:

    available_models() -> model names Ollama currently has pulled,
                           for the model-picker dropdown
    set_model(name)    -> switch which model routes/generates. Cheap:
                           only changes which name is sent to Ollama,
                           none of the indexes are touched or reloaded.
    answer(query)      -> run the full adaptive RAG flow, same trace
                           dict AdaptiveRAG.answer() returns

Deliberately has no GUI imports, so it can be reused from a different
frontend, or exercised in a script/test without starting the app.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from adaptive_rag import AdaptiveRAG, list_models  # noqa: E402


class RagService:
    def __init__(self, variant: str = "a", device: str = "cuda") -> None:
        self.rag = AdaptiveRAG(variant=variant, device=device)

    def available_models(self) -> list[str]:
        try:
            return list_models()
        except Exception:
            return []

    def set_model(self, model: str) -> None:
        self.rag.set_model(model)

    @property
    def current_model(self) -> str:
        return self.rag.model

    def answer(self, query: str, history: list[tuple[str, str]] | None = None) -> dict:
        return self.rag.answer(query, history)
