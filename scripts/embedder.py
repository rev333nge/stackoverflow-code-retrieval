"""Shared embedding model for dense retrieval (Phase 3).

bge-v1.5 wants an instruction on QUERIES only; documents are encoded as-is.
Vectors are L2-normalized so inner product == cosine similarity.
Kept in one place so documents and queries are always encoded the same way.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def load_model(device: str = "cuda") -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME, device=device)
    if device == "cuda":
        model.half()   # fp16 inference: ~2x faster on GPU, no meaningful quality loss
    return model


def encode_documents(model, texts, batch_size: int = 256):
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")


def encode_queries(model, texts, batch_size: int = 256):
    q = [QUERY_INSTRUCTION + t for t in texts]
    return model.encode(
        q,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
