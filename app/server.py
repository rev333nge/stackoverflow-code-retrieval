"""Local FastAPI backend for the desktop app.

Binds to 127.0.0.1 only -- never exposed to the network. The Electron app
spawns this as a child process on launch and kills it on quit; the React
frontend talks to it exclusively over HTTP, never importing db.py or
rag_service.py directly.

Run standalone for development:
    python -m uvicorn app.server:app --host 127.0.0.1 --port 8756 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import db
from app.rag_service import RagService

service: RagService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    print("loading retrieval indexes + model onto GPU...")
    service = RagService()
    print("ready.")
    yield


app = FastAPI(lifespan=lifespan)

# Only exercised while developing against the Vite dev server (npm run dev).
# The packaged app loads its UI from this same server, so that path never
# makes a cross-origin request at all.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewConversation(BaseModel):
    title: str = "New chat"
    model: str


class ConversationUpdate(BaseModel):
    title: str | None = None
    model: str | None = None


class NewMessage(BaseModel):
    content: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/models")
def get_models():
    return service.available_models()


@app.get("/api/conversations")
def get_conversations():
    con = db.connect()
    return [dict(r) for r in db.list_conversations(con)]


@app.post("/api/conversations")
def create_conversation(body: NewConversation):
    con = db.connect()
    cid = db.create_conversation(con, body.title, body.model)
    return {"id": cid, "title": body.title, "model": body.model}


@app.patch("/api/conversations/{conversation_id}")
def update_conversation(conversation_id: int, body: ConversationUpdate):
    con = db.connect()
    if body.title is not None:
        db.rename_conversation(con, conversation_id, body.title)
    if body.model is not None:
        db.set_conversation_model(con, conversation_id, body.model)
    return {"ok": True}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: int):
    con = db.connect()
    db.delete_conversation(con, conversation_id)
    return {"ok": True}


@app.get("/api/conversations/{conversation_id}/messages")
def get_messages(conversation_id: int):
    con = db.connect()
    return db.list_messages(con, conversation_id)


@app.post("/api/conversations/{conversation_id}/messages")
def post_message(conversation_id: int, body: NewMessage):
    con = db.connect()
    conv = next((c for c in db.list_conversations(con) if c["id"] == conversation_id), None)
    if conv is None:
        raise HTTPException(404, "conversation not found")

    # last few turns only, for conversational continuity, not full replay
    history = [(m["role"], m["content"]) for m in db.list_messages(con, conversation_id)[-6:]]
    db.add_message(con, conversation_id, "user", body.content)

    service.set_model(conv["model"])
    try:
        result = service.answer(body.content, history)
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    sources = result["retrieved"] if result["used_docs"] else None

    msg_id = db.add_message(
        con, conversation_id, "assistant", result["answer"],
        route=result["route"], top_cosine=result["top_cosine"],
        gate=result["gate"], used_docs=result["used_docs"], sources=sources,
    )
    return {
        "id": msg_id,
        "role": "assistant",
        "content": result["answer"],
        "route": result["route"],
        "top_cosine": result["top_cosine"],
        "gate": result["gate"],
        "used_docs": result["used_docs"],
        "sources": sources,
    }
