"""SQLite storage for chat history: conversations and their messages.

One file, no server (sqlite3 is in the Python standard library). Schema:

    conversations(id, title, model, created_at)
    messages(id, conversation_id, role, content, created_at,
             route, top_cosine, gate, used_docs, sources_json)

The trace columns (route/top_cosine/gate/used_docs/sources_json) are only
filled in for assistant messages -- they record what AdaptiveRAG.answer()
decided, so the UI can show "why" an answer looks the way it does.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    model      TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    created_at      REAL NOT NULL,
    route           TEXT,
    top_cosine      REAL,
    gate            TEXT,
    used_docs       INTEGER,
    sources_json    TEXT
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def create_conversation(con: sqlite3.Connection, title: str, model: str) -> int:
    cur = con.execute(
        "INSERT INTO conversations (title, model, created_at) VALUES (?, ?, ?)",
        (title, model, time.time()),
    )
    con.commit()
    return cur.lastrowid


def list_conversations(con: sqlite3.Connection) -> list[sqlite3.Row]:
    con.row_factory = sqlite3.Row
    return con.execute(
        "SELECT id, title, model, created_at FROM conversations ORDER BY created_at DESC"
    ).fetchall()


def rename_conversation(con: sqlite3.Connection, conversation_id: int, title: str) -> None:
    con.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
    con.commit()


def set_conversation_model(con: sqlite3.Connection, conversation_id: int, model: str) -> None:
    con.execute("UPDATE conversations SET model = ? WHERE id = ?", (model, conversation_id))
    con.commit()


def delete_conversation(con: sqlite3.Connection, conversation_id: int) -> None:
    con.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    con.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    con.commit()


def add_message(
    con: sqlite3.Connection,
    conversation_id: int,
    role: str,
    content: str,
    *,
    route: str | None = None,
    top_cosine: float | None = None,
    gate: str | None = None,
    used_docs: bool | None = None,
    sources: list[str] | None = None,
) -> int:
    cur = con.execute(
        """INSERT INTO messages
           (conversation_id, role, content, created_at, route, top_cosine, gate, used_docs, sources_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            conversation_id,
            role,
            content,
            time.time(),
            route,
            top_cosine,
            gate,
            None if used_docs is None else int(used_docs),
            None if sources is None else json.dumps(sources),
        ),
    )
    con.commit()
    return cur.lastrowid


def list_messages(con: sqlite3.Connection, conversation_id: int) -> list[dict]:
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    ).fetchall()
    messages = []
    for r in rows:
        d = dict(r)
        d["used_docs"] = None if d["used_docs"] is None else bool(d["used_docs"])
        d["sources"] = json.loads(d["sources_json"]) if d["sources_json"] else None
        del d["sources_json"]
        messages.append(d)
    return messages
