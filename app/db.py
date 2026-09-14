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

import functools
import json
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "history.db"

# The FastAPI backend opens one connection at startup (check_same_thread=False,
# since sync request handlers run on a threadpool) and reuses it for every
# request instead of reconnecting each time. A raw sqlite3.Connection isn't
# safe for genuinely concurrent use from multiple threads even with that flag
# off, so every function below that touches `con` is serialized through this
# lock -- cheap for a local single-user app, and avoids "database is locked".
_lock = threading.Lock()


def _synchronized(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _lock:
            return fn(*args, **kwargs)
    return wrapper

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


def connect(check_same_thread: bool = True) -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=check_same_thread)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


@_synchronized
def create_conversation(con: sqlite3.Connection, title: str, model: str) -> int:
    cur = con.execute(
        "INSERT INTO conversations (title, model, created_at) VALUES (?, ?, ?)",
        (title, model, time.time()),
    )
    con.commit()
    return cur.lastrowid


@_synchronized
def list_conversations(con: sqlite3.Connection) -> list[sqlite3.Row]:
    return con.execute(
        "SELECT id, title, model, created_at FROM conversations ORDER BY created_at DESC"
    ).fetchall()


@_synchronized
def get_conversation(con: sqlite3.Connection, conversation_id: int) -> sqlite3.Row | None:
    return con.execute(
        "SELECT id, title, model, created_at FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()


@_synchronized
def rename_conversation(con: sqlite3.Connection, conversation_id: int, title: str) -> None:
    con.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
    con.commit()


@_synchronized
def set_conversation_model(con: sqlite3.Connection, conversation_id: int, model: str) -> None:
    con.execute("UPDATE conversations SET model = ? WHERE id = ?", (model, conversation_id))
    con.commit()


@_synchronized
def delete_conversation(con: sqlite3.Connection, conversation_id: int) -> None:
    con.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    con.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    con.commit()


@_synchronized
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


@_synchronized
def delete_message(con: sqlite3.Connection, message_id: int) -> None:
    con.execute("DELETE FROM messages WHERE id = ?", (message_id,))
    con.commit()


def _row_to_message(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["used_docs"] = None if d["used_docs"] is None else bool(d["used_docs"])
    d["sources"] = json.loads(d["sources_json"]) if d["sources_json"] else None
    del d["sources_json"]
    return d


@_synchronized
def list_messages(con: sqlite3.Connection, conversation_id: int) -> list[dict]:
    rows = con.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    ).fetchall()
    return [_row_to_message(r) for r in rows]


@_synchronized
def list_recent_messages(con: sqlite3.Connection, conversation_id: int, limit: int = 6) -> list[dict]:
    """Last `limit` messages, oldest first -- for threading into RAG history
    without fetching and re-decoding a whole (potentially long) conversation."""
    rows = con.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
        (conversation_id, limit),
    ).fetchall()
    return [_row_to_message(r) for r in reversed(rows)]
