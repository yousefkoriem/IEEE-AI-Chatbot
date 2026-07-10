from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    def __init__(self, db_path: str) -> None:
        self._db_path = str(Path(db_path).resolve())
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TIMESTAMP NOT NULL DEFAULT (datetime('now')),
                updated_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages(conversation_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_conversations_session
                ON conversations(session_key, updated_at DESC);
            CREATE TABLE IF NOT EXISTS chunk_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                chunk_id TEXT NOT NULL,
                chunk_score REAL DEFAULT 0.0
            );
            CREATE INDEX IF NOT EXISTS idx_chunk_context_msg
                ON chunk_context(message_id);
            CREATE TABLE IF NOT EXISTS chunk_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                score INTEGER NOT NULL CHECK(score IN (-1, 1)),
                created_at TIMESTAMP NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_chunk_feedback_cid
                ON chunk_feedback(chunk_id);
        """)
        conn.commit()

    def get_or_create_conversation(self, session_key: str) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT id FROM conversations WHERE session_key = ? ORDER BY updated_at DESC LIMIT 1",
            (session_key,),
        )
        row = cursor.fetchone()
        if row:
            return int(row["id"])
        cursor = conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            (session_key, "New Chat"),
        )
        conn.commit()
        return int(cursor.lastrowid)

    def create_new_conversation(self, session_key: str) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            (session_key, "New Chat"),
        )
        conn.commit()
        return int(cursor.lastrowid)

    def add_message(self, conv_id: int, role: str, content: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conv_id, role, content),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
            (conv_id,),
        )
        conn.commit()

    def get_history(self, conv_id: int, limit: int = 50) -> list[dict[str, str]]:
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC LIMIT ?",
            (conv_id, limit),
        )
        return [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]

    def list_conversations(self, session_key: str, max_count: int = 50) -> list[dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT id, title, created_at, updated_at
               FROM conversations
               WHERE session_key = ?
               ORDER BY updated_at DESC
               LIMIT ?""",
            (session_key, max_count),
        )
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })
        return results

    def auto_title(self, conv_id: int, message: str) -> None:
        title = (message[:60] + "...") if len(message) > 60 else message
        conn = self._get_conn()
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ? AND title = 'New Chat'",
            (title, conv_id),
        )
        conn.commit()

    def rename_conversation(self, conv_id: int, title: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ?",
            (title, conv_id),
        )
        conn.commit()

    def delete_conversation(self, conv_id: int) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()

    def get_last_message_id(self, conv_id: int, role: str = "assistant") -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id FROM messages WHERE conversation_id = ? AND role = ? ORDER BY id DESC LIMIT 1",
            (conv_id, role),
        ).fetchone()
        return int(row["id"]) if row else 0

    def store_chunk_context(self, message_id: int, chunks: list[tuple[str, float]]) -> None:
        if not message_id or not chunks:
            return
        conn = self._get_conn()
        conn.executemany(
            "INSERT INTO chunk_context (message_id, chunk_id, chunk_score) VALUES (?, ?, ?)",
            [(message_id, cid, score) for cid, score in chunks],
        )
        conn.commit()

    def record_chunk_feedback(self, chunk_id: str, score: int) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO chunk_feedback (chunk_id, score) VALUES (?, ?)",
            (chunk_id, score),
        )
        conn.commit()

    def get_chunk_boosts(self) -> dict[str, float]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT chunk_id, SUM(score) as net FROM chunk_feedback GROUP BY chunk_id"
        ).fetchall()
        return {row["chunk_id"]: float(row["net"]) for row in rows if row["net"] != 0}

    def delete_old_chunk_feedback(self, days: int = 90) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM chunk_feedback WHERE created_at < datetime('now', ?)",
            (f"-{days} days",),
        )
        conn.commit()
        return cursor.rowcount

    def cleanup_old(self, days: int = 30) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM conversations WHERE updated_at < datetime('now', ?)",
            (f"-{days} days",),
        )
        conn.commit()
        return cursor.rowcount
