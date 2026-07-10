from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class ShareManager:
    """SQLite-backed manager for shareable answer links."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute(
                """CREATE TABLE IF NOT EXISTS shared_answers (
                    share_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT,
                    confidence TEXT,
                    created_at REAL NOT NULL
                )"""
            )
        return self._local.conn

    def create_share(self, question: str, answer: str, sources: list[str] | None = None, confidence: str = "") -> str:
        raw = f"{question}:{answer}:{time.time()}"
        share_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        conn = self._conn()
        conn.execute(
            "INSERT OR IGNORE INTO shared_answers (share_id, question, answer, sources, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (share_id, question, answer, "\n".join(sources or []), confidence, time.time()),
        )
        conn.commit()
        return share_id

    def get_share(self, share_id: str) -> dict | None:
        conn = self._conn()
        row = conn.execute(
            "SELECT share_id, question, answer, sources, confidence, created_at FROM shared_answers WHERE share_id = ?",
            (share_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "share_id": row[0],
            "question": row[1],
            "answer": row[2],
            "sources": row[3].split("\n") if row[3] else [],
            "confidence": row[4] or "",
            "created_at": row[5],
        }
