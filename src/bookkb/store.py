"""SQLite store — owns schema, connection, and typed row access.

Hides raw sqlite tuples from callers; returns dicts so callers never touch
positional indices.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

# Default DB lives under CWD (not __file__) so it works when installed as a package.
DEFAULT_DB = Path.cwd() / "data" / "bookkb.sqlite"
DB_PATH = Path(os.environ.get("BOOKKB_DB", DEFAULT_DB))


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            kind TEXT NOT NULL,
            path TEXT NOT NULL,
            sha256 TEXT NOT NULL UNIQUE,
            ingested_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            page INTEGER,
            section TEXT,
            text TEXT NOT NULL,
            embedding BLOB,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        )"""
    )
    conn.commit()
    return conn


def insert_document(conn: sqlite3.Connection, title: str, kind: str, path: str, sha: str, now: str) -> int:
    cur = conn.execute(
        "INSERT INTO documents(title, kind, path, sha256, ingested_at) VALUES(?,?,?,?,?)",
        (title, kind, path, sha, now),
    )
    return cur.lastrowid


def insert_chunk(conn: sqlite3.Connection, doc_id: int, page: int | None, section: str | None, text: str, embedding: bytes) -> None:
    conn.execute(
        "INSERT INTO chunks(doc_id, page, section, text, embedding) VALUES(?,?,?,?,?)",
        (doc_id, page, section, text, embedding),
    )


def find_document_by_sha(conn: sqlite3.Connection, sha: str) -> int | None:
    row = conn.execute("SELECT id FROM documents WHERE sha256=?", (sha,)).fetchone()
    return row[0] if row else None


def fetch_all_chunks(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT c.id, c.text, c.page, c.section, c.embedding, d.title, d.kind, d.path "
        "FROM chunks c JOIN documents d ON c.doc_id=d.id"
    ).fetchall()
    return [
        {
            "id": r[0],
            "text": r[1],
            "page": r[2],
            "section": r[3],
            "embedding": r[4],
            "title": r[5],
            "kind": r[6],
            "path": r[7],
        }
        for r in rows
    ]
