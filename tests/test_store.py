"""Tests for store read/write — no model dependency."""

import os
import tempfile

import numpy as np
import pytest

from bookkb import store


@pytest.fixture
def conn():
    with tempfile.TemporaryDirectory() as d:
        old = os.environ.get("BOOKKB_DB")
        os.environ["BOOKKB_DB"] = os.path.join(d, "test.sqlite")
        import importlib
        importlib.reload(store)
        c = store.connect()
        yield c
        c.close()
        if old is not None:
            os.environ["BOOKKB_DB"] = old
        else:
            os.environ.pop("BOOKKB_DB", None)


def test_connect_creates_tables(conn):
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {t[0] for t in tables}
    assert "documents" in names
    assert "chunks" in names


def test_insert_and_find_document(conn):
    doc_id = store.insert_document(conn, "title", "md", "/path", "sha123", "2024-01-01T00:00:00")
    assert doc_id > 0
    found = store.find_document_by_sha(conn, "sha123")
    assert found == doc_id
    assert store.find_document_by_sha(conn, "nope") is None


def test_insert_and_fetch_chunks(conn):
    doc_id = store.insert_document(conn, "title", "md", "/path", "sha123", "2024-01-01")
    vec = np.array([1.0, 0.0, 0.0] * 128, dtype=np.float32)
    store.insert_chunk(conn, doc_id, None, "Section A", "chunk text", vec.tobytes())
    store.insert_chunk(conn, doc_id, None, "Section B", "another", vec.tobytes())
    conn.commit()

    chunks = store.fetch_all_chunks(conn)
    assert len(chunks) == 2
    assert chunks[0]["title"] == "title"
    assert chunks[0]["section"] == "Section A"
    assert chunks[0]["text"] == "chunk text"
    assert chunks[0]["kind"] == "md"


def test_fetch_empty_db(conn):
    assert store.fetch_all_chunks(conn) == []


def test_sha_uniqueness(conn):
    store.insert_document(conn, "a", "md", "/a", "sha", "2024")
    conn.commit()
    with pytest.raises(Exception):
        store.insert_document(conn, "b", "md", "/b", "sha", "2024")
        conn.commit()
