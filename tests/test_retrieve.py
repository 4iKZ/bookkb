"""Tests for retrieval ranking and citation formatting — no model dependency."""

import numpy as np

from bookkb.retrieve import build_matrix, format_citation, format_text, search

import pytest


def _unit(vec):
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).astype(np.float32)


def _row(id, text, page, section, title, kind, path, vec):
    return {
        "id": id, "text": text, "page": page, "section": section,
        "embedding": vec, "title": title, "kind": kind, "path": path,
    }


def test_search_ranks_by_cosine():
    v1 = _unit([1.0, 0.0] * 192)
    v2 = _unit([0.0, 1.0] * 192)
    qvec = _unit([1.0, 0.0] * 192)
    chunks = [
        _row(1, "b", 1, None, "B", "pdf", "/b.pdf", v2.tobytes()),
        _row(2, "a", 1, None, "A", "pdf", "/a.pdf", v1.tobytes()),
    ]
    mat, _ = build_matrix(chunks)
    results = search(qvec, chunks, mat, k=2)
    assert results[0]["id"] == 2
    assert results[0]["rank"] == 1
    assert results[0]["score"] > results[1]["score"]


def test_search_k_clamped():
    v = _unit([1.0] * 384)
    chunks = [_row(i, "t", i, None, "T", "pdf", "/t.pdf", v.tobytes()) for i in range(3)]
    mat, _ = build_matrix(chunks)
    results = search(v, chunks, mat, k=100)
    assert len(results) == 3


def test_search_empty_chunks():
    assert search(np.zeros(384, dtype=np.float32), [], np.zeros((0, 384), dtype=np.float32), k=5) == []


def test_build_matrix_missing_embedding():
    chunks = [_row(1, "t", 1, None, "T", "pdf", "/t.pdf", None)]
    with pytest.raises(ValueError, match="no embedding"):
        build_matrix(chunks)


def test_build_matrix_corrupt_embedding():
    bad = np.zeros(128, dtype=np.float32).tobytes()  # wrong dim
    chunks = [_row(1, "t", 1, None, "T", "pdf", "/t.pdf", bad)]
    with pytest.raises(ValueError, match="corrupt"):
        build_matrix(chunks)


def test_format_citation_pdf():
    row = {"kind": "pdf", "title": "纳瓦尔宝典", "page": 42, "path": "x.pdf"}
    assert format_citation(row) == "《纳瓦尔宝典》第42页"


def test_format_citation_markdown_with_section():
    row = {"kind": "md", "title": "05", "page": None, "path": "/r/book/05-x.md", "section": "5. 标题"}
    assert format_citation(row) == "HowToLiveBetter/book/05-x.md · 5. 标题"


def test_format_citation_markdown_no_section():
    row = {"kind": "md", "title": "05", "page": None, "path": "/r/docs/note.md", "section": None}
    assert format_citation(row) == "HowToLiveBetter/docs/note.md"


def test_format_text_short():
    assert format_text("short") == "short"


def test_format_text_long_truncated():
    text = "x" * 700
    result = format_text(text)
    assert len(result) == 601
    assert result.endswith("…")
