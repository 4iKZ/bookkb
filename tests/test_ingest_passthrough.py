"""Test page_no passthrough in ingest_pdf — no model dependency."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from bookkb import ingest, store


class FakeEmbedder:
    def encode(self, texts, prefix):
        return np.ones((len(texts), 384), dtype=np.float32)


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


def test_ingest_pdf_page_no_passthrough(conn, tmp_path):
    # Build a fake PDF by monkeypatching PdfReader
    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, path):
            self.pages = [
                FakePage("page one content here"),
                FakePage("page two content here"),
                FakePage("page three content here"),
            ]

    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("fake")  # content doesn't matter; reader is mocked

    with patch("pypdf.PdfReader", FakeReader):
        files, chunks, skipped = ingest.ingest_pdf(pdf_path, conn)

    assert files == 1
    assert skipped == 0

    rows = store.fetch_all_chunks(conn)
    assert len(rows) == 3
    # Each chunk should carry its page number, not None
    pages = [r["page"] for r in rows]
    assert pages == [1, 2, 3]
    assert all(p is not None for p in pages)


def test_ingest_markdown_section_passthrough(conn, tmp_path):
    md = tmp_path / "note.md"
    md.write_text("# Title\n\n## Section A\n\nContent A\n\n## Section B\n\nContent B", encoding="utf-8")
    files, chunks, skipped = ingest.ingest_markdown(md, conn, "repo")

    assert files == 1
    rows = store.fetch_all_chunks(conn)
    sections = [r["section"] for r in rows if r["section"] is not None]
    assert "Section A" in sections
    assert "Section B" in sections
