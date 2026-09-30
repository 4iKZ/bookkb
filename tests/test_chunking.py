"""Tests for text splitting logic — no model dependency."""

from bookkb.ingest import split_by_len, split_markdown


def test_split_by_len_short_text_single_chunk():
    assert split_by_len("hello world", 1000) == ["hello world"]


def test_split_by_len_empty_text():
    assert split_by_len("", 1000) == []

def test_split_by_len_strips_whitespace():
    assert split_by_len("  hello  ", 1000) == ["hello"]


def test_split_by_len_splits_on_paragraph_boundaries():
    text = "a" * 600 + "\n\n" + "b" * 600
    parts = split_by_len(text, 1000)
    assert len(parts) == 2
    assert parts[0] == "a" * 600
    assert parts[1] == "b" * 600


def test_split_by_len_merges_short_paragraphs():
    text = "short one\n\nshort two"
    parts = split_by_len(text, 1000)
    assert parts == ["short one\n\nshort two"]


def test_split_by_len_hard_wraps_oversized_sentence():
    text = "x" * 2500
    parts = split_by_len(text, 1000)
    assert all(len(p) <= 1000 for p in parts)
    assert "".join(parts) == text


def test_split_by_len_splits_on_sentence_boundary():
    text = ". ".join(["word" * 200 for _ in range(5)])
    parts = split_by_len(text, 1000)
    assert len(parts) >= 2
    assert all(len(p) <= 1000 for p in parts)


def test_split_markdown_no_h2_single_chunk():
    text = "# Title\n\nSome content here."
    parts = split_markdown(text, 1000)
    assert len(parts) == 1
    assert parts[0][0] == "Title"
    assert "Some content" in parts[0][1]


def test_split_markdown_h2_sections():
    text = "# Doc\n\n## Section A\n\nContent A\n\n## Section B\n\nContent B"
    parts = split_markdown(text, 1000)
    # H1 line becomes content of the first section (titled "Doc");
    # two H2 sections follow — preserves original behavior.
    assert len(parts) == 3
    assert parts[0][0] == "Doc"
    assert parts[1][0] == "Section A"
    assert "Content A" in parts[1][1]
    assert parts[2][0] == "Section B"

def test_split_markdown_no_h1_uses_quanwen():
    text = "## Section\n\nContent"
    parts = split_markdown(text, 1000)
    assert parts[0][0] == "Section"


def test_split_markdown_long_section_further_split():
    text = "## Section\n\n" + "x" * 2500
    parts = split_markdown(text, 1000)
    assert len(parts) >= 2
    assert all(p[0] == "Section" for p in parts)
    assert all(len(p[1]) <= 1000 for p in parts)

