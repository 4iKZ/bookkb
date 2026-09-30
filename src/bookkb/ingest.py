"""Document discovery + text splitting + ingest pipeline."""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

from bookkb import store
from bookkb.embed import encode_batched

# char budget: ~2.5 chars/token, leave room for "passage: " prefix
_CHAR_BUDGET = 1000
_H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def split_by_len(text: str, budget: int) -> list[str]:
    """Split text into pieces no longer than *budget* chars, on paragraph / sentence boundaries."""
    text = text.strip()
    if len(text) <= budget:
        return [text] if text else []
    out: list[str] = []
    paras = re.split(r"\n\s*\n", text)
    cur = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(p) <= budget:
            if len(cur) + len(p) + 2 <= budget:
                cur = (cur + "\n\n" + p) if cur else p
            else:
                if cur:
                    out.append(cur)
                cur = p
        else:
            if cur:
                out.append(cur)
                cur = ""
            sents = re.split(r"(?<=[.。！？!?；;])\s+", p)
            sc = ""
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                if len(s) <= budget:
                    if len(sc) + len(s) + 1 <= budget:
                        sc = (sc + " " + s) if sc else s
                    else:
                        if sc:
                            out.append(sc)
                        sc = s
                else:
                    if sc:
                        out.append(sc)
                        sc = ""
                    for i in range(0, len(s), budget):
                        out.append(s[i : i + budget])
            if sc:
                out.append(sc)
    if cur:
        out.append(cur)
    return [c for c in out if c.strip()]


def split_markdown(text: str, budget: int) -> list[tuple[str, str]]:
    """Split markdown into (section_title, chunk_text) pairs."""
    lines = text.split("\n")
    h1 = None
    for ln in lines:
        if ln.startswith("# "):
            h1 = ln[2:].strip()
            break
    sections: list[tuple[str, list[str]]] = []
    cur_title = h1 or "全文"
    cur_lines: list[str] = []
    found_h2 = False
    for ln in lines:
        m = re.match(r"^##\s+(.+)$", ln)
        if m:
            if cur_lines:
                sections.append((cur_title, cur_lines))
            cur_title = m.group(1).strip()
            cur_lines = []
            found_h2 = True
        else:
            cur_lines.append(ln)
    if cur_lines:
        sections.append((cur_title, cur_lines))

    out: list[tuple[str, str]] = []
    for sec_title, sec_lines in sections:
        sec_text = "\n".join(sec_lines).strip()
        if not sec_text:
            continue
        if found_h2 and len(sec_text) <= budget:
            out.append((sec_title, sec_text))
        else:
            for piece in split_by_len(sec_text, budget):
                out.append((sec_title, piece))
    return out


def iter_pdfs(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".pdf":
        return [path]
    if path.is_dir():
        return sorted(f for f in path.glob("**/*") if f.is_file() and f.suffix.lower() == ".pdf")
    return []


def iter_markdown(repo_dir: Path) -> list[Path]:
    """book/*.md and docs/*.md (top-level only, not sub-dirs)."""
    out: list[Path] = []
    for sub in ("book", "docs"):
        d = repo_dir / sub
        if d.is_dir():
            out.extend(sorted(d.glob("*.md")))
    return out


def ingest_pdf(path: Path, conn) -> tuple[int, int, int]:
    """Return (files, chunks, skipped)."""
    from pypdf import PdfReader

    sha = _sha256(path)
    if store.find_document_by_sha(conn, sha) is not None:
        print(f"  [skip] already ingested: {path.name}")
        return 0, 0, 1

    title = path.stem
    reader = PdfReader(str(path))
    texts: list[tuple[int, str]] = []
    skipped_pages = 0
    for i, page in enumerate(reader.pages, 1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = txt.strip()
        if not txt:
            skipped_pages += 1
            continue
        texts.append((i, txt))
    if skipped_pages:
        print(f"  [warn] {path.name}: {skipped_pages} page(s) had no extractable text (scanned?) — skipped")
    if not texts:
        print(f"  [warn] {path.name}: no text extracted at all, skipping")
        return 0, 0, 1

    chunk_texts: list[str] = []
    chunk_pages: list[int] = []
    for page_no, txt in texts:
        for piece in split_by_len(txt, _CHAR_BUDGET):
            chunk_texts.append(piece)
            chunk_pages.append(page_no)

    all_vecs = encode_batched(chunk_texts, "passage: ")

    doc_id = store.insert_document(conn, title, "pdf", str(path), sha, _now())
    for piece, page_no, vec in zip(chunk_texts, chunk_pages, all_vecs):
        store.insert_chunk(conn, doc_id, page_no, None, piece, vec.tobytes())
    conn.commit()
    print(f"  [ok] {path.name}: {len(chunk_texts)} chunks from {len(texts)} pages")
    return 1, len(chunk_texts), 0


def ingest_markdown(path: Path, conn, repo_label: str) -> tuple[int, int, int]:
    sha = _sha256(path)
    if store.find_document_by_sha(conn, sha) is not None:
        print(f"  [skip] already ingested: {repo_label}/{path.name}")
        return 0, 0, 1

    text = path.read_text(encoding="utf-8", errors="replace")
    parts = split_markdown(text, _CHAR_BUDGET)
    if not parts:
        print(f"  [warn] {path.name}: no text, skipping")
        return 0, 0, 1

    chunk_texts = [p[1] for p in parts]
    sections = [p[0] for p in parts]
    all_vecs = encode_batched(chunk_texts, "passage: ")

    title = path.stem
    doc_id = store.insert_document(conn, title, "md", str(path), sha, _now())
    for piece, sec, vec in zip(chunk_texts, sections, all_vecs):
        store.insert_chunk(conn, doc_id, None, sec, piece, vec.tobytes())
    conn.commit()
    print(f"  [ok] {repo_label}/{path.name}: {len(chunk_texts)} chunks")
    return 1, len(chunk_texts), 0
