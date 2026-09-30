#!/usr/bin/env python3
"""bookkb — personal book knowledge-base CLI.

Ingests PDFs and markdown repos into a SQLite store with local ONNX embeddings,
then answers queries with cosine-similarity evidence retrieval (no LLM synthesis).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

# --- paths ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB = PROJECT_ROOT / "data" / "bookkb.sqlite"
DB_PATH = Path(os.environ.get("BOOKKB_DB", DEFAULT_DB))

# e5-small: 384 dims, 512 max tokens — keep chunks well under that.
_EMBED_DIM = 384
_MAX_TOKENS = 512
# char budget: ~2.5 chars/token, leave room for "passage: " prefix
_CHAR_BUDGET = 1000


def _model_dir() -> Path:
    override = os.environ.get("BOOKKB_MODEL_DIR")
    if override:  # dir containing tokenizer.json + onnx/model.onnx
        return Path(override)
    base = Path("/home/hatch/.hf-models/models--intfloat--multilingual-e5-small/snapshots")
    snaps = sorted(base.glob("*/onnx"))
    if not snaps:
        raise RuntimeError("e5-small ONNX snapshot not found under " + str(base))
    return snaps[0].parent  # snapshot dir (tokenizer.json + onnx/)


# --- embedding (lazy-loaded so ingest/ask share one session) ----------

class _Embedder:
    def __init__(self):
        self._session = None
        self._tok = None

    def _ensure(self):
        if self._session is not None:
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer

        mdir = _model_dir()
        onnx_path = mdir / "onnx" / "model.onnx"
        if not onnx_path.exists():
            raise RuntimeError(f"ONNX model not found: {onnx_path}")
        self._session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self._tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
        self._tok.enable_padding(pad_id=0, pad_token="<pad>")
        # e5-small positional limit is 512; truncate so long chunks don't crash the graph
        self._tok.enable_truncation(max_length=512)

    def encode(self, texts: list[str], prefix: str) -> np.ndarray:
        """Return L2-normalised (n, 384) float32 matrix."""
        self._ensure()
        prefixed = [f"{prefix}{t}" for t in texts]
        encs = self._tok.encode_batch(prefixed)
        input_ids = np.array([e.ids for e in encs], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)
        out = self._session.run(
            ["last_hidden_state"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )[0]
        # mean-pool over non-pad tokens, masking out padding
        mask = attention_mask[:, :, None].astype(np.float32)
        summed = (out * mask).sum(axis=1)
        counts = mask.sum(axis=1).clip(min=1e-9)
        vecs = (summed / counts).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return vecs / norms


_EMBEDDER = _Embedder()


# --- DB ----------------------------------------------------------------

def _connect() -> sqlite3.Connection:
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


# --- text splitting ---------------------------------------------------

def _split_by_len(text: str, budget: int) -> list[str]:
    """Split text into pieces no longer than *budget* chars, on paragraph / sentence boundaries."""
    text = text.strip()
    if len(text) <= budget:
        return [text] if text else []
    out: list[str] = []
    # try paragraph splits first
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
            # hard-wrap long paragraph on sentence boundaries
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


# --- PDF --------------------------------------------------------------

def _ingest_pdf(path: Path, conn: sqlite3.Connection) -> tuple[int, int, int]:
    """Return (files, chunks, skipped)."""
    from pypdf import PdfReader

    sha = _sha256(path)
    row = conn.execute(
        "SELECT id FROM documents WHERE sha256=?", (sha,)
    ).fetchone()
    if row:
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

    # build chunks: page-level, split overlong pages
    chunk_texts: list[str] = []
    chunk_meta: list[tuple[int, str | None]] = []  # (page, section=None)
    for page_no, txt in texts:
        pieces = _split_by_len(txt, _CHAR_BUDGET)
        for piece in pieces:
            chunk_texts.append(piece)
            chunk_meta.append((page_no, None))

    # embed in batches
    all_vecs = _embed_batched(chunk_texts, "passage: ")

    cur = conn.execute(
        "INSERT INTO documents(title, kind, path, sha256, ingested_at) VALUES(?,?,?,?,?)",
        (title, "pdf", str(path), sha, _now()),
    )
    doc_id = cur.lastrowid
    for piece, (page_no, _sec), vec in zip(chunk_texts, chunk_meta, all_vecs):
        conn.execute(
            "INSERT INTO chunks(doc_id, page, section, text, embedding) VALUES(?,?,?,?,?)",
            (doc_id, page_no, None, piece, vec.tobytes()),
        )
    conn.commit()
    print(f"  [ok] {path.name}: {len(chunk_texts)} chunks from {len(texts)} pages")
    return 1, len(chunk_texts), 0


# --- Markdown ---------------------------------------------------------

# H2 markers like "## " or the first H1 acts as file title; we split on H2.
_H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def _split_markdown(text: str, budget: int) -> list[tuple[str, str]]:
    """Split markdown into (section_title, chunk_text) pairs.
    Long files: split on H2 headers; within a section, further split by length.
    Short files: single chunk with section = first H1 title or '全文'.
    """
    lines = text.split("\n")
    # find first H1 as doc title
    h1 = None
    for ln in lines:
        if ln.startswith("# "):
            h1 = ln[2:].strip()
            break
    # find H2 split points
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
            for piece in _split_by_len(sec_text, budget):
                out.append((sec_title, piece))
    return out


def _ingest_markdown(path: Path, conn: sqlite3.Connection, repo_label: str) -> tuple[int, int, int]:
    sha = _sha256(path)
    row = conn.execute("SELECT id FROM documents WHERE sha256=?", (sha,)).fetchone()
    if row:
        print(f"  [skip] already ingested: {repo_label}/{path.name}")
        return 0, 0, 1

    text = path.read_text(encoding="utf-8", errors="replace")
    parts = _split_markdown(text, _CHAR_BUDGET)
    if not parts:
        print(f"  [warn] {path.name}: no text, skipping")
        return 0, 0, 1

    chunk_texts = [p[1] for p in parts]
    sections = [p[0] for p in parts]
    all_vecs = _embed_batched(chunk_texts, "passage: ")

    title = path.stem
    cur = conn.execute(
        "INSERT INTO documents(title, kind, path, sha256, ingested_at) VALUES(?,?,?,?,?)",
        (title, "md", str(path), sha, _now()),
    )
    doc_id = cur.lastrowid
    for piece, sec, vec in zip(chunk_texts, sections, all_vecs):
        conn.execute(
            "INSERT INTO chunks(doc_id, page, section, text, embedding) VALUES(?,?,?,?,?)",
            (doc_id, None, sec, piece, vec.tobytes()),
        )
    conn.commit()
    print(f"  [ok] {repo_label}/{path.name}: {len(chunk_texts)} chunks")
    return 1, len(chunk_texts), 0


# --- helpers ----------------------------------------------------------

def _embed_batched(texts: list[str], prefix: str, batch: int = 16) -> np.ndarray:
    if not texts:
        return np.zeros((0, _EMBED_DIM), dtype=np.float32)
    vecs = []
    for i in range(0, len(texts), batch):
        vecs.append(_EMBEDDER.encode(texts[i : i + batch], prefix))
    return np.vstack(vecs)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _iter_pdfs(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".pdf":
        return [path]
    if path.is_dir():
        return sorted(f for f in path.glob("**/*") if f.is_file() and f.suffix.lower() == ".pdf")
    return []


def _iter_markdown(repo_dir: Path) -> list[Path]:
    """book/*.md and docs/*.md (top-level only, not sub-dirs)."""
    out: list[Path] = []
    for sub in ("book", "docs"):
        d = repo_dir / sub
        if d.is_dir():
            out.extend(sorted(d.glob("*.md")))
    return out


# --- commands ---------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> int:
    conn = _connect()
    total_files = total_chunks = total_skipped = 0
    ok_paths = 0
    for raw in args.paths:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            print(f"[error] path not found: {p}")
            continue
        ok_paths += 1
        # PDF file or directory of PDFs
        pdfs = _iter_pdfs(p)
        if pdfs:
            print(f"\n== PDF: {p} ==")
            for pdf in pdfs:
                f, c, s = _ingest_pdf(pdf, conn)
                total_files += f
                total_chunks += c
                total_skipped += s
        # repo dir with book/ and docs/
        if p.is_dir() and ((p / "book").is_dir() or (p / "docs").is_dir()):
            mds = _iter_markdown(p)
            if mds:
                label = p.name
                print(f"\n== Markdown: {p} ==")
                for md in mds:
                    f, c, s = _ingest_markdown(md, conn, label)
                    total_files += f
                    total_chunks += c
                    total_skipped += s
    if not ok_paths:
        print("[error] no valid paths given; nothing ingested")
        conn.close()
        return 1
    conn.close()
    print(f"\n=== Ingest complete ===")
    print(f"  files ingested : {total_files}")
    print(f"  chunks created : {total_chunks}")
    print(f"  skipped (dup)  : {total_skipped}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    conn = _connect()
    rows = conn.execute(
        "SELECT c.id, c.text, c.page, c.section, c.embedding, d.title, d.kind, d.path FROM chunks c JOIN documents d ON c.doc_id=d.id"
    ).fetchall()
    if not rows:
        print("[error] no chunks in database. Run `bookkb ingest <path>` first.")
        conn.close()
        return 1
    # build matrix
    texts, pages, sections, titles, kinds, paths = [], [], [], [], [], []
    mat = np.zeros((len(rows), _EMBED_DIM), dtype=np.float32)
    for i, r in enumerate(rows):
        if r[4] is None:
            print(f"[error] chunk {r[0]} has no embedding; delete the DB file and re-ingest.")
            conn.close()
            return 1
        vec = np.frombuffer(r[4], dtype=np.float32)
        if vec.size != _EMBED_DIM:
            print(f"[error] chunk {r[0]} has a corrupt embedding; delete the DB file and re-ingest.")
            conn.close()
            return 1
        texts.append(r[1])
        pages.append(r[2])
        sections.append(r[3])
        titles.append(r[5])
        kinds.append(r[6])
        paths.append(r[7])
        mat[i] = vec
    # query
    qvec = _EMBEDDER.encode([args.question], "query: ")[0]
    sims = mat @ qvec  # both L2-normalised → dot = cosine
    k = min(max(args.k, 1), len(rows))
    top = np.argsort(sims)[::-1][:k]
    for rank, idx in enumerate(top, 1):
        score = float(sims[idx])
        title = titles[idx]
        page = pages[idx]
        section = sections[idx]
        kind = kinds[idx]
        if kind == "pdf":
            cite = f"《{title}》第{page}页"
        else:
            fname = Path(paths[idx]).name
            cite = f"HowToLiveBetter/{Path(paths[idx]).parent.name}/{fname}"
            if section:
                cite += f" · {section}"
        text = texts[idx]
        if len(text) > 600:
            text = text[:600] + "…"
        print(f"[{rank}] {cite} ({score:.2f})")
        print(text)
        print()
    conn.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="bookkb",
        description="Personal book knowledge-base: ingest PDFs & markdown, ask for evidence.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest PDF files/dirs or a cloned repo dir.")
    p_ingest.add_argument("paths", nargs="+", help="PDF file/dir or repo dir containing book/ and docs/")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question; returns top-k evidence chunks.")
    p_ask.add_argument("question", help="Question text")
    p_ask.add_argument("-k", type=int, default=8, help="Number of results (default 8)")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
