"""Cosine retrieval + citation formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bookkb.embed import EMBED_DIM


def build_matrix(chunks: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build (matrix, rows) from store rows; raises on corrupt/missing embedding."""
    mat = np.zeros((len(chunks), EMBED_DIM), dtype=np.float32)
    for i, row in enumerate(chunks):
        if row["embedding"] is None:
            raise ValueError(f"chunk {row['id']} has no embedding; delete the DB file and re-ingest.")
        vec = np.frombuffer(row["embedding"], dtype=np.float32)
        if vec.size != EMBED_DIM:
            raise ValueError(f"chunk {row['id']} has a corrupt embedding; delete the DB file and re-ingest.")
        mat[i] = vec
    return mat, chunks


def search(query_vec: np.ndarray, chunks: list[dict[str, Any]], mat: np.ndarray, k: int) -> list[dict[str, Any]]:
    """Return top-*k* chunks by cosine similarity, ranked desc with `score` added.

    *mat* is the pre-built matrix from `build_matrix` — built once, not rebuilt here.
    """
    if not chunks:
        return []
    sims = mat @ query_vec  # both L2-normalised → dot = cosine
    kk = min(max(k, 1), len(chunks))
    top = np.argsort(sims)[::-1][:kk]
    results = []
    for rank, idx in enumerate(top, 1):
        row = chunks[idx]
        results.append({**row, "rank": rank, "score": float(sims[idx])})
    return results


def format_citation(row: dict[str, Any]) -> str:
    kind = row["kind"]
    title = row["title"]
    if kind == "pdf":
        return f"《{title}》第{row['page']}页"
    fname = Path(row["path"]).name
    cite = f"HowToLiveBetter/{Path(row['path']).parent.name}/{fname}"
    if row.get("section"):
        cite += f" · {row['section']}"
    return cite


def format_text(text: str, limit: int = 600) -> str:
    if len(text) > limit:
        return text[:limit] + "…"
    return text
