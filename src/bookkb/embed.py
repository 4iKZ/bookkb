"""ONNX embedding — volatile infrastructure behind an injectable seam.

`encode(texts, prefix, embedder=...)` lets callers (and tests) pass any object
with `.encode(texts, prefix) -> ndarray`; the default lazy-loads the real ONNX model.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

EMBED_DIM = 384
# e5-small: 384 dims, 512 max tokens — keep chunks well under that.
_MAX_TOKENS = 512


def _model_dir() -> Path:
    override = os.environ.get("BOOKKB_MODEL_DIR")
    if override:  # dir containing tokenizer.json + onnx/model.onnx
        return Path(override)
    roots = [
        os.environ.get("HF_HOME"),
        str(Path.home() / ".cache" / "huggingface"),
        "/home/hatch/.hf-models",  # legacy local path
    ]
    for root in filter(None, roots):
        snaps = sorted(Path(root, "models--intfloat--multilingual-e5-small", "snapshots").glob("*/onnx"))
        if snaps:
            return snaps[0].parent  # snapshot dir (tokenizer.json + onnx/)
    raise RuntimeError("e5-small ONNX snapshot not found; set BOOKKB_MODEL_DIR or HF_HOME")


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
        self._tok.enable_truncation(max_length=_MAX_TOKENS)

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


# lazy module-level singleton so ingest/ask share one session
_EMBEDDER = _Embedder()


def encode(texts: list[str], prefix: str, embedder=None) -> np.ndarray:
    """Encode *texts* with *prefix*; inject *embedder* for tests (duck-typed)."""
    emb = embedder if embedder is not None else _EMBEDDER
    return emb.encode(texts, prefix)


def encode_batched(texts: list[str], prefix: str, batch: int = 16, embedder=None) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    vecs = []
    for i in range(0, len(texts), batch):
        vecs.append(encode(texts[i : i + batch], prefix, embedder=embedder))
    return np.vstack(vecs)
