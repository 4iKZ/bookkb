"""Tests for embed seam: prefix passthrough, batching, output shape — no model."""

import numpy as np
import pytest

from bookkb.embed import EMBED_DIM, encode, encode_batched


class FakeEmbedder:
    def __init__(self, dim=EMBED_DIM):
        self.calls = []
        self.dim = dim

    def encode(self, texts, prefix):
        self.calls.append((list(texts), prefix))
        return np.ones((len(texts), self.dim), dtype=np.float32)


def test_encode_prefix_passthrough():
    fake = FakeEmbedder()
    encode(["hello"], "query: ", embedder=fake)
    assert fake.calls == [(  ["hello"], "query: ")]  # space included


def test_encode_batched_batches():
    fake = FakeEmbedder()
    texts = [f"t{i}" for i in range(35)]
    out = encode_batched(texts, "passage: ", batch=16, embedder=fake)
    assert out.shape == (35, EMBED_DIM)
    assert len(fake.calls) == 3  # 16 + 16 + 3
    assert fake.calls[0][0] == [f"t{i}" for i in range(16)]
    assert fake.calls[2][0] == ["t32", "t33", "t34"]


def test_encode_batched_empty():
    out = encode_batched([], "passage: ", embedder=FakeEmbedder())
    assert out.shape == (0, EMBED_DIM)


def test_encode_output_shape():
    fake = FakeEmbedder()
    out = encode(["a", "b", "c"], "passage: ", embedder=fake)
    assert out.shape == (3, EMBED_DIM)
    assert out.dtype == np.float32
