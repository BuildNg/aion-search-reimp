import numpy as np
import pytest
import torch

from aion_reimp.text_embeddings import EmbeddingSpec, QwenEmbedder, _chunked


def test_chunked_splits_with_remainder() -> None:
    chunks = list(_chunked(list(range(7)), 3))
    assert chunks == [[0, 1, 2], [3, 4, 5], [6]]


def test_chunked_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(_chunked([1, 2, 3], 0))


class _FakeOutputs:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _FakeTokenizer:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        length = min(max(len(text) for text in texts), max_length)
        input_ids = torch.ones((len(texts), length), dtype=torch.long)
        attention_mask = torch.ones((len(texts), length), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeModel:
    """Deterministic stand-in: embedding value encodes call count and batch size."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.call_sizes = []

    def __call__(self, input_ids, attention_mask):
        self.call_sizes.append(input_ids.shape[0])
        batch, length = input_ids.shape
        hidden = torch.full((batch, length, 4), float(len(self.call_sizes)))
        return _FakeOutputs(hidden)


def _make_embedder(batch_size: int) -> QwenEmbedder:
    embedder = QwenEmbedder.__new__(QwenEmbedder)
    embedder.spec = EmbeddingSpec(
        model_id="fake",
        revision="fake",
        dimension=4,
        normalize=False,
        max_length=16,
        query_instruction="",
        batch_size=batch_size,
    )
    embedder.tokenizer = _FakeTokenizer()
    embedder.model = _FakeModel()
    return embedder


def test_encode_chunks_large_input_into_multiple_forward_passes() -> None:
    embedder = _make_embedder(batch_size=4)
    texts = [f"caption number {i}" for i in range(10)]
    embeddings = embedder.encode(texts, "document")

    assert embeddings.shape == (10, 4)
    # 10 texts at batch_size=4 -> three forward passes of sizes 4, 4, 2.
    assert embedder.model.call_sizes == [4, 4, 2]


def test_encode_concatenates_chunks_in_order() -> None:
    texts = [f"caption number {i}" for i in range(5)]
    chunked = _make_embedder(batch_size=2)

    result = chunked.encode(texts, "document")

    assert chunked.model.call_sizes == [2, 2, 1]
    # Each forward pass fills its rows with a distinct call-index value, so
    # row groups reveal whether concatenation preserved input order.
    np.testing.assert_array_equal(result[0:2], np.full((2, 4), 1.0))
    np.testing.assert_array_equal(result[2:4], np.full((2, 4), 2.0))
    np.testing.assert_array_equal(result[4:5], np.full((1, 4), 3.0))


def test_encode_returns_empty_array_for_no_texts() -> None:
    embedder = _make_embedder(batch_size=8)
    embeddings = embedder.encode([], "document")
    assert embeddings.shape == (0, 4)
    assert embedder.model.call_sizes == []
