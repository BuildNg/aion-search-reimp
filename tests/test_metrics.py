import numpy as np
import pytest

from aion_reimp.metrics import dcg_at_k, ndcg_at_k, recall_at_k, summary_statistics


def test_ndcg_is_one_for_ideal_fractional_ranking() -> None:
    relevance = [1.0, 0.7, 0.2, 0.0]
    assert ndcg_at_k(relevance, relevance, 4) == 1.0


def test_dcg_uses_exponential_gain() -> None:
    expected = (2.0**2.0 - 1.0) / np.log2(2.0)
    assert np.isclose(dcg_at_k([2.0], 1), expected)


def test_ndcg_zero_when_no_relevant_candidates() -> None:
    assert ndcg_at_k([0.0, 0.0], [0.0, 0.0], 2) == 0.0


def test_caption_to_image_recall() -> None:
    candidates = np.eye(3, dtype=np.float32)
    queries = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.2, 0.8]], dtype=np.float32)
    result = recall_at_k(queries, candidates, [0, 1], ks=(1, 2))
    assert result[1] == 0.5
    assert result[2] == 1.0


def test_summary_statistics_across_seeds() -> None:
    stats = summary_statistics([0.4, 0.5, 0.6])
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["std"] == pytest.approx(np.std([0.4, 0.5, 0.6]))
    assert stats["min"] == pytest.approx(0.4)
    assert stats["max"] == pytest.approx(0.6)
    assert stats["n"] == 3


def test_summary_statistics_rejects_empty() -> None:
    with pytest.raises(ValueError):
        summary_statistics([])
