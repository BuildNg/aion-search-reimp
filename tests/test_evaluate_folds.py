import numpy as np
import pytest

from aion_reimp.evaluate import assert_ten_folds, fold_ndcg_at_k


def test_assert_ten_folds_accepts_exactly_ten_distinct_values() -> None:
    assert_ten_folds(list(range(10)) * 3)


def test_assert_ten_folds_rejects_wrong_fold_count() -> None:
    with pytest.raises(ValueError, match="Expected exactly ten folds"):
        assert_ten_folds([0, 1, 2])


def test_fold_ndcg_matches_full_ndcg_when_query_is_confined_to_one_fold() -> None:
    object_ids = [f"obj{i}" for i in range(10)]
    candidates = np.eye(10, dtype=np.float32)
    relevance = [1.0 if i == 3 else 0.0 for i in range(10)]
    query_vector = candidates[3]
    folds = list(range(10))

    result = fold_ndcg_at_k(object_ids, candidates, relevance, folds, query_vector, k=1)

    assert result["by_fold"][3] == 1.0
    for fold_id, score in result["by_fold"].items():
        if fold_id != 3:
            assert score == 0.0
    assert result["mean"] == pytest.approx(0.1)


def test_fold_ndcg_requires_ten_folds() -> None:
    object_ids = ["a", "b"]
    candidates = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError, match="Expected exactly ten folds"):
        fold_ndcg_at_k(object_ids, candidates, [1.0, 0.0], [0, 1], candidates[0], k=1)
