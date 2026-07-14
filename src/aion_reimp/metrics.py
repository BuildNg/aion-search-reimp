"""Single implementations of retrieval and caption-audit metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


def dcg_at_k(ranked_relevance: Sequence[float], k: int) -> float:
    relevance = np.asarray(ranked_relevance, dtype=np.float64)[:k]
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2, dtype=np.float64))
    gains = np.power(2.0, relevance) - 1.0
    return float(np.sum(gains / discounts))


def ndcg_at_k(
    ranked_relevance: Sequence[float],
    all_relevance: Sequence[float],
    k: int,
) -> float:
    actual = dcg_at_k(ranked_relevance, k)
    ideal_ranked = np.sort(np.asarray(all_relevance, dtype=np.float64))[::-1]
    ideal = dcg_at_k(ideal_ranked, k)
    return 0.0 if ideal == 0.0 else float(actual / ideal)


def recall_at_k(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    target_indices: Sequence[int],
    ks: Iterable[int] = (1, 10, 100),
) -> Dict[int, float]:
    queries = np.asarray(query_embeddings, dtype=np.float32)
    candidates = np.asarray(candidate_embeddings, dtype=np.float32)
    targets = np.asarray(target_indices, dtype=np.int64)
    if queries.ndim != 2 or candidates.ndim != 2:
        raise ValueError("query_embeddings and candidate_embeddings must be matrices")
    if queries.shape[0] != targets.shape[0]:
        raise ValueError("One target index is required per query")
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError("Query and candidate dimensions differ")
    if np.any(targets < 0) or np.any(targets >= candidates.shape[0]):
        raise ValueError("target_indices contains an invalid candidate index")

    similarities = queries @ candidates.T
    order = np.argsort(-similarities, axis=1, kind="stable")
    result: Dict[int, float] = {}
    for k in ks:
        if k <= 0:
            raise ValueError("Recall k must be positive")
        top = order[:, : min(k, candidates.shape[0])]
        result[int(k)] = float(np.mean(np.any(top == targets[:, None], axis=1)))
    return result


def summary_statistics(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("summary_statistics requires at least one value")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "n": int(array.size),
    }


def bootstrap_mean_interval(
    values: Sequence[float],
    samples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[draw_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))
