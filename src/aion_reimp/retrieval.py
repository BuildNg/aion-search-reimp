"""Condition-independent candidate ranking and row-level evidence."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import ndcg_at_k


def rank_candidates(
    candidate_ids: Sequence[str],
    candidate_embeddings: np.ndarray,
    query_name: str,
    query_text: str,
    query_embedding: np.ndarray,
    relevance: Sequence[float],
) -> pd.DataFrame:
    candidates = np.asarray(candidate_embeddings, dtype=np.float32)
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if candidates.shape[0] != len(candidate_ids) or candidates.shape[0] != len(relevance):
        raise ValueError("Candidate IDs, embeddings, and relevance must have equal rows")
    if candidates.shape[1] != query.size:
        raise ValueError("Candidate and query dimensions differ")
    scores = candidates @ query
    order = np.argsort(-scores, kind="stable")
    frame = pd.DataFrame(
        {
            "query_name": query_name,
            "query_text": query_text,
            "rank": np.arange(1, len(order) + 1),
            "object_id": np.asarray(candidate_ids, dtype=str)[order],
            "score": scores[order],
            "relevance": np.asarray(relevance, dtype=np.float32)[order],
        }
    )
    return frame


def score_ranked_rows(rows: pd.DataFrame, k: int = 10) -> float:
    return ndcg_at_k(rows["relevance"].to_numpy(), rows["relevance"].to_numpy(), k)
