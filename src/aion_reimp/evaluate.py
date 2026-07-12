"""Frozen benchmark evaluation from row-level projected embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .model import AIONSearchModel
from .retrieval import rank_candidates, score_ranked_rows


def project_query(model: AIONSearchModel, raw_embedding: Sequence[float]) -> np.ndarray:
    tensor = torch.tensor(raw_embedding, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        projected = model.text_projector(tensor)
    return projected.squeeze(0).cpu().numpy()


def evaluate_one_query(
    object_ids: Sequence[str],
    candidate_embeddings: np.ndarray,
    relevance: Sequence[float],
    query_vector: np.ndarray,
    query_name: str,
    query_text: str,
    k: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = rank_candidates(
        object_ids,
        candidate_embeddings,
        query_name,
        query_text,
        query_vector,
        relevance,
    )
    metrics = {f"ndcg@{k}": score_ranked_rows(rows, k)}
    return rows, metrics


def load_hf_frame(repo_id: str, revision: str, columns: Sequence[str]) -> pd.DataFrame:
    from datasets import load_dataset

    dataset = load_dataset(repo_id, revision=revision, split="train")
    missing = set(columns) - set(dataset.column_names)
    if missing:
        raise ValueError(f"{repo_id} missing evaluation columns: {sorted(missing)}")
    return dataset.select_columns(list(columns)).to_pandas()


def write_evaluation(
    ranked_frames: Iterable[pd.DataFrame],
    metrics: Mapping[str, Mapping[str, float]],
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(list(ranked_frames), ignore_index=True).to_parquet(
        output_dir / "ranked_rows.parquet", index=False
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )


def check_reference_gate(
    metrics: Mapping[str, Mapping[str, float]],
    ranked_rows: pd.DataFrame,
    reference_gate: Mapping[str, object],
) -> Tuple[Dict[str, object], Sequence[str]]:
    targets = reference_gate["published_rounded_targets"]
    failures = []
    checks: Dict[str, object] = {}
    for category in ("spiral", "merger", "lens"):
        observed = metrics[category]["ndcg@10"]
        expected = float(targets[category])
        passed = round(observed, 3) == expected
        checks[category] = {
            "expected_paper_rounded": expected,
            "observed": observed,
            "observed_rounded": round(observed, 3),
            "passed": passed,
        }
        if not passed:
            failures.append(
                f"{category} nDCG@10 expected {expected:.3f}, observed {observed:.6f}"
            )

    lens_top10 = ranked_rows[
        (ranked_rows["query_name"] == "lens") & (ranked_rows["rank"] <= 10)
    ]
    confirmed_lenses = int((lens_top10["relevance"] > 0).sum())
    expected_lenses = int(reference_gate["lens_top10_positives"])
    checks["lens_top10_confirmed"] = {
        "expected": expected_lenses,
        "observed": confirmed_lenses,
        "passed": confirmed_lenses == expected_lenses,
    }
    if confirmed_lenses != expected_lenses:
        failures.append(
            f"lens top-10 expected {expected_lenses} confirmed positives, got {confirmed_lenses}"
        )
    return checks, failures


def evaluate_released_benchmarks(
    model: AIONSearchModel,
    query_cache: pd.DataFrame,
    benchmark_specs: Sequence[Mapping[str, str]],
    output_dir: Path,
    reference_gate: Optional[Mapping[str, object]] = None,
    canonical_only: bool = True,
) -> None:
    query_rows = query_cache.copy()
    if canonical_only:
        query_rows = query_rows[query_rows["variant"] == "canonical"]
    query_by_category = {row.category: row for row in query_rows.itertuples(index=False)}
    ranked_frames = []
    metrics: Dict[str, Dict[str, float]] = {}
    for benchmark in benchmark_specs:
        name = benchmark["name"]
        if name == "gz_decals":
            frame = load_hf_frame(
                benchmark["repo_id"],
                benchmark["revision"],
                [
                    "aion_search_embedding",
                    "ra",
                    "dec",
                    "has-spiral-arms_yes_fraction",
                    "merging_merger_fraction",
                ],
            )
            candidates = np.asarray(frame["aion_search_embedding"].tolist(), dtype=np.float32)
            object_ids = [f"{ra:.8f},{dec:.8f}" for ra, dec in zip(frame["ra"], frame["dec"])]
            tasks = {
                "spiral": frame["has-spiral-arms_yes_fraction"].to_numpy(dtype=np.float32),
                "merger": frame["merging_merger_fraction"].to_numpy(dtype=np.float32),
            }
        elif name == "lens":
            frame = load_hf_frame(
                benchmark["repo_id"],
                benchmark["revision"],
                ["aion_search_embedding", "ra", "dec", "is_lens"],
            )
            candidates = np.asarray(frame["aion_search_embedding"].tolist(), dtype=np.float32)
            object_ids = [f"{ra:.8f},{dec:.8f}" for ra, dec in zip(frame["ra"], frame["dec"])]
            tasks = {"lens": frame["is_lens"].to_numpy(dtype=np.float32)}
        else:
            raise ValueError(f"Unsupported benchmark: {name}")

        for category, relevance in tasks.items():
            query_row = query_by_category[category]
            projected = project_query(model, query_row.embedding)
            rows, query_metrics = evaluate_one_query(
                object_ids,
                candidates,
                relevance,
                projected,
                category,
                query_row.text,
            )
            ranked_frames.append(rows)
            metrics[category] = query_metrics
    write_evaluation(ranked_frames, metrics, output_dir)
    if reference_gate is not None:
        all_rows = pd.concat(ranked_frames, ignore_index=True)
        checks, failures = check_reference_gate(metrics, all_rows, reference_gate)
        (Path(output_dir) / "reference_gate.json").write_text(
            json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8"
        )
        if failures:
            raise AssertionError("Reference gate failed: " + "; ".join(failures))
