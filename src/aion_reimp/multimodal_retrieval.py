"""Structured morphology-by-redshift retrieval on frozen Phase-6 embeddings."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .manifest import split_fraction
from .metrics import ndcg_at_k
from spec_probes.probes import (
    apply_scaler,
    fit_scaler,
    make_cv_folds,
    select_ridge_alpha,
)


MODEL_CONDITIONS = ("image_only", "spectrum_only", "image_plus_spectrum")
ABLATION_CONDITIONS = ("fusion_shuffled_image", "fusion_shuffled_spectrum")
REFERENCE_CONDITIONS = ("oracle", "no_information")
TASKS = ("morphology", "redshift", "joint")


def morphology_strength(frame: pd.DataFrame, label: str) -> np.ndarray:
    """Return a decision-tree-aware continuous Galaxy Zoo target in [0, 1]."""
    if label == "featured_or_disk":
        columns = ["smooth-or-featured_featured-or-disk_fraction"]
    elif label == "spiral":
        columns = [
            "smooth-or-featured_featured-or-disk_fraction",
            "disk-edge-on_no_fraction",
            "has-spiral-arms_yes_fraction",
        ]
    else:
        raise ValueError(f"Unsupported headline morphology: {label!r}")
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Morphology rows missing vote fractions: {sorted(missing)}")
    values = frame[columns].to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    return np.clip(values.min(axis=1), 0.0, 1.0)


def interval_score(values: Sequence[float], z_low: float, z_high: float) -> np.ndarray:
    """Score predicted redshift membership, flat inside and decaying outside."""
    if not np.isfinite([z_low, z_high]).all() or z_low >= z_high:
        raise ValueError("A redshift interval requires finite z_low < z_high")
    z = np.asarray(values, dtype=float)
    distance = np.maximum(np.maximum(float(z_low) - z, z - float(z_high)), 0.0)
    bandwidth = (float(z_high) - float(z_low)) / 2.0
    return np.exp(-distance / bandwidth)


def query_targets(
    manifest: pd.DataFrame,
    query: Mapping[str, Any],
    *,
    morphology_threshold: float,
) -> Dict[str, np.ndarray]:
    """Build locked continuous and binary targets for one structured query."""
    label = str(query["morphology"])
    reliable_column = f"reliable_{label}"
    if reliable_column not in manifest:
        raise ValueError(f"Joint manifest missing {reliable_column}")
    strength = morphology_strength(manifest, label)
    reliable = manifest[reliable_column].astype(bool).to_numpy()
    # The soft graded target and hard positive set must encode the same
    # Galaxy Zoo branch: min(path votes) >= t iff every path vote >= t.
    path_binary = strength >= float(morphology_threshold)
    if not np.array_equal(reliable, path_binary):
        raise ValueError(
            f"Graded and binary {label!r} targets use different Galaxy Zoo paths"
        )
    z = manifest["z"].to_numpy(dtype=float)
    in_redshift = (z >= float(query["z_low"])) & (z < float(query["z_high"]))
    return {
        "morphology_strength": strength,
        "morphology_binary": reliable,
        "redshift_binary": in_redshift,
        "joint_strength": strength * in_redshift.astype(float),
        "joint_binary": reliable & in_redshift,
    }


def assemble_embeddings(
    ordered_ids: Sequence[str],
    base_ids: Sequence[str],
    base_values: np.ndarray,
    new_ids: Sequence[str],
    new_values: np.ndarray,
) -> np.ndarray:
    """Assemble one ordered cache from disjoint base and extension rows."""
    base = np.asarray(base_values, dtype=np.float32)
    new = np.asarray(new_values, dtype=np.float32)
    if base.ndim != 2 or new.ndim != 2 or base.shape[1:] != new.shape[1:]:
        raise ValueError("Base and new embedding matrices must have the same width")
    base_ids = [str(value) for value in base_ids]
    new_ids = [str(value) for value in new_ids]
    if len(base_ids) != len(base) or len(new_ids) != len(new):
        raise ValueError("Embedding IDs and rows must align")
    if set(base_ids) & set(new_ids):
        raise ValueError("Base and new embedding IDs must be disjoint")
    row = {value: base[index] for index, value in enumerate(base_ids)}
    row.update({value: new[index] for index, value in enumerate(new_ids)})
    ordered = [str(value) for value in ordered_ids]
    missing = [value for value in ordered if value not in row]
    if missing:
        raise ValueError(f"Missing embeddings for object_id={missing[0]!r}")
    return np.stack([row[value] for value in ordered]).astype(np.float32, copy=False)


def _split(object_ids: Sequence[str], seed: int, train_ratio: float) -> pd.DataFrame:
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError("train_ratio must be between zero and one")
    ids = [str(value) for value in object_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("Retrieval object IDs must be unique")
    return pd.DataFrame(
        {
            "object_id": ids,
            "split_seed": int(seed),
            "split": [
                "train" if split_fraction(value, int(seed)) < float(train_ratio) else "test"
                for value in ids
            ],
        }
    )


def _fit_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha_grid: Sequence[float],
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> Tuple[Any, Ridge, float]:
    alpha = select_ridge_alpha(x_train, y_train, alpha_grid, folds, seed=seed)
    scaler = fit_scaler(x_train)
    model = Ridge(alpha=alpha, solver="svd", random_state=seed)
    model.fit(apply_scaler(scaler, x_train), np.asarray(y_train, dtype=float))
    return scaler, model, float(alpha)


def _predict(head: Tuple[Any, Ridge, float], values: np.ndarray) -> np.ndarray:
    scaler, model, _ = head
    return model.predict(apply_scaler(scaler, values))


def _derangement(size: int, seed: int) -> np.ndarray:
    if size < 2:
        raise ValueError("A shuffled-modality ablation needs at least two test objects")
    rng = np.random.default_rng(seed)
    identity = np.arange(size)
    for _ in range(32):
        order = rng.permutation(size)
        if not np.any(order == identity):
            return order
    return np.roll(identity, 1)


def _stable_noise(object_ids: Sequence[str], key: str) -> np.ndarray:
    values = []
    for object_id in object_ids:
        digest = hashlib.sha256(f"{key}|{object_id}".encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:8], "big") / float(2**64))
    return np.asarray(values, dtype=float)


def _rank_rows(
    manifest: pd.DataFrame,
    test_index: np.ndarray,
    *,
    query: Mapping[str, Any],
    task: str,
    condition: str,
    split_seed: int,
    score: np.ndarray,
    predicted_morphology: np.ndarray,
    predicted_z: np.ndarray,
    targets: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    if task == "morphology":
        graded = targets["morphology_strength"][test_index]
        binary = targets["morphology_binary"][test_index]
    elif task == "redshift":
        graded = targets["redshift_binary"][test_index].astype(float)
        binary = targets["redshift_binary"][test_index]
    elif task == "joint":
        graded = targets["joint_strength"][test_index]
        binary = targets["joint_binary"][test_index]
    else:
        raise ValueError(f"Unknown retrieval task: {task}")
    rows = manifest.iloc[test_index]
    ranked = pd.DataFrame(
        {
            "query_name": str(query["name"]),
            "query_text": str(query["text"]),
            "task": task,
            "condition": condition,
            "split_seed": int(split_seed),
            "object_id": rows["object_id"].astype(str).to_numpy(),
            "selection_reason": rows["selection_reason"].astype(str).to_numpy(),
            "score": np.asarray(score, dtype=float),
            "graded_relevance": graded,
            "binary_relevance": binary,
            "morphology_strength": targets["morphology_strength"][test_index],
            "reliable_morphology": targets["morphology_binary"][test_index],
            "z": rows["z"].to_numpy(dtype=float),
            "predicted_morphology": np.asarray(predicted_morphology, dtype=float),
            "predicted_z": np.asarray(predicted_z, dtype=float),
        }
    )
    return ranked.sort_values(["score", "object_id"], ascending=[False, True]).assign(
        rank=lambda frame: np.arange(1, len(frame) + 1)
    )


def _metrics_for_rows(rows: pd.DataFrame, query: Mapping[str, Any], k: int) -> Dict[str, Any]:
    ranked = rows.sort_values(["score", "object_id"], ascending=[False, True])
    top = ranked.head(k)
    positives = int(ranked["binary_relevance"].sum())
    hits = int(top["binary_relevance"].sum())
    z = top["z"].to_numpy(dtype=float)
    z_distance = np.maximum(
        np.maximum(float(query["z_low"]) - z, z - float(query["z_high"])), 0.0
    )
    return {
        "candidate_objects": int(len(ranked)),
        "relevant_objects": positives,
        "relevant_in_top_k": hits,
        "ndcg_at_k": ndcg_at_k(
            top["graded_relevance"].to_numpy(),
            ranked["graded_relevance"].to_numpy(),
            k,
        ),
        "recall_at_k": float(hits / positives) if positives else float("nan"),
        "recall_at_k_ceiling": (
            float(min(k, positives) / positives) if positives else float("nan")
        ),
        "precision_at_k": float(hits / min(k, len(ranked))) if len(ranked) else float("nan"),
        "top_k_morphology_strength": float(top["morphology_strength"].mean()),
        "top_k_redshift_distance": float(np.mean(z_distance)),
    }


def _readout(
    ranked_rows: pd.DataFrame,
    queries: Sequence[Mapping[str, Any]],
    *,
    k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query_by_name = {str(query["name"]): query for query in queries}
    metric_rows = []
    group_columns = ["query_name", "task", "condition", "split_seed"]
    for keys, group in ranked_rows.groupby(group_columns, sort=False):
        query_name, task, condition, split_seed = keys
        populations = {
            "enriched_all": group,
            "anchor": group.loc[group["selection_reason"].str.startswith("anchor_")],
            "morphology_priority": group.loc[group["selection_reason"].eq("morphology_priority")],
        }
        for population, rows in populations.items():
            if rows.empty:
                continue
            metric_rows.append(
                {
                    "query_name": query_name,
                    "task": task,
                    "condition": condition,
                    "split_seed": int(split_seed),
                    "candidate_population": population,
                    "k": int(k),
                    **_metrics_for_rows(rows, query_by_name[query_name], k),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    measure_columns = [
        "ndcg_at_k", "recall_at_k", "recall_at_k_ceiling", "precision_at_k",
        "top_k_morphology_strength", "top_k_redshift_distance",
    ]
    table_rows = []
    aggregate_keys = ["query_name", "task", "condition", "candidate_population", "k"]
    for keys, group in metrics.groupby(aggregate_keys, sort=False):
        row = dict(zip(aggregate_keys, keys))
        row["split_seeds"] = int(group["split_seed"].nunique())
        row["relevant_objects_min"] = int(group["relevant_objects"].min())
        for column in measure_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_split_std"] = float(group[column].std(ddof=0))
        table_rows.append(row)
    tables = pd.DataFrame(table_rows)

    comparison_rows = []
    primary = metrics.loc[
        metrics["task"].eq("joint") & metrics["candidate_population"].eq("enriched_all")
    ]
    for baseline in ("image_only", "spectrum_only", *ABLATION_CONDITIONS):
        left = primary.loc[primary["condition"].eq("image_plus_spectrum")]
        right = primary.loc[primary["condition"].eq(baseline)]
        joined = left.merge(right, on=["query_name", "task", "split_seed", "candidate_population", "k"], suffixes=("_fusion", "_baseline"))
        for row in joined.itertuples(index=False):
            comparison_rows.append(
                {
                    "query_name": row.query_name,
                    "split_seed": int(row.split_seed),
                    "comparison": f"image_plus_spectrum_minus_{baseline}",
                    "ndcg_at_k_difference": float(row.ndcg_at_k_fusion - row.ndcg_at_k_baseline),
                    "recall_at_k_difference": float(row.recall_at_k_fusion - row.recall_at_k_baseline),
                }
            )
    return metrics, tables, pd.DataFrame(comparison_rows)


def run_joint_retrieval(
    manifest: pd.DataFrame,
    image_embeddings: np.ndarray,
    spectrum_embeddings: np.ndarray,
    queries: Sequence[Mapping[str, Any]],
    *,
    split_seeds: Sequence[int],
    train_ratio: float,
    cv_folds: int,
    alpha_grid: Sequence[float],
    seed: int,
    k: int,
    morphology_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit matched ridge heads and return row-level rankings and readouts."""
    required = {"object_id", "selection_reason", "z"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Joint manifest missing columns: {sorted(missing)}")
    if not queries:
        raise ValueError("At least one retrieval query is required")
    image = np.asarray(image_embeddings, dtype=np.float32)
    spectrum = np.asarray(spectrum_embeddings, dtype=np.float32)
    if image.ndim != 2 or spectrum.ndim != 2 or image.shape[0] != len(manifest) or spectrum.shape[0] != len(manifest):
        raise ValueError("Image and spectrum embeddings must align with the joint manifest")
    fusion = np.concatenate([image, spectrum], axis=1)
    arrays = {"image_only": image, "spectrum_only": spectrum, "image_plus_spectrum": fusion}
    targets_by_query = {
        str(query["name"]): query_targets(
            manifest, query, morphology_threshold=morphology_threshold
        )
        for query in queries
    }
    for query in queries:
        expected = int(query["expected_positive_objects"])
        observed = int(targets_by_query[str(query["name"])]["joint_binary"].sum())
        if observed != expected:
            raise ValueError(f"Query {query['name']!r} expected {expected} positives, found {observed}")

    ranked_frames = []
    split_frames = []
    selection_rows = []
    object_ids = manifest["object_id"].astype(str).tolist()
    row_by_id = {value: index for index, value in enumerate(object_ids)}
    morphology_labels = sorted({str(query["morphology"]) for query in queries})

    for split_seed in split_seeds:
        split = _split(object_ids, int(split_seed), float(train_ratio))
        split_frames.append(split)
        train_ids = split.loc[split["split"].eq("train"), "object_id"]
        test_ids = split.loc[split["split"].eq("test"), "object_id"]
        train_index = np.asarray([row_by_id[value] for value in train_ids], dtype=int)
        test_index = np.asarray([row_by_id[value] for value in test_ids], dtype=int)
        folds = make_cv_folds(len(train_index), int(cv_folds), seed=int(seed))
        z_train = manifest.iloc[train_index]["z"].to_numpy(dtype=float)
        predictions: Dict[str, Dict[str, Any]] = {}
        heads: Dict[str, Dict[str, Any]] = {}

        for condition in MODEL_CONDITIONS:
            x_train = arrays[condition][train_index]
            x_test = arrays[condition][test_index]
            z_head = _fit_head(x_train, z_train, alpha_grid, folds, int(seed))
            morph_heads = {}
            morph_predictions = {}
            for label in morphology_labels:
                target = morphology_strength(manifest, label)[train_index]
                head = _fit_head(x_train, target, alpha_grid, folds, int(seed))
                morph_heads[label] = head
                morph_predictions[label] = np.clip(_predict(head, x_test), 0.0, 1.0)
                selection_rows.append(
                    {
                        "condition": condition,
                        "target": label,
                        "split_seed": int(split_seed),
                        "ridge_alpha": head[2],
                        "at_grid_max": head[2] == max(alpha_grid),
                    }
                )
            heads[condition] = {"z": z_head, "morphology": morph_heads}
            predictions[condition] = {
                "z": _predict(z_head, x_test),
                "morphology": morph_predictions,
            }
            selection_rows.append(
                {
                    "condition": condition,
                    "target": "spec_z",
                    "split_seed": int(split_seed),
                    "ridge_alpha": z_head[2],
                    "at_grid_max": z_head[2] == max(alpha_grid),
                }
            )

        permutation = _derangement(len(test_index), int(split_seed) + int(seed))
        shuffled = {
            "fusion_shuffled_image": np.concatenate([image[test_index][permutation], spectrum[test_index]], axis=1),
            "fusion_shuffled_spectrum": np.concatenate([image[test_index], spectrum[test_index][permutation]], axis=1),
        }
        fusion_heads = heads["image_plus_spectrum"]
        for condition, x_test in shuffled.items():
            predictions[condition] = {
                "z": _predict(fusion_heads["z"], x_test),
                "morphology": {
                    label: np.clip(_predict(fusion_heads["morphology"][label], x_test), 0.0, 1.0)
                    for label in morphology_labels
                },
            }

        test_object_ids = manifest.iloc[test_index]["object_id"].astype(str).tolist()
        for query in queries:
            name = str(query["name"])
            label = str(query["morphology"])
            targets = targets_by_query[name]
            true_morphology = targets["morphology_strength"][test_index]
            true_z = manifest.iloc[test_index]["z"].to_numpy(dtype=float)
            for task in TASKS:
                for condition in (*MODEL_CONDITIONS, *ABLATION_CONDITIONS, *REFERENCE_CONDITIONS):
                    if condition == "oracle":
                        predicted_morphology, predicted_z = true_morphology, true_z
                    elif condition == "no_information":
                        predicted_morphology = np.zeros(len(test_index), dtype=float)
                        predicted_z = np.full(len(test_index), float(np.median(z_train)))
                    else:
                        predicted_morphology = predictions[condition]["morphology"][label]
                        predicted_z = predictions[condition]["z"]
                    morph_score = np.clip(predicted_morphology, 0.0, 1.0)
                    z_score = interval_score(predicted_z, float(query["z_low"]), float(query["z_high"]))
                    if condition == "oracle":
                        score = (
                            true_morphology if task == "morphology" else
                            targets["redshift_binary"][test_index].astype(float) if task == "redshift" else
                            targets["joint_strength"][test_index]
                        )
                    elif condition == "no_information":
                        score = _stable_noise(test_object_ids, f"{seed}|{split_seed}|{name}|{task}")
                    else:
                        score = morph_score if task == "morphology" else z_score if task == "redshift" else morph_score * z_score
                    ranked_frames.append(
                        _rank_rows(
                            manifest, test_index, query=query, task=task, condition=condition,
                            split_seed=int(split_seed), score=score,
                            predicted_morphology=predicted_morphology, predicted_z=predicted_z,
                            targets=targets,
                        )
                    )

    ranked = pd.concat(ranked_frames, ignore_index=True)
    metrics, tables, comparisons = _readout(ranked, queries, k=int(k))
    return (
        ranked,
        metrics,
        tables,
        comparisons,
        pd.DataFrame(selection_rows),
        pd.concat(split_frames, ignore_index=True),
    )


def run_cached_distance_retrieval(
    predictions: pd.DataFrame,
    queries: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank cached held-out redshift predictions without refitting a model."""
    required = {"object_id", "encoder", "split_seed", "y_true_numeric", "y_pred_numeric"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Cached predictions missing columns: {sorted(missing)}")
    frames = []
    for split_seed in sorted(predictions["split_seed"].unique()):
        split_rows = predictions.loc[predictions["split_seed"].eq(split_seed)]
        for query in queries:
            for condition in sorted(split_rows["encoder"].unique()):
                rows = split_rows.loc[split_rows["encoder"].eq(condition)].sort_values("object_id")
                z = rows["y_true_numeric"].to_numpy(dtype=float)
                predicted = rows["y_pred_numeric"].to_numpy(dtype=float)
                relevant = (z >= float(query["z_low"])) & (z < float(query["z_high"]))
                score = interval_score(predicted, float(query["z_low"]), float(query["z_high"]))
                frame = pd.DataFrame(
                    {
                        "query_name": str(query["name"]),
                        "query_text": str(query["text"]),
                        "task": "redshift",
                        "condition": condition,
                        "split_seed": int(split_seed),
                        "object_id": rows["object_id"].astype(str).to_numpy(),
                        "selection_reason": "anchor_phase6_hsc_crossmatch_18k_v1",
                        "score": score,
                        "graded_relevance": relevant.astype(float),
                        "binary_relevance": relevant,
                        "morphology_strength": np.nan,
                        "reliable_morphology": False,
                        "z": z,
                        "predicted_morphology": np.nan,
                        "predicted_z": predicted,
                    }
                )
                frames.append(frame.sort_values(["score", "object_id"], ascending=[False, True]).assign(rank=lambda value: np.arange(1, len(value) + 1)))
            base = split_rows.drop_duplicates("object_id").sort_values("object_id")
            z = base["y_true_numeric"].to_numpy(dtype=float)
            ids = base["object_id"].astype(str).tolist()
            relevant = (z >= float(query["z_low"])) & (z < float(query["z_high"]))
            for condition, score in (
                ("oracle", relevant.astype(float)),
                ("no_information", _stable_noise(ids, f"distance|{seed}|{split_seed}|{query['name']}")),
            ):
                frame = pd.DataFrame(
                    {
                        "query_name": str(query["name"]), "query_text": str(query["text"]),
                        "task": "redshift", "condition": condition, "split_seed": int(split_seed),
                        "object_id": ids, "selection_reason": "anchor_phase6_hsc_crossmatch_18k_v1",
                        "score": score, "graded_relevance": relevant.astype(float),
                        "binary_relevance": relevant, "morphology_strength": np.nan,
                        "reliable_morphology": False, "z": z, "predicted_morphology": np.nan,
                        "predicted_z": z if condition == "oracle" else np.nan,
                    }
                )
                frames.append(frame.sort_values(["score", "object_id"], ascending=[False, True]).assign(rank=lambda value: np.arange(1, len(value) + 1)))
    ranked = pd.concat(frames, ignore_index=True)
    metrics, tables, _ = _readout(ranked, queries, k=int(k))
    return ranked, metrics, tables
