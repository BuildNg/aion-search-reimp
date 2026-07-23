import numpy as np
import pandas as pd
import pytest

from aion_reimp.multimodal_retrieval import (
    ABLATION_CONDITIONS,
    MODEL_CONDITIONS,
    REFERENCE_CONDITIONS,
    assemble_embeddings,
    interval_score,
    morphology_strength,
    query_targets,
    run_cached_distance_retrieval,
    run_joint_retrieval,
    validate_cached_distance_predictions,
)


def _manifest(n=90):
    z = np.linspace(0.01, 0.30, n)
    featured = np.where(np.arange(n) % 3 == 0, 0.9, 0.15)
    spiral = np.where(np.arange(n) % 6 == 0, 0.85, 0.1)
    return pd.DataFrame(
        {
            "object_id": [f"object-{index:03d}" for index in range(n)],
            "selection_reason": ["anchor_test"] * (n - 12) + ["morphology_priority"] * 12,
            "z": z,
            "smooth-or-featured_featured-or-disk_fraction": featured,
            "disk-edge-on_no_fraction": np.where(featured > 0.7, 0.9, 0.2),
            "has-spiral-arms_yes_fraction": spiral,
            "reliable_featured_or_disk": featured >= 0.7,
            "reliable_spiral": (featured >= 0.7) & (spiral >= 0.7),
        }
    )


def test_query_targets_use_path_minimum_and_locked_interval():
    frame = _manifest(12)
    featured = morphology_strength(frame, "featured_or_disk")
    spiral = morphology_strength(frame, "spiral")
    assert featured[0] == pytest.approx(0.9)
    assert spiral[0] == pytest.approx(0.85)
    assert spiral[3] == pytest.approx(0.1)
    query = {
        "name": "spiral",
        "text": "spirals",
        "morphology": "spiral",
        "z_low": 0.05,
        "z_high": 0.20,
    }
    targets = query_targets(frame, query, morphology_threshold=0.7)
    assert np.all(targets["joint_binary"] <= targets["morphology_binary"])
    np.testing.assert_array_equal(
        targets["morphology_binary"], morphology_strength(frame, "spiral") >= 0.7
    )
    mismatched = frame.copy()
    mismatched.loc[1, "reliable_spiral"] = True
    with pytest.raises(ValueError, match="different Galaxy Zoo paths"):
        query_targets(mismatched, query, morphology_threshold=0.7)
    assert interval_score([0.10, 0.20, 0.25], 0.10, 0.20).tolist() == pytest.approx(
        [1.0, 1.0, np.exp(-1.0)]
    )


def test_assemble_embeddings_reuses_base_and_extension_in_requested_order():
    values = assemble_embeddings(
        ["new", "base"],
        ["base"],
        np.array([[1.0, 2.0]], dtype=np.float32),
        ["new"],
        np.array([[3.0, 4.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(values, [[3.0, 4.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="disjoint"):
        assemble_embeddings(["same"], ["same"], np.ones((1, 2)), ["same"], np.ones((1, 2)))


def test_joint_retrieval_uses_identical_test_objects_and_writes_controls():
    manifest = _manifest()
    rng = np.random.default_rng(4)
    z = manifest["z"].to_numpy()
    morphology = manifest["smooth-or-featured_featured-or-disk_fraction"].to_numpy()
    image = np.column_stack([morphology, morphology**2, rng.normal(size=len(manifest))]).astype(np.float32)
    spectrum = np.column_stack([z, z**2, rng.normal(size=len(manifest))]).astype(np.float32)
    query = {
        "name": "featured",
        "text": "Find featured galaxies between redshift 0.05 and 0.15.",
        "morphology": "featured_or_disk",
        "z_low": 0.05,
        "z_high": 0.15,
    }
    query["expected_positive_objects"] = int(
        query_targets(manifest, query, morphology_threshold=0.7)["joint_binary"].sum()
    )
    ranked, metrics, tables, comparisons, heads, splits = run_joint_retrieval(
        manifest,
        image,
        spectrum,
        [query],
        split_seeds=[11],
        train_ratio=0.75,
        cv_folds=2,
        alpha_grid=[0.1, 1.0],
        seed=7,
        k=10,
        morphology_threshold=0.7,
    )
    assert set(ranked["condition"]) == set(MODEL_CONDITIONS + ABLATION_CONDITIONS + REFERENCE_CONDITIONS)
    assert set(ranked["task"]) == {"morphology", "redshift", "joint"}
    for _, rows in ranked.groupby(["query_name", "task", "split_seed"]):
        object_sets = [set(group["object_id"]) for _, group in rows.groupby("condition")]
        assert all(values == object_sets[0] for values in object_sets[1:])
    oracle = metrics.loc[
        metrics["condition"].eq("oracle")
        & metrics["task"].eq("joint")
        & metrics["candidate_population"].eq("enriched_all")
    ]
    assert oracle.iloc[0]["ndcg_at_k"] == pytest.approx(1.0)
    assert np.all(oracle["recall_at_k"] <= oracle["recall_at_k_ceiling"])
    assert set(metrics["candidate_population"]) == {"enriched_all", "anchor", "morphology_priority"}
    assert not tables.empty and not comparisons.empty and not heads.empty
    assert splits["object_id"].nunique() == len(manifest)


def test_cached_distance_retrieval_ranks_every_condition_on_same_objects():
    rows = []
    z = np.linspace(0.01, 0.3, 20)
    for split_seed in (1, 2):
        for condition, offset in (
            ("trivial_baseline", 0.08),
            ("image_only", 0.03),
            ("spectrum_only", 0.01),
            ("image_plus_spectrum", 0.015),
        ):
            for index, value in enumerate(z):
                rows.append(
                    {
                        "object_id": f"object-{index}",
                        "encoder": condition,
                        "split_seed": split_seed,
                        "y_true_numeric": value,
                        "y_pred_numeric": value + offset,
                    }
                )
    query = {
        "name": "z-window",
        "text": "Find galaxies between redshift 0.05 and 0.15.",
        "morphology": "featured_or_disk",
        "z_low": 0.05,
        "z_high": 0.15,
    }
    ranked, metrics, tables = run_cached_distance_retrieval(
        pd.DataFrame(rows), [query], seed=5, k=10
    )
    assert {"oracle", "no_information", "image_only", "spectrum_only"} <= set(ranked["condition"])
    assert set(metrics["split_seed"]) == {1, 2}
    assert not tables.empty


def test_cached_distance_validation_uses_held_out_rows_not_every_base_object():
    base_ids = [f"object-{index}" for index in range(8)]
    assignments = pd.DataFrame(
        [
            {"object_id": object_id, "split_seed": seed, "split": "test" if index in test else "train"}
            for seed, test in ((1, {0, 1}), (2, {2, 3, 4}))
            for index, object_id in enumerate(base_ids)
        ]
    )
    predictions = pd.DataFrame(
        [
            {"object_id": object_id, "split_seed": seed, "encoder": encoder}
            for seed, test in ((1, [0, 1]), (2, [2, 3, 4]))
            for encoder in ("image_only", "spectrum_only")
            for object_id in [base_ids[index] for index in test]
        ]
    )
    validate_cached_distance_predictions(predictions, assignments, base_ids, [1, 2])
    with pytest.raises(ValueError, match="held-out assignment"):
        validate_cached_distance_predictions(
            predictions.iloc[:-1], assignments, base_ids, [1, 2]
        )
