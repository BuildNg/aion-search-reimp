"""Paired image/spectrum redshift comparison on one frozen object set."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .probes import apply_scaler, fit_scaler, make_cv_folds, ridge_probe, select_ridge_alpha
from .run_probes import PREDICTIONS_COLUMNS, run_baseline_suite
from .spectra_data import object_level_split


PAIRED_MANIFEST_COLUMNS = (
    "object_id", "image_object_id", "spectrum_object_id", "source_row_id",
    "spectrum_ra", "spectrum_dec", "z", "zerr", "separation_arcsec",
)


def build_paired_manifest(selected_matches: pd.DataFrame) -> pd.DataFrame:
    """Reduce locked one-arcsecond matches to the paired experiment schema."""
    source_columns = {
        "source_object_id", "desi_object_id", "source_row_id",
        "desi_ra", "desi_dec", "desi_z", "desi_zerr", "separation_arcsec",
    }
    missing = source_columns - set(selected_matches)
    if missing:
        raise ValueError(f"Selected matches missing columns: {sorted(missing)}")
    manifest = selected_matches.loc[:, sorted(source_columns)].rename(
        columns={
            "source_object_id": "image_object_id",
            "desi_object_id": "spectrum_object_id",
            "desi_ra": "spectrum_ra",
            "desi_dec": "spectrum_dec",
            "desi_z": "z",
            "desi_zerr": "zerr",
        }
    )
    manifest.insert(0, "object_id", manifest["image_object_id"].astype(str))
    manifest["image_object_id"] = manifest["image_object_id"].astype(str)
    manifest["spectrum_object_id"] = manifest["spectrum_object_id"].astype(str)
    for column in ("spectrum_ra", "spectrum_dec", "z", "zerr", "separation_arcsec"):
        manifest[column] = pd.to_numeric(manifest[column], errors="raise")
    if manifest["object_id"].duplicated().any() or manifest["spectrum_object_id"].duplicated().any():
        raise ValueError("Paired manifest requires one unique image and one unique spectrum per row")
    if not np.isfinite(
        manifest[["spectrum_ra", "spectrum_dec", "z", "zerr", "separation_arcsec"]].to_numpy()
    ).all():
        raise ValueError("Paired manifest contains non-finite labels or separations")
    return manifest.loc[:, list(PAIRED_MANIFEST_COLUMNS)].sort_values("object_id").reset_index(drop=True)


def run_paired_redshift_comparison(
    manifest: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
    revisions: Mapping[str, str],
    *,
    split_seeds: Sequence[int],
    train_ratio: float,
    cv_folds: int,
    ridge_alpha_grid: Sequence[float],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the same standardized ridge head to every modality condition."""
    missing = set(PAIRED_MANIFEST_COLUMNS) - set(manifest)
    if missing:
        raise ValueError(f"Paired manifest missing columns: {sorted(missing)}")
    if set(embeddings) != set(revisions):
        raise ValueError("Every embedding condition needs one revision label")
    n_rows = len(manifest)
    arrays = {name: np.asarray(values, dtype=np.float32) for name, values in embeddings.items()}
    for name, values in arrays.items():
        if values.ndim != 2 or values.shape[0] != n_rows:
            raise ValueError(f"{name} embeddings must have shape (len(manifest), dimension)")

    object_ids = manifest["object_id"].astype(str).to_numpy()
    z = manifest["z"].to_numpy(dtype=np.float64)
    row_by_id = {object_id: index for index, object_id in enumerate(object_ids)}
    prediction_frames: List[pd.DataFrame] = []
    split_frames: List[pd.DataFrame] = []

    for split_seed in split_seeds:
        split = object_level_split(object_ids, int(split_seed), float(train_ratio))
        split.insert(0, "split_seed", int(split_seed))
        split_frames.append(split)
        train_ids = split.loc[split["split"].eq("train"), "object_id"].tolist()
        test_ids = split.loc[split["split"].eq("test"), "object_id"].tolist()
        train_index = np.asarray([row_by_id[value] for value in train_ids], dtype=np.int64)
        test_index = np.asarray([row_by_id[value] for value in test_ids], dtype=np.int64)
        z_train, z_test = z[train_index], z[test_index]
        folds = make_cv_folds(len(train_index), int(cv_folds), seed=int(seed))
        prediction_frames.append(run_baseline_suite(test_ids, z_train, z_test, int(split_seed)))

        for condition, values in arrays.items():
            x_train, x_test = values[train_index], values[test_index]
            alpha = select_ridge_alpha(x_train, z_train, ridge_alpha_grid, folds, seed=int(seed))
            scaler = fit_scaler(x_train)
            predicted = ridge_probe(
                apply_scaler(scaler, x_train), z_train,
                apply_scaler(scaler, x_test), alpha, seed=int(seed),
            )
            rows = [
                {
                    "object_id": object_id,
                    "encoder": condition,
                    "encoder_revision": revisions[condition],
                    "target": "spec_z",
                    "probe_family": "linear",
                    "split": "test",
                    "split_seed": int(split_seed),
                    "y_true_numeric": float(true_value),
                    "y_pred_numeric": float(predicted_value),
                    "y_true_label": "",
                    "y_pred_label": "",
                    "hyperparameter_name": "ridge_alpha",
                    "hyperparameter_value": float(alpha),
                }
                for object_id, true_value, predicted_value in zip(test_ids, z_test, predicted)
            ]
            prediction_frames.append(pd.DataFrame(rows, columns=PREDICTIONS_COLUMNS))

    return pd.concat(prediction_frames, ignore_index=True), pd.concat(split_frames, ignore_index=True)


BOOTSTRAP_SCALES = ("one_plus_z", "absolute")


def paired_error_bootstrap(
    predictions: pd.DataFrame,
    baseline_condition: str,
    comparison_condition: str,
    *,
    scale: str,
    n_resamples: int,
    seed: int,
) -> Dict[str, object]:
    """Paired object-cluster CI for mean-|residual| improvement between conditions.

    ``scale`` selects the residual convention: ``"one_plus_z"`` uses the
    photo-z standard |z_pred - z_true| / (1 + z_true), matching the headline
    NMAD and catastrophic-outlier metrics; ``"absolute"`` uses raw |Delta z|.
    An object can appear in more than one seeded test set. Its paired
    residual difference is averaged first, then unique objects are
    resampled, so overlapping test sets are never treated as independent.
    Positive improvement means the comparison condition has lower error.
    """
    if scale not in BOOTSTRAP_SCALES:
        raise ValueError(f"scale must be one of {BOOTSTRAP_SCALES}, got {scale!r}")
    required = {"object_id", "encoder", "split_seed", "y_true_numeric", "y_pred_numeric"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Predictions missing bootstrap columns: {sorted(missing)}")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    selected = predictions.loc[
        predictions["encoder"].isin([baseline_condition, comparison_condition]),
        list(required),
    ].copy()
    y_true = selected["y_true_numeric"].to_numpy(dtype=float)
    error = np.abs(y_true - selected["y_pred_numeric"].to_numpy(dtype=float))
    if scale == "one_plus_z":
        error = error / (1.0 + y_true)
    selected["scaled_error"] = error
    paired = selected.pivot(
        index=["object_id", "split_seed"], columns="encoder", values="scaled_error"
    )
    for condition in (baseline_condition, comparison_condition):
        if condition not in paired:
            raise ValueError(f"No paired predictions for condition {condition!r}")
    if paired[[baseline_condition, comparison_condition]].isna().to_numpy().any():
        raise ValueError(
            "Paired predictions are incomplete: some (object, split_seed) test rows "
            "are missing one of the two compared conditions"
        )
    per_object = (
        paired[baseline_condition] - paired[comparison_condition]
    ).groupby(level="object_id").mean()
    if per_object.empty:
        raise ValueError("No paired objects available for bootstrap")
    values = per_object.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, 1000):
        count = min(1000, n_resamples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        bootstrap_means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    decision = (
        "comparison_better"
        if low > 0.0
        else "comparison_worse"
        if high < 0.0
        else "no_clear_difference"
    )
    residual = "|z_pred - z_true| / (1 + z_true)" if scale == "one_plus_z" else "|z_pred - z_true|"
    return {
        "baseline_condition": baseline_condition,
        "comparison_condition": comparison_condition,
        "scale": scale,
        "estimand": f"mean(baseline {residual} - comparison {residual}), averaged per object across test appearances",
        "positive_favors": comparison_condition,
        "n_unique_objects": int(len(values)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "error_improvement": float(values.mean()),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "decision": decision,
    }
