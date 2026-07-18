"""Phase 6 probe orchestration: for one frozen encoder and one train/test
split seed, fit each probe family against each physical target on the
shared object-level split, and assemble row-level predictions plus metrics
recomputed strictly from those rows.

Every final probe -- ridge, logistic, and kNN -- is fit on the same
outer-train-only ``StandardScaler`` output. Ridge/logistic regularization is
chosen by identical k-fold CV with a fresh fold-train-only scaler inside
each fold (the same fold indices are reused across every encoder for a given
split seed), rather than one fixed alpha/C shared across encoders of very
different representation scale. Trivial baselines (train-median
redshift, majority-class spectral type) are computed once per split seed,
independent of any encoder, and reported through the same predictions/
metrics machinery so they land in the same tables. The whole suite runs
once per config-declared split seed; ``aggregate_seed_metrics`` collapses
those runs to mean/std, with the per-seed row-level predictions remaining
the primary evidence (architecture.md: "Row-level rankings are primary
evaluation evidence" -- the same principle applied to probe predictions).

Deliberately does not import aion_reimp.model / .training / .retrieval /
.losses / .datasets: probes are a bounded review instrument, not a retrieval
or training pipeline (architecture.md decision 12). It reuses only the
generic aion_reimp.manifest.manifest_fingerprint helper for embedding
fingerprints, matching the "reuse where genuinely identical" contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from aion_reimp.manifest import manifest_fingerprint

from .encoders import SpectrumBatch, SpectrumEncoderAdapter
from .probe_metrics import classification_metrics, regression_metrics
from .probes import (
    apply_scaler,
    fit_scaler,
    knn_classification_probe,
    knn_regression_probe,
    logistic_probe,
    majority_class_baseline_predict,
    median_baseline_predict,
    ridge_probe,
    select_logistic_c,
    select_ridge_alpha,
)

TARGET_SPEC_Z = "spec_z"
TARGET_SPECTYPE = "spectype"
PROBE_LINEAR = "linear"
PROBE_KNN = "knn"
PROBE_BASELINE = "baseline"
BASELINE_ENCODER = "trivial_baseline"
BASELINE_REVISION = "no-pretrained-revision"

PREDICTIONS_COLUMNS = (
    "object_id",
    "encoder",
    "encoder_revision",
    "target",
    "probe_family",
    "split",
    "split_seed",
    "y_true_numeric",
    "y_pred_numeric",
    "y_true_label",
    "y_pred_label",
    "hyperparameter_name",
    "hyperparameter_value",
)


def _row(
    encoder_name: str,
    encoder_revision: str,
    target: str,
    probe_family: str,
    split_seed: int,
    object_id: str,
    y_true_numeric: float = float("nan"),
    y_pred_numeric: float = float("nan"),
    y_true_label: str = "",
    y_pred_label: str = "",
    hyperparameter_name: str = "",
    hyperparameter_value: float = float("nan"),
) -> Dict[str, Any]:
    return {
        "object_id": str(object_id),
        "encoder": encoder_name,
        "encoder_revision": encoder_revision,
        "target": target,
        "probe_family": probe_family,
        "split": "test",
        "split_seed": int(split_seed),
        "y_true_numeric": y_true_numeric,
        "y_pred_numeric": y_pred_numeric,
        "y_true_label": y_true_label,
        "y_pred_label": y_pred_label,
        "hyperparameter_name": hyperparameter_name,
        "hyperparameter_value": hyperparameter_value,
    }


def run_probe_suite(
    encoder_name: str,
    encoder_revision: str,
    embeddings_train: np.ndarray,
    embeddings_test: np.ndarray,
    object_ids_test: Sequence[str],
    z_train: np.ndarray,
    z_test: np.ndarray,
    probe_config: Mapping[str, Any],
    cv_folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    split_seed: int,
    seed: int = 0,
    spectype_train: Optional[Sequence[str]] = None,
    spectype_test: Optional[Sequence[str]] = None,
    spectype_classes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Fit ridge/logistic/kNN probes and return one row-level predictions frame.

    Pure given already-embedded arrays: it never touches an encoder or the
    network, so it is fully exercisable with synthetic fixtures. Spectral-
    class probes only run when ``spectype_train``/``spectype_test``/
    ``spectype_classes`` are all provided (``labels.spectral_class.enabled``
    in config; currently always false -- see spectra_data.py).
    """
    linear_config = probe_config["linear"]
    knn_config = probe_config["knn"]
    rows: List[Dict[str, Any]] = []

    best_alpha = select_ridge_alpha(
        embeddings_train, z_train, linear_config["ridge_alpha_grid"], cv_folds, seed=seed
    )
    scaler = fit_scaler(embeddings_train)
    train_scaled = apply_scaler(scaler, embeddings_train)
    test_scaled = apply_scaler(scaler, embeddings_test)
    z_pred_ridge = ridge_probe(train_scaled, z_train, test_scaled, best_alpha, seed=seed)
    z_pred_knn = knn_regression_probe(train_scaled, z_train, test_scaled, int(knn_config["k"]))
    for probe_family, predictions, parameter_name, parameter_value in (
        (PROBE_LINEAR, z_pred_ridge, "ridge_alpha", best_alpha),
        (PROBE_KNN, z_pred_knn, "k", int(knn_config["k"])),
    ):
        for object_id, true_value, predicted_value in zip(object_ids_test, z_test, predictions):
            rows.append(
                _row(
                    encoder_name,
                    encoder_revision,
                    TARGET_SPEC_Z,
                    probe_family,
                    split_seed,
                    object_id,
                    y_true_numeric=float(true_value),
                    y_pred_numeric=float(predicted_value),
                    hyperparameter_name=parameter_name,
                    hyperparameter_value=float(parameter_value),
                )
            )

    spectral_class_enabled = spectype_train is not None and spectype_test is not None and spectype_classes is not None
    if spectral_class_enabled:
        best_c = select_logistic_c(
            embeddings_train,
            spectype_train,
            linear_config["logistic_c_grid"],
            cv_folds,
            max_iter=int(linear_config["logistic_max_iter"]),
            seed=seed,
        )
        spectype_pred_logistic, _ = logistic_probe(
            train_scaled,
            spectype_train,
            test_scaled,
            spectype_classes,
            C=best_c,
            max_iter=int(linear_config["logistic_max_iter"]),
            seed=seed,
        )
        spectype_pred_knn = knn_classification_probe(
            train_scaled, spectype_train, test_scaled, int(knn_config["k"])
        )
        for probe_family, predictions, parameter_name, parameter_value in (
            (PROBE_LINEAR, spectype_pred_logistic, "logistic_c", best_c),
            (PROBE_KNN, spectype_pred_knn, "k", int(knn_config["k"])),
        ):
            for object_id, true_value, predicted_value in zip(object_ids_test, spectype_test, predictions):
                rows.append(
                    _row(
                        encoder_name,
                        encoder_revision,
                        TARGET_SPECTYPE,
                        probe_family,
                        split_seed,
                        object_id,
                        y_true_label=str(true_value),
                        y_pred_label=str(predicted_value),
                        hyperparameter_name=parameter_name,
                        hyperparameter_value=float(parameter_value),
                    )
                )
    return pd.DataFrame(rows, columns=list(PREDICTIONS_COLUMNS))


def run_probe_suite_for_encoder(
    encoder: SpectrumEncoderAdapter,
    train_batch: SpectrumBatch,
    test_batch: SpectrumBatch,
    z_train: np.ndarray,
    z_test: np.ndarray,
    probe_config: Mapping[str, Any],
    cv_folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    split_seed: int,
    seed: int = 0,
    spectype_train: Optional[Sequence[str]] = None,
    spectype_test: Optional[Sequence[str]] = None,
    spectype_classes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Fit the encoder (no-op for frozen pretrained encoders), embed, and probe."""
    encoder.fit(train_batch)
    embeddings_train = encoder.embed(train_batch)
    embeddings_test = encoder.embed(test_batch)
    if embeddings_train.shape[1] != encoder.output_dim or embeddings_test.shape[1] != encoder.output_dim:
        raise ValueError(f"{encoder.name} embedded a dimension different from its declared output_dim")
    return run_probe_suite(
        encoder.name,
        encoder.revision,
        embeddings_train,
        embeddings_test,
        list(test_batch.object_id),
        z_train,
        z_test,
        probe_config,
        cv_folds,
        split_seed,
        seed=seed,
        spectype_train=spectype_train,
        spectype_test=spectype_test,
        spectype_classes=spectype_classes,
    )


def run_baseline_suite(
    object_ids_test: Sequence[str],
    z_train: np.ndarray,
    z_test: np.ndarray,
    split_seed: int,
    spectype_train: Optional[Sequence[str]] = None,
    spectype_test: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Trivial baselines: train-median redshift, majority-class spectral type.

    Computed once per split seed, independent of any encoder (there are no
    embeddings involved), and reported through the same predictions schema
    (``encoder`` is the synthetic name ``trivial_baseline``) so they appear
    in the same tables as every real encoder for direct comparison.
    """
    rows: List[Dict[str, Any]] = []
    z_pred_baseline = median_baseline_predict(z_train, len(object_ids_test))
    for object_id, true_value, predicted_value in zip(object_ids_test, z_test, z_pred_baseline):
        rows.append(
            _row(
                BASELINE_ENCODER,
                BASELINE_REVISION,
                TARGET_SPEC_Z,
                PROBE_BASELINE,
                split_seed,
                object_id,
                y_true_numeric=float(true_value),
                y_pred_numeric=float(predicted_value),
            )
        )
    if spectype_train is not None and spectype_test is not None:
        spectype_pred_baseline = majority_class_baseline_predict(spectype_train, len(object_ids_test))
        for object_id, true_value, predicted_value in zip(object_ids_test, spectype_test, spectype_pred_baseline):
            rows.append(
                _row(
                    BASELINE_ENCODER,
                    BASELINE_REVISION,
                    TARGET_SPECTYPE,
                    PROBE_BASELINE,
                    split_seed,
                    object_id,
                    y_true_label=str(true_value),
                    y_pred_label=str(predicted_value),
                )
            )
    return pd.DataFrame(rows, columns=list(PREDICTIONS_COLUMNS))


def embeddings_fingerprint(object_ids: Sequence[str], embeddings: np.ndarray) -> str:
    """Row-order-invariant fingerprint of an embedding matrix, via aion_reimp.manifest.

    Reuses ``manifest_fingerprint`` (a generic, object_id-keyed content hash)
    rather than re-deriving a second fingerprint scheme.
    """
    frame = pd.DataFrame(
        {"object_id": [str(value) for value in object_ids], "embedding": list(np.asarray(embeddings).tolist())}
    )
    return manifest_fingerprint(frame)


def metrics_from_predictions(
    predictions: pd.DataFrame,
    outlier_threshold: float,
    spectype_classes: Sequence[str],
) -> Dict[str, Any]:
    """Recompute every headline metric strictly from row-level predictions.

    Nothing here reads any other artifact; metrics.json must always be
    reproducible by re-running this function (and ``aggregate_seed_metrics``
    over its output) over predictions.parquet. Keyed by
    ``"{encoder}|{target}|{probe_family}|{split_seed}"`` -- one entry per
    split seed, per the config's ``split.seeds`` list.
    """
    required = {
        "encoder",
        "target",
        "probe_family",
        "split_seed",
        "y_true_numeric",
        "y_pred_numeric",
        "y_true_label",
        "y_pred_label",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions frame missing columns: {sorted(missing)}")

    results: Dict[str, Any] = {}
    for (encoder, target, probe_family, split_seed), group in predictions.groupby(
        ["encoder", "target", "probe_family", "split_seed"], sort=True
    ):
        key = f"{encoder}|{target}|{probe_family}|{int(split_seed)}"
        if target == TARGET_SPEC_Z:
            results[key] = regression_metrics(
                group["y_pred_numeric"].to_numpy(), group["y_true_numeric"].to_numpy(), outlier_threshold
            )
        elif target == TARGET_SPECTYPE:
            results[key] = classification_metrics(
                group["y_pred_label"].to_numpy(), group["y_true_label"].to_numpy(), spectype_classes
            )
        else:
            raise ValueError(f"Unknown probe target: {target}")
    return results


def aggregate_seed_metrics(metrics_by_seed: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse per-split-seed metrics (Finding 4d) to mean/std per
    encoder/target/probe_family, keyed by ``"{encoder}|{target}|{probe_family}"``.

    Every scalar numeric field present in the per-seed metric dict (e.g.
    ``nmad``, ``mae``, ``r2``, ``accuracy``, ``macro_f1``, ``n``) gets a
    ``{field}_mean`` and ``{field}_std`` entry; ``per_class`` (a nested
    dict, only present for classification targets) is carried forward from
    the first split seed as a representative diagnostic rather than
    averaged. ``n_seeds`` records how many split seeds contributed.
    """
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for key, value in metrics_by_seed.items():
        encoder, target, probe_family, _split_seed = key.split("|")
        grouped.setdefault((encoder, target, probe_family), []).append(value)

    aggregated: Dict[str, Any] = {}
    for (encoder, target, probe_family), values in grouped.items():
        key = f"{encoder}|{target}|{probe_family}"
        scalar_fields = [
            field
            for field, value in values[0].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        entry: Dict[str, Any] = {"n_seeds": len(values)}
        for field in scalar_fields:
            series = np.array([value[field] for value in values], dtype=np.float64)
            entry[f"{field}_mean"] = float(np.mean(series))
            entry[f"{field}_std"] = float(np.std(series, ddof=0))
        if "per_class" in values[0]:
            entry["per_class"] = values[0]["per_class"]
        aggregated[key] = entry
    return aggregated


def tables_from_metrics(metrics: Mapping[str, Any]) -> pd.DataFrame:
    """Compact, report-ready rows: one per encoder/target/probe_family,
    consuming the seed-aggregated dict from ``aggregate_seed_metrics``."""
    rows: List[Dict[str, Any]] = []
    for key, value in metrics.items():
        encoder, target, probe_family = key.split("|")
        row: Dict[str, Any] = {"encoder": encoder, "target": target, "probe_family": probe_family}
        row.update({field: field_value for field, field_value in value.items() if field != "per_class"})
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["encoder", "target", "probe_family"])
    return pd.DataFrame(rows).sort_values(["target", "probe_family", "encoder"]).reset_index(drop=True)
