"""Pure metric formulas for the Phase 6 spectrum-encoder probes.

Mirrors aion_reimp.metrics in spirit (the sole graded implementations, pure
numpy, no I/O) but is scoped to spectroscopic-redshift and spectral-class
recovery, which aion_reimp.metrics does not define. Kept in this package
rather than added to aion_reimp.metrics because these formulas have no
retrieval or training use.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def redshift_residual(z_pred: Sequence[float], z_true: Sequence[float]) -> np.ndarray:
    """(z_pred - z_true) / (1 + z_true), the standard photo-/spec-z residual."""
    pred = np.asarray(z_pred, dtype=np.float64)
    true = np.asarray(z_true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError("z_pred and z_true must have the same shape")
    if pred.size == 0:
        raise ValueError("redshift_residual requires at least one value")
    return (pred - true) / (1.0 + true)


def nmad(residual: Sequence[float]) -> float:
    """Normalized median absolute deviation: 1.4826 * median(|x - median(x)|)."""
    values = np.asarray(residual, dtype=np.float64)
    if values.size == 0:
        raise ValueError("nmad requires at least one value")
    center = np.median(values)
    return float(1.4826 * np.median(np.abs(values - center)))


def spec_z_nmad(z_pred: Sequence[float], z_true: Sequence[float]) -> float:
    return nmad(redshift_residual(z_pred, z_true))


def catastrophic_outlier_fraction(
    z_pred: Sequence[float], z_true: Sequence[float], threshold: float = 0.15
) -> float:
    """Fraction of objects with |Delta z| / (1 + z_true) > threshold (strict)."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    residual = redshift_residual(z_pred, z_true)
    return float(np.mean(np.abs(residual) > threshold))


def mae(y_pred: Sequence[float], y_true: Sequence[float]) -> float:
    pred = np.asarray(y_pred, dtype=np.float64)
    true = np.asarray(y_true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError("y_pred and y_true must have the same shape")
    if pred.size == 0:
        raise ValueError("mae requires at least one value")
    return float(np.mean(np.abs(pred - true)))


def r2_score(y_pred: Sequence[float], y_true: Sequence[float]) -> float:
    pred = np.asarray(y_pred, dtype=np.float64)
    true = np.asarray(y_true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError("y_pred and y_true must have the same shape")
    if pred.size == 0:
        raise ValueError("r2_score requires at least one value")
    total = float(np.sum((true - np.mean(true)) ** 2))
    if total == 0.0:
        raise ValueError("r2_score is undefined when y_true has zero variance")
    residual = float(np.sum((true - pred) ** 2))
    return float(1.0 - residual / total)


def regression_metrics(
    z_pred: Sequence[float], z_true: Sequence[float], outlier_threshold: float = 0.15
) -> Dict[str, float]:
    """Bundle spec-z headline metrics: NMAD, catastrophic-outlier rate, MAE, R2."""
    return {
        "nmad": spec_z_nmad(z_pred, z_true),
        "catastrophic_outlier_fraction": catastrophic_outlier_fraction(
            z_pred, z_true, outlier_threshold
        ),
        "mae": mae(z_pred, z_true),
        "r2": r2_score(z_pred, z_true),
        "n": int(len(z_true)),
    }


def accuracy(y_pred: Sequence[str], y_true: Sequence[str]) -> float:
    pred = np.asarray(y_pred)
    true = np.asarray(y_true)
    if pred.shape != true.shape:
        raise ValueError("y_pred and y_true must have the same shape")
    if pred.size == 0:
        raise ValueError("accuracy requires at least one value")
    return float(np.mean(pred == true))


def per_class_counts(y_true: Sequence[str], labels: Sequence[str]) -> Dict[str, int]:
    true = np.asarray(y_true)
    return {str(label): int(np.sum(true == label)) for label in labels}


def precision_recall_f1(
    y_pred: Sequence[str], y_true: Sequence[str], labels: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    pred = np.asarray(y_pred)
    true = np.asarray(y_true)
    if pred.shape != true.shape:
        raise ValueError("y_pred and y_true must have the same shape")
    result: Dict[str, Dict[str, float]] = {}
    for label in labels:
        true_positive = int(np.sum((pred == label) & (true == label)))
        predicted_positive = int(np.sum(pred == label))
        actual_positive = int(np.sum(true == label))
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        result[str(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": actual_positive,
        }
    return result


def macro_f1(y_pred: Sequence[str], y_true: Sequence[str], labels: Sequence[str]) -> float:
    per_class = precision_recall_f1(y_pred, y_true, labels)
    return float(np.mean([per_class[str(label)]["f1"] for label in labels]))


def classification_metrics(
    y_pred: Sequence[str], y_true: Sequence[str], labels: Sequence[str]
) -> Dict[str, object]:
    """Bundle spectral-class headline metrics: accuracy, macro-F1, per-class counts."""
    per_class = precision_recall_f1(y_pred, y_true, labels)
    return {
        "accuracy": accuracy(y_pred, y_true),
        "macro_f1": float(np.mean([per_class[str(label)]["f1"] for label in labels])),
        "per_class": per_class,
        "n": int(len(y_true)),
    }
