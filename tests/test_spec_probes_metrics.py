import numpy as np
import pytest

from spec_probes.probe_metrics import (
    accuracy,
    catastrophic_outlier_fraction,
    classification_metrics,
    mae,
    macro_f1,
    nmad,
    per_class_counts,
    r2_score,
    regression_metrics,
    spec_z_nmad,
)


def test_nmad_hand_computed() -> None:
    residual = np.array([-0.02, -0.01, 0.0, 0.01, 0.05])
    # median = 0.0; abs deviations = [0.02, 0.01, 0.0, 0.01, 0.05]; median = 0.01
    assert nmad(residual) == pytest.approx(1.4826 * 0.01)


def test_spec_z_nmad_matches_manual_residual() -> None:
    z_true = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    z_pred = z_true + np.array([-0.02, -0.01, 0.0, 0.01, 0.05]) * (1 + z_true)
    assert spec_z_nmad(z_pred, z_true) == pytest.approx(1.4826 * 0.01)


def test_nmad_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        nmad([])


def test_catastrophic_outlier_fraction_edge_case_at_threshold_is_not_counted() -> None:
    z_true = np.array([0.0])
    z_pred = np.array([0.15])  # |Delta z| / (1 + 0) == 0.15 exactly
    assert catastrophic_outlier_fraction(z_pred, z_true, threshold=0.15) == 0.0


def test_catastrophic_outlier_fraction_just_above_threshold_is_counted() -> None:
    z_true = np.array([0.0])
    z_pred = np.array([0.150001])
    assert catastrophic_outlier_fraction(z_pred, z_true, threshold=0.15) == 1.0


def test_catastrophic_outlier_fraction_mixed() -> None:
    z_true = np.array([0.0, 0.0, 1.0, 1.0])
    z_pred = np.array([0.0, 0.2, 1.0, 1.5])  # residuals: 0, 0.2, 0, 0.25
    assert catastrophic_outlier_fraction(z_pred, z_true, threshold=0.15) == pytest.approx(0.5)


def test_mae_hand_computed() -> None:
    assert mae([1.0, 2.0, 3.0], [1.5, 2.0, 2.0]) == pytest.approx((0.5 + 0.0 + 1.0) / 3)


def test_r2_perfect_prediction_is_one() -> None:
    y_true = [1.0, 2.0, 3.0, 4.0]
    assert r2_score(y_true, y_true) == pytest.approx(1.0)


def test_r2_rejects_zero_variance_truth() -> None:
    with pytest.raises(ValueError):
        r2_score([1.0, 1.0], [1.0, 1.0])


def test_accuracy_hand_computed() -> None:
    assert accuracy(["A", "B", "A"], ["A", "B", "B"]) == pytest.approx(2 / 3)


def test_per_class_counts() -> None:
    counts = per_class_counts(["GALAXY", "GALAXY", "QSO"], ["GALAXY", "QSO", "STAR"])
    assert counts == {"GALAXY": 2, "QSO": 1, "STAR": 0}


def test_macro_f1_hand_computed() -> None:
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "B"]
    # class A: tp=1, predicted_positive=1, actual_positive=2 -> precision=1, recall=0.5, f1=2/3
    # class B: tp=2, predicted_positive=3, actual_positive=2 -> precision=2/3, recall=1, f1=0.8
    expected = float(np.mean([2 / 3, 0.8]))
    assert macro_f1(y_pred, y_true, ["A", "B"]) == pytest.approx(expected)


def test_regression_metrics_bundle_has_expected_keys() -> None:
    z_true = np.array([0.5, 1.0, 1.5])
    z_pred = np.array([0.5, 1.0, 1.5])
    result = regression_metrics(z_pred, z_true)
    assert set(result) == {"nmad", "catastrophic_outlier_fraction", "mae", "r2", "n"}
    assert result["nmad"] == 0.0
    assert result["n"] == 3


def test_classification_metrics_bundle_has_expected_keys() -> None:
    result = classification_metrics(["A", "B"], ["A", "A"], ["A", "B"])
    assert set(result) == {"accuracy", "macro_f1", "per_class", "n"}
    assert result["n"] == 2
