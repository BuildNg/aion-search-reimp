import numpy as np
import pytest

from spec_probes.probes import (
    apply_scaler,
    fit_scaler,
    knn_classification_probe,
    knn_regression_probe,
    logistic_probe,
    majority_class_baseline_predict,
    make_cv_folds,
    median_baseline_predict,
    ridge_probe,
    select_logistic_c,
    select_ridge_alpha,
)


def _linear_regression_fixture(n=40, dim=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    true_weights = rng.normal(size=dim)
    y = X @ true_weights + 0.01 * rng.normal(size=n)
    return X, y


def test_ridge_probe_is_deterministic() -> None:
    X, y = _linear_regression_fixture()
    X_train, X_test = X[:30], X[30:]
    y_train = y[:30]
    first = ridge_probe(X_train, y_train, X_test, alpha=1.0, seed=5)
    second = ridge_probe(X_train, y_train, X_test, alpha=1.0, seed=5)
    np.testing.assert_array_equal(first, second)


def test_ridge_probe_recovers_a_near_linear_relationship() -> None:
    X, y = _linear_regression_fixture(n=200, dim=4, seed=1)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    predictions = ridge_probe(X_train, y_train, X_test, alpha=0.1, seed=0)
    mean_absolute_error = float(np.mean(np.abs(predictions - y_test)))
    assert mean_absolute_error < 0.5


def test_logistic_probe_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 4))
    y = np.where(X[:, 0] > 0, "GALAXY", "STAR")
    X_train, X_test = X[:45], X[45:]
    y_train = y[:45]
    labels = ["GALAXY", "QSO", "STAR"]
    predicted_a, proba_a = logistic_probe(X_train, y_train, X_test, labels, C=1.0, max_iter=200, seed=1)
    predicted_b, proba_b = logistic_probe(X_train, y_train, X_test, labels, C=1.0, max_iter=200, seed=1)
    np.testing.assert_array_equal(predicted_a, predicted_b)
    np.testing.assert_allclose(proba_a, proba_b)


def test_logistic_probe_proba_columns_match_requested_label_order() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 3))
    y = np.array(["GALAXY", "QSO", "STAR"])[rng.integers(0, 3, size=60)]
    labels = ["GALAXY", "QSO", "STAR"]
    predicted, proba = logistic_probe(X, y, X, labels, C=1.0, max_iter=200, seed=0)
    assert proba.shape == (60, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(60), atol=1e-6)
    for row_prediction, row_proba in zip(predicted, proba):
        assert labels[int(np.argmax(row_proba))] == row_prediction


def test_knn_regression_probe_matches_nearest_neighbor_for_k1() -> None:
    X_train = np.eye(3)
    y_train = np.array([10.0, 20.0, 30.0])
    X_test = np.array([[0.9, 0.1, 0.0]])
    prediction = knn_regression_probe(X_train, y_train, X_test, k=1)
    assert prediction[0] == pytest.approx(10.0)


def test_knn_classification_probe_matches_nearest_neighbor_for_k1() -> None:
    X_train = np.eye(3)
    y_train = np.array(["GALAXY", "QSO", "STAR"])
    X_test = np.array([[0.0, 1.0, 0.05]])
    prediction = knn_classification_probe(X_train, y_train, X_test, k=1)
    assert prediction[0] == "QSO"


def test_knn_regression_is_deterministic() -> None:
    X_train = np.eye(3)
    y_train = np.array([10.0, 20.0, 30.0])
    X_test = np.array([[0.9, 0.1, 0.0]])
    first = knn_regression_probe(X_train, y_train, X_test, k=2)
    second = knn_regression_probe(X_train, y_train, X_test, k=2)
    np.testing.assert_array_equal(first, second)


def test_knn_rejects_k_larger_than_train_set() -> None:
    X_train = np.eye(2)
    y_train = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="exceeds training set size"):
        knn_regression_probe(X_train, y_train, X_train, k=5)


# -- Finding 4: train-fitted standardization, CV regularization selection, baselines --


def test_fit_scaler_is_deterministic_and_train_only() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(loc=5.0, scale=3.0, size=(50, 4))
    scaler_a = fit_scaler(X_train)
    scaler_b = fit_scaler(X_train)
    np.testing.assert_allclose(scaler_a.mean_, scaler_b.mean_)
    np.testing.assert_allclose(scaler_a.scale_, scaler_b.scale_)
    # Hand-computed: scaler statistics come from the train split's own mean/std.
    np.testing.assert_allclose(scaler_a.mean_, X_train.mean(axis=0))
    np.testing.assert_allclose(scaler_a.scale_, X_train.std(axis=0))


def test_apply_scaler_standardizes_the_train_split_itself() -> None:
    rng = np.random.default_rng(1)
    X_train = rng.normal(loc=10.0, scale=2.0, size=(60, 3))
    scaler = fit_scaler(X_train)
    transformed = apply_scaler(scaler, X_train)
    np.testing.assert_allclose(transformed.mean(axis=0), np.zeros(3), atol=1e-8)
    np.testing.assert_allclose(transformed.std(axis=0), np.ones(3), atol=1e-8)


def test_apply_scaler_uses_train_statistics_on_test_data() -> None:
    """The scaler must be fit on train only: applying it to a shifted test
    set must not re-center on the test set's own mean."""
    X_train = np.zeros((10, 2))
    scaler = fit_scaler(X_train)  # mean=0, scale=0 -> StandardScaler treats zero-variance as scale 1
    X_test = np.ones((3, 2)) * 5.0
    transformed = apply_scaler(scaler, X_test)
    np.testing.assert_allclose(transformed, np.ones((3, 2)) * 5.0)


def test_make_cv_folds_is_deterministic_given_fixed_seed() -> None:
    first = make_cv_folds(20, cv_folds=4, seed=7)
    second = make_cv_folds(20, cv_folds=4, seed=7)
    assert len(first) == len(second) == 4
    for (train_a, test_a), (train_b, test_b) in zip(first, second):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)


def test_make_cv_folds_partitions_every_index_exactly_once_per_fold() -> None:
    folds = make_cv_folds(20, cv_folds=5, seed=3)
    covered = np.concatenate([test_idx for _, test_idx in folds])
    assert sorted(covered.tolist()) == list(range(20))
    for train_idx, test_idx in folds:
        assert set(train_idx.tolist()) & set(test_idx.tolist()) == set()


def test_cv_folds_reused_across_encoders_are_identical_by_construction() -> None:
    """"Same folds across encoders" (Finding 4b) is a structural property:
    one make_cv_folds() call, reused for every encoder's grid search --
    not just "same seed produces the same folds" independently per call."""
    folds = make_cv_folds(30, cv_folds=3, seed=11)
    # Simulate two different encoders' embeddings consuming the identical
    # folds object.
    encoder_a_embeddings = np.random.default_rng(0).normal(size=(30, 5))
    encoder_b_embeddings = np.random.default_rng(1).normal(size=(30, 9))
    y = np.random.default_rng(2).normal(size=30)
    alpha_a = select_ridge_alpha(encoder_a_embeddings, y, [1.0, 10.0], folds, seed=0)
    alpha_b = select_ridge_alpha(encoder_b_embeddings, y, [1.0, 10.0], folds, seed=0)
    assert alpha_a in {1.0, 10.0}
    assert alpha_b in {1.0, 10.0}


def test_select_ridge_alpha_picks_the_grid_value_with_lowest_cv_error() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 6))
    true_weights = rng.normal(size=6)
    y = X @ true_weights  # near-noiseless linear signal favors small alpha
    folds = make_cv_folds(80, cv_folds=5, seed=1)
    best_alpha = select_ridge_alpha(X, y, [0.001, 1000.0], folds, seed=0)
    assert best_alpha == 0.001


def test_select_ridge_alpha_fits_scaler_inside_each_cv_training_fold(monkeypatch) -> None:
    import spec_probes.probes as probes_module

    X, y = _linear_regression_fixture(n=20, dim=3, seed=7)
    folds = make_cv_folds(20, cv_folds=4, seed=2)
    observed_fit_sizes = []
    real_fit_scaler = probes_module.fit_scaler

    def recording_fit_scaler(values):
        observed_fit_sizes.append(len(values))
        return real_fit_scaler(values)

    monkeypatch.setattr(probes_module, "fit_scaler", recording_fit_scaler)
    select_ridge_alpha(X, y, [0.1, 1.0], folds, seed=0)
    assert observed_fit_sizes == [15] * 4  # one scaler per fold, reused across both alpha candidates


def test_select_ridge_alpha_breaks_ties_toward_the_smaller_value() -> None:
    # Constant y: every alpha gives identical CV error, so the tie-break
    # (smaller alpha) is the only thing determining the result.
    X = np.zeros((20, 3))
    y = np.ones(20) * 4.0
    folds = make_cv_folds(20, cv_folds=4, seed=0)
    best_alpha = select_ridge_alpha(X, y, [10.0, 1.0, 5.0], folds, seed=0)
    assert best_alpha == 1.0


def test_select_logistic_c_picks_the_grid_value_with_highest_cv_accuracy() -> None:
    rng = np.random.default_rng(6)
    X = rng.normal(size=(90, 4))
    y = np.where(X[:, 0] > 0, "GALAXY", "STAR")
    folds = make_cv_folds(90, cv_folds=3, seed=2)
    best_c = select_logistic_c(X, y, [0.001, 1.0], folds, max_iter=200, seed=0)
    assert best_c in {0.001, 1.0}


def test_median_baseline_predict_matches_hand_computed_median() -> None:
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    predictions = median_baseline_predict(y_train, n_test=4)
    assert predictions.tolist() == [3.0, 3.0, 3.0, 3.0]


def test_majority_class_baseline_predict_matches_hand_computed_mode() -> None:
    y_train = np.array(["GALAXY", "GALAXY", "STAR", "GALAXY", "QSO"])
    predictions = majority_class_baseline_predict(y_train, n_test=3)
    assert predictions.tolist() == ["GALAXY", "GALAXY", "GALAXY"]


def test_majority_class_baseline_predict_breaks_ties_alphabetically() -> None:
    y_train = np.array(["STAR", "GALAXY"])  # tied 1-1 -> alphabetically first
    predictions = majority_class_baseline_predict(y_train, n_test=2)
    assert predictions.tolist() == ["GALAXY", "GALAXY"]
