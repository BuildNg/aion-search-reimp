"""Ridge / multinomial-logistic / kNN probe fitting and prediction, plus the
shared fairness machinery every probe family and every encoder goes through
identically: train-fitted standardization, k-fold regularization selection
on a fixed, encoder-independent set of folds, and trivial baselines.

Pure functions: frozen encoder embeddings and labels go in, predictions
come out. No file I/O and no encoder-specific code lives here (that stays
in encoders.py, per the module-ownership split this package mirrors from
aion_reimp). Every probe kind is deterministic given fixed inputs: Ridge is
a closed-form solve, the multinomial logistic fit uses the deterministic
'lbfgs' solver, kNN has no stochastic step, and StandardScaler/KFold are
themselves deterministic given their inputs and (for KFold) a fixed seed.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a ``StandardScaler`` on the train split only.

    Every probe -- ridge, logistic, and kNN alike -- consumes embeddings
    through this same fitted scaler, so raw representation scale (128 for
    PCA, 768 for the neural encoders) no longer interacts differently
    with each probe family's regularization or distance metric.
    """
    scaler = StandardScaler()
    scaler.fit(np.asarray(X_train, dtype=np.float64))
    return scaler


def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(np.asarray(X, dtype=np.float64))


def _scaled_cv_views(
    X_train: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Fit one scaler per fold and reuse its views across grid candidates."""
    views = []
    for train_idx, test_idx in folds:
        scaler = fit_scaler(X_train[train_idx])
        views.append(
            (
                train_idx,
                test_idx,
                apply_scaler(scaler, X_train[train_idx]),
                apply_scaler(scaler, X_train[test_idx]),
            )
        )
    return views


def make_cv_folds(n_samples: int, cv_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """One seeded k-fold split of ``range(n_samples)``, computed once.

    Callers pass the *same* returned list into ``select_ridge_alpha`` and
    ``select_logistic_c`` for every encoder being compared on a given data
    split, so "same folds across encoders" is a structural guarantee (the
    identical index arrays are reused), not just a probabilistic
    consequence of passing the same seed.
    """
    if n_samples < 2:
        raise ValueError("make_cv_folds requires at least two samples")
    if cv_folds < 2 or cv_folds > n_samples:
        raise ValueError("cv_folds must be between 2 and n_samples")
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    return [(train_idx, test_idx) for train_idx, test_idx in kfold.split(np.arange(n_samples))]


def select_ridge_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha_grid: Sequence[float],
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    seed: int = 0,
) -> float:
    """Pick the ridge_alpha in ``alpha_grid`` with the lowest mean CV MAE.

    Each candidate is evaluated with a scaler fitted only on that fold's
    training rows; the validation fold never contributes scaling statistics.
    Ties are broken by the smaller alpha, so the result is reproducible.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    scaled_folds = _scaled_cv_views(X_train, folds)
    best_alpha = None
    best_score = np.inf
    for alpha in alpha_grid:
        fold_errors = []
        for train_idx, test_idx, fold_train, fold_test in scaled_folds:
            model = Ridge(alpha=float(alpha), solver="svd", random_state=seed)
            model.fit(fold_train, y_train[train_idx])
            prediction = model.predict(fold_test)
            fold_errors.append(float(np.mean(np.abs(prediction - y_train[test_idx]))))
        mean_error = float(np.mean(fold_errors))
        if mean_error < best_score or (mean_error == best_score and float(alpha) < best_alpha):
            best_score = mean_error
            best_alpha = float(alpha)
    return best_alpha


def select_logistic_c(
    X_train: np.ndarray,
    y_train: Sequence[str],
    c_grid: Sequence[float],
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    max_iter: int,
    seed: int = 0,
) -> float:
    """Pick the logistic_c in ``c_grid`` with the highest mean CV accuracy.

    Each candidate is evaluated with a scaler fitted only on that fold's
    training rows. Ties are broken by the smaller C, so the result is
    reproducible.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    scaled_folds = _scaled_cv_views(X_train, folds)
    best_c = None
    best_score = -np.inf
    for c in c_grid:
        fold_scores = []
        for train_idx, test_idx, fold_train, fold_test in scaled_folds:
            model = LogisticRegression(
                C=float(c), max_iter=max_iter, multi_class="multinomial", solver="lbfgs", random_state=seed
            )
            model.fit(fold_train, y_train[train_idx])
            prediction = model.predict(fold_test)
            fold_scores.append(float(np.mean(prediction == y_train[test_idx])))
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score or (mean_score == best_score and float(c) < best_c):
            best_score = mean_score
            best_c = float(c)
    return best_c


def median_baseline_predict(y_train: Sequence[float], n_test: int) -> np.ndarray:
    """Trivial regression baseline: predict the train-split median for every test row."""
    y_train = np.asarray(y_train, dtype=np.float64)
    if y_train.size == 0:
        raise ValueError("median_baseline_predict requires a non-empty y_train")
    return np.full(int(n_test), float(np.median(y_train)), dtype=np.float64)


def majority_class_baseline_predict(y_train: Sequence[str], n_test: int) -> np.ndarray:
    """Trivial classification baseline: predict the train-split majority class for every test row.

    Ties broken alphabetically (``np.unique`` returns sorted labels), so the
    result is fully reproducible.
    """
    y_train = np.asarray(y_train)
    if y_train.size == 0:
        raise ValueError("majority_class_baseline_predict requires a non-empty y_train")
    labels, counts = np.unique(y_train, return_counts=True)
    majority_label = labels[int(np.argmax(counts))]
    return np.full(int(n_test), majority_label, dtype=y_train.dtype)


def ridge_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    seed: int = 0,
) -> np.ndarray:
    """Ridge regression for spec-z.

    solver="svd" is pinned explicitly (rather than "auto") because it is the
    one Ridge solver with no version-dependent SciPy keyword surface (older
    scikit-learn's "cholesky"/"auto" path calls `scipy.linalg.solve(...,
    sym_pos=...)`, a kwarg SciPy has since removed), so this stays correct
    across the local test environment's older scikit-learn and whatever
    modern version runs on the cluster.
    """
    model = Ridge(alpha=alpha, solver="svd", random_state=seed)
    model.fit(np.asarray(X_train, dtype=np.float64), np.asarray(y_train, dtype=np.float64))
    return model.predict(np.asarray(X_test, dtype=np.float64))


def logistic_probe(
    X_train: np.ndarray,
    y_train: Sequence[str],
    X_test: np.ndarray,
    labels: Sequence[str],
    C: float,
    max_iter: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multinomial logistic regression for spectral class.

    Returns (predicted_labels, predicted_probabilities); probability columns
    are reindexed to match ``labels`` exactly, regardless of the order
    scikit-learn assigns to ``model.classes_``.
    """
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(np.asarray(X_train, dtype=np.float64), np.asarray(y_train))
    predicted = model.predict(np.asarray(X_test, dtype=np.float64))
    proba_raw = model.predict_proba(np.asarray(X_test, dtype=np.float64))
    column_for_label = {label: index for index, label in enumerate(model.classes_)}
    proba = np.zeros((proba_raw.shape[0], len(labels)), dtype=np.float64)
    for output_index, label in enumerate(labels):
        source_index = column_for_label.get(label)
        if source_index is not None:
            proba[:, output_index] = proba_raw[:, source_index]
    return predicted, proba


def knn_regression_probe(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int
) -> np.ndarray:
    """kNN regression for spec-z under cosine distance."""
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(X_train):
        raise ValueError(f"k={k} exceeds training set size {len(X_train)}")
    model = KNeighborsRegressor(n_neighbors=k, metric="cosine", algorithm="brute")
    model.fit(np.asarray(X_train, dtype=np.float64), np.asarray(y_train, dtype=np.float64))
    return model.predict(np.asarray(X_test, dtype=np.float64))


def knn_classification_probe(
    X_train: np.ndarray, y_train: Sequence[str], X_test: np.ndarray, k: int
) -> np.ndarray:
    """kNN majority-vote classification for spectral class under cosine distance."""
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(X_train):
        raise ValueError(f"k={k} exceeds training set size {len(X_train)}")
    model = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
    model.fit(np.asarray(X_train, dtype=np.float64), np.asarray(y_train))
    return model.predict(np.asarray(X_test, dtype=np.float64))
