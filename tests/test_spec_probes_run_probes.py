import numpy as np
import pytest

from spec_probes.encoders import SpectrumBatch, SpectrumEncoderAdapter
from spec_probes.probe_metrics import classification_metrics, regression_metrics
from spec_probes.probes import make_cv_folds
from spec_probes.run_probes import (
    BASELINE_ENCODER,
    PROBE_BASELINE,
    PROBE_KNN,
    PROBE_LINEAR,
    TARGET_SPEC_Z,
    TARGET_SPECTYPE,
    aggregate_seed_metrics,
    embeddings_fingerprint,
    metrics_from_predictions,
    run_baseline_suite,
    run_probe_suite,
    run_probe_suite_for_encoder,
    tables_from_metrics,
)


PROBE_CONFIG = {
    "linear": {"ridge_alpha_grid": [0.1, 1.0, 10.0], "logistic_c_grid": [0.1, 1.0, 10.0], "logistic_max_iter": 200},
    "knn": {"k": 3, "metric": "cosine"},
}
SPECTYPE_CLASSES = ["GALAXY", "QSO", "STAR"]


def _synthetic(n=60, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n, dim))
    z = rng.uniform(0.1, 1.5, size=n)
    spectype = np.array(SPECTYPE_CLASSES)[rng.integers(0, 3, size=n)]
    object_ids = [f"obj-{i:03d}" for i in range(n)]
    return object_ids, embeddings, z, spectype


def _run(seed=0, n=60, split_seed=99, with_spectype=True):
    object_ids, embeddings, z, spectype = _synthetic(n=n, seed=seed)
    split_point = int(n * 0.75)
    train_slice, test_slice = slice(0, split_point), slice(split_point, n)
    folds = make_cv_folds(split_point, cv_folds=3, seed=0)
    kwargs = {}
    if with_spectype:
        kwargs = dict(
            spectype_train=spectype[train_slice],
            spectype_test=spectype[test_slice],
            spectype_classes=SPECTYPE_CLASSES,
        )
    return (
        run_probe_suite(
            "fake_encoder",
            "fake-rev",
            embeddings[train_slice],
            embeddings[test_slice],
            object_ids[split_point:],
            z[train_slice],
            z[test_slice],
            PROBE_CONFIG,
            folds,
            split_seed,
            seed=0,
            **kwargs,
        ),
        n - split_point,
    )


def test_run_probe_suite_returns_expected_predictions_schema() -> None:
    predictions, n_test = _run()
    assert set(predictions["target"]) == {TARGET_SPEC_Z, TARGET_SPECTYPE}
    assert set(predictions["probe_family"]) == {PROBE_LINEAR, PROBE_KNN}
    # n_test objects x 2 targets x 2 probe families
    assert len(predictions) == n_test * 2 * 2
    assert set(predictions["split"]) == {"test"}
    assert set(predictions["split_seed"]) == {99}
    assert set(predictions["hyperparameter_name"]) == {"ridge_alpha", "logistic_c", "k"}
    assert predictions["hyperparameter_value"].notna().all()


def test_run_probe_suite_skips_spectype_when_disabled() -> None:
    predictions, n_test = _run(with_spectype=False)
    assert set(predictions["target"]) == {TARGET_SPEC_Z}
    assert len(predictions) == n_test * 2  # spec_z only, 2 probe families


def test_metrics_from_predictions_matches_direct_computation() -> None:
    predictions, _ = _run(seed=1)
    metrics = metrics_from_predictions(predictions, outlier_threshold=0.15, spectype_classes=SPECTYPE_CLASSES)

    linear_z_rows = predictions[
        (predictions["target"] == TARGET_SPEC_Z) & (predictions["probe_family"] == PROBE_LINEAR)
    ]
    expected = regression_metrics(
        linear_z_rows["y_pred_numeric"].to_numpy(), linear_z_rows["y_true_numeric"].to_numpy(), 0.15
    )
    assert metrics["fake_encoder|spec_z|linear|99"] == expected

    knn_class_rows = predictions[
        (predictions["target"] == TARGET_SPECTYPE) & (predictions["probe_family"] == PROBE_KNN)
    ]
    expected_class = classification_metrics(
        knn_class_rows["y_pred_label"].to_numpy(), knn_class_rows["y_true_label"].to_numpy(), SPECTYPE_CLASSES
    )
    assert metrics["fake_encoder|spectype|knn|99"] == expected_class


def test_metrics_from_predictions_rejects_missing_columns() -> None:
    predictions, _ = _run(seed=5)
    with pytest.raises(ValueError, match="missing columns"):
        metrics_from_predictions(predictions.drop(columns=["y_pred_numeric"]), 0.15, SPECTYPE_CLASSES)


def test_aggregate_seed_metrics_matches_hand_computed_mean_and_std() -> None:
    predictions_a, _ = _run(seed=1, split_seed=1)
    predictions_b, _ = _run(seed=2, split_seed=2)
    import pandas as pd

    combined = pd.concat([predictions_a, predictions_b], ignore_index=True)
    per_seed = metrics_from_predictions(combined, outlier_threshold=0.15, spectype_classes=SPECTYPE_CLASSES)
    aggregated = aggregate_seed_metrics(per_seed)

    key = "fake_encoder|spec_z|linear"
    seed_1_nmad = per_seed["fake_encoder|spec_z|linear|1"]["nmad"]
    seed_2_nmad = per_seed["fake_encoder|spec_z|linear|2"]["nmad"]
    assert aggregated[key]["n_seeds"] == 2
    assert aggregated[key]["nmad_mean"] == pytest.approx(np.mean([seed_1_nmad, seed_2_nmad]))
    assert aggregated[key]["nmad_std"] == pytest.approx(np.std([seed_1_nmad, seed_2_nmad]))


def test_aggregate_seed_metrics_keeps_per_class_from_first_seed() -> None:
    predictions_a, _ = _run(seed=1, split_seed=1)
    predictions_b, _ = _run(seed=2, split_seed=2)
    import pandas as pd

    combined = pd.concat([predictions_a, predictions_b], ignore_index=True)
    per_seed = metrics_from_predictions(combined, outlier_threshold=0.15, spectype_classes=SPECTYPE_CLASSES)
    aggregated = aggregate_seed_metrics(per_seed)
    key = "fake_encoder|spectype|linear"
    assert "per_class" in aggregated[key]


def test_tables_from_metrics_has_one_row_per_combination() -> None:
    predictions, _ = _run(seed=2)
    per_seed = metrics_from_predictions(predictions, 0.15, SPECTYPE_CLASSES)
    aggregated = aggregate_seed_metrics(per_seed)
    table = tables_from_metrics(aggregated)
    assert len(table) == 4  # 2 targets x 2 probe families x 1 encoder
    assert "nmad_mean" in table.columns or "accuracy_mean" in table.columns


class _FakeEncoder(SpectrumEncoderAdapter):
    name = "fake_batch_encoder"

    def __init__(self, output_dim=6, seed=0):
        self.output_dim = output_dim
        self.revision = "fake-rev"
        self._rng = np.random.default_rng(seed)

    def embed(self, batch):
        return self._rng.normal(size=(len(batch), self.output_dim)).astype(np.float32)


def _batch(object_ids, n_pix=10, seed=0):
    rng = np.random.default_rng(seed)
    n = len(object_ids)
    return SpectrumBatch(
        object_id=np.array(object_ids),
        flux=1.0 + 0.05 * rng.normal(size=(n, n_pix)),
        wave=np.linspace(3600.0, 9800.0, n_pix),
    )


def test_run_probe_suite_for_encoder_exercises_the_adapter_interface() -> None:
    object_ids, _, z, spectype = _synthetic(n=40, seed=3)
    train_ids, test_ids = object_ids[:30], object_ids[30:]
    encoder = _FakeEncoder(output_dim=5)
    folds = make_cv_folds(30, cv_folds=3, seed=0)
    predictions = run_probe_suite_for_encoder(
        encoder,
        _batch(train_ids, seed=1),
        _batch(test_ids, seed=2),
        z[:30],
        z[30:],
        PROBE_CONFIG,
        folds,
        split_seed=7,
        seed=0,
        spectype_train=spectype[:30],
        spectype_test=spectype[30:],
        spectype_classes=SPECTYPE_CLASSES,
    )
    assert set(predictions["object_id"]) == set(test_ids)
    assert (predictions["encoder"] == "fake_batch_encoder").all()
    assert (predictions["split_seed"] == 7).all()


def test_run_probe_suite_for_encoder_rejects_dimension_mismatch() -> None:
    object_ids, _, z, spectype = _synthetic(n=20, seed=4)
    folds = make_cv_folds(15, cv_folds=3, seed=0)

    class _BadEncoder(SpectrumEncoderAdapter):
        name = "bad"

        def __init__(self):
            self.output_dim = 5
            self.revision = "bad-rev"

        def embed(self, batch):
            return np.zeros((len(batch), 3), dtype=np.float32)  # wrong dimension

    with pytest.raises(ValueError, match="different from its declared output_dim"):
        run_probe_suite_for_encoder(
            _BadEncoder(),
            _batch(object_ids[:15]),
            _batch(object_ids[15:]),
            z[:15],
            z[15:],
            PROBE_CONFIG,
            folds,
            split_seed=0,
            seed=0,
        )


def test_run_baseline_suite_uses_trivial_encoder_name() -> None:
    object_ids, _, z, spectype = _synthetic(n=20, seed=8)
    predictions = run_baseline_suite(
        object_ids[15:], z[:15], z[15:], split_seed=3, spectype_train=spectype[:15], spectype_test=spectype[15:]
    )
    assert set(predictions["encoder"]) == {BASELINE_ENCODER}
    assert set(predictions["probe_family"]) == {PROBE_BASELINE}
    assert set(predictions["target"]) == {TARGET_SPEC_Z, TARGET_SPECTYPE}


def test_run_baseline_suite_matches_hand_computed_median_and_majority() -> None:
    object_ids, _, z, spectype = _synthetic(n=20, seed=9)
    predictions = run_baseline_suite(
        object_ids[15:], z[:15], z[15:], split_seed=1, spectype_train=spectype[:15], spectype_test=spectype[15:]
    )
    z_rows = predictions[predictions["target"] == TARGET_SPEC_Z]
    assert (z_rows["y_pred_numeric"] == np.median(z[:15])).all()
    labels, counts = np.unique(spectype[:15], return_counts=True)
    majority = labels[int(np.argmax(counts))]
    label_rows = predictions[predictions["target"] == TARGET_SPECTYPE]
    assert (label_rows["y_pred_label"] == majority).all()


def test_embeddings_fingerprint_is_row_order_invariant_and_content_sensitive() -> None:
    object_ids = ["a", "b", "c"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    forward = embeddings_fingerprint(object_ids, embeddings)
    reversed_fingerprint = embeddings_fingerprint(list(reversed(object_ids)), embeddings[::-1])
    assert forward == reversed_fingerprint
    mutated = embeddings.copy()
    mutated[0, 0] = 5.0
    assert embeddings_fingerprint(object_ids, mutated) != forward
