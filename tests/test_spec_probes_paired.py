import numpy as np
import pandas as pd
import pytest

from spec_probes.encoders import SpectrumBatch
from spec_probes.paired import (
    build_paired_manifest,
    paired_error_bootstrap,
    run_paired_redshift_comparison,
)
from spec_probes.spectra_data import (
    assert_spectrum_value_binding,
    load_spectrum_batch,
    save_spectrum_batch,
)


def test_build_paired_manifest_uses_image_identity_and_locked_redshift():
    matches = pd.DataFrame(
        {
            "source_object_id": ["image-b", "image-a"],
            "desi_object_id": ["20", "10"],
            "source_row_id": [2, 1],
            "desi_ra": [20.0, 10.0],
            "desi_dec": [2.0, 1.0],
            "desi_z": [0.2, 0.1],
            "desi_zerr": [0.01, 0.01],
            "separation_arcsec": [0.2, 0.1],
        }
    )
    manifest = build_paired_manifest(matches)
    assert manifest["object_id"].tolist() == ["image-a", "image-b"]
    assert manifest["spectrum_object_id"].tolist() == ["10", "20"]
    assert manifest["z"].tolist() == [0.1, 0.2]


def test_paired_comparison_reuses_test_objects_across_conditions():
    rng = np.random.default_rng(7)
    n = 80
    z = np.linspace(0.01, 0.8, n)
    manifest = pd.DataFrame(
        {
            "object_id": [f"image-{index}" for index in range(n)],
            "image_object_id": [f"image-{index}" for index in range(n)],
            "spectrum_object_id": [f"spec-{index}" for index in range(n)],
            "source_row_id": np.arange(n),
            "spectrum_ra": np.linspace(150.0, 151.0, n),
            "spectrum_dec": np.linspace(1.0, 2.0, n),
            "z": z,
            "zerr": np.full(n, 0.01),
            "separation_arcsec": np.full(n, 0.1),
        }
    )
    image = rng.normal(size=(n, 4)).astype(np.float32)
    spectrum = np.column_stack([z, z**2, np.sin(z), np.ones(n)]).astype(np.float32)
    predictions, splits = run_paired_redshift_comparison(
        manifest,
        {
            "image_only": image,
            "spectrum_only": spectrum,
            "image_plus_spectrum": np.concatenate([image, spectrum], axis=1),
        },
        {"image_only": "image-r", "spectrum_only": "spec-r", "image_plus_spectrum": "both-r"},
        split_seeds=[11, 12],
        train_ratio=0.8,
        cv_folds=3,
        ridge_alpha_grid=[0.1, 1.0],
        seed=5,
    )
    assert set(predictions["encoder"]) == {
        "trivial_baseline", "image_only", "spectrum_only", "image_plus_spectrum"
    }
    for split_seed in (11, 12):
        groups = predictions.loc[predictions["split_seed"].eq(split_seed)].groupby("encoder")["object_id"]
        test_sets = [set(values) for _, values in groups]
        assert all(values == test_sets[0] for values in test_sets[1:])
    assert len(splits) == 2 * n

    comparison = paired_error_bootstrap(
        predictions,
        "image_only",
        "spectrum_only",
        scale="one_plus_z",
        n_resamples=500,
        seed=9,
    )
    assert comparison["n_unique_objects"] > 0
    assert comparison["error_improvement"] > 0
    assert comparison["decision"] == "comparison_better"


def test_spectrum_batch_cache_round_trip(tmp_path):
    batch = SpectrumBatch(
        object_id=np.array(["1", "2"]),
        flux=np.arange(8, dtype=np.float32).reshape(2, 4),
        wave=np.linspace(4000, 5000, 4, dtype=np.float32),
        ivar=np.ones((2, 4), dtype=np.float32),
        mask=np.zeros((2, 4), dtype=bool),
    )
    path = tmp_path / "spectra.npz"
    save_spectrum_batch(path, batch)
    loaded = load_spectrum_batch(path)
    np.testing.assert_array_equal(loaded.object_id, batch.object_id)
    np.testing.assert_allclose(loaded.flux, batch.flux)
    np.testing.assert_allclose(loaded.wave, batch.wave)


def test_spectrum_value_binding_checks_all_arrays():
    batch = SpectrumBatch(
        object_id=np.array(["1"]),
        flux=np.array([[1.0, 2.0]], dtype=np.float32),
        wave=np.array([4000.0, 4001.0], dtype=np.float32),
        ivar=np.array([[3.0, 4.0]], dtype=np.float32),
        mask=np.array([[False, True]]),
    )
    report = assert_spectrum_value_binding(batch, batch, "1")
    assert report["max_abs_flux_diff"] == 0.0
    changed = SpectrumBatch(
        object_id=batch.object_id,
        flux=batch.flux.copy(),
        wave=batch.wave,
        ivar=batch.ivar,
        mask=np.array([[False, False]]),
    )
    with pytest.raises(ValueError, match="mask"):
        assert_spectrum_value_binding(batch, changed, "1")


def test_paired_bootstrap_averages_overlapping_test_appearances_per_object():
    rows = []
    for object_id, split_seed, image_error, fusion_error in (
        ("a", 1, 1.0, 0.0),
        ("a", 2, 1.0, 0.0),
        ("b", 1, 0.0, 1.0),
    ):
        for condition, prediction in (
            ("image_only", image_error),
            ("image_plus_spectrum", fusion_error),
        ):
            rows.append(
                {
                    "object_id": object_id,
                    "encoder": condition,
                    "split_seed": split_seed,
                    "y_true_numeric": 0.0,
                    "y_pred_numeric": prediction,
                }
            )
    result = paired_error_bootstrap(
        pd.DataFrame(rows),
        "image_only",
        "image_plus_spectrum",
        scale="absolute",
        n_resamples=200,
        seed=3,
    )
    assert result["n_unique_objects"] == 2
    assert result["error_improvement"] == 0.0


def _two_condition_predictions(z_true, image_pred, fusion_pred):
    rows = []
    for index, (true_value, image_value, fusion_value) in enumerate(
        zip(z_true, image_pred, fusion_pred)
    ):
        for condition, prediction in (("image_only", image_value), ("image_plus_spectrum", fusion_value)):
            rows.append(
                {
                    "object_id": f"object-{index}",
                    "encoder": condition,
                    "split_seed": 1,
                    "y_true_numeric": float(true_value),
                    "y_pred_numeric": float(prediction),
                }
            )
    return pd.DataFrame(rows)


def test_paired_bootstrap_one_plus_z_scale_downweights_high_redshift_errors():
    # Identical raw |dz| improvements at z=0 and z=1: the absolute scale sees
    # the same improvement for both objects, the (1+z) scale halves the z=1 one.
    predictions = _two_condition_predictions(
        z_true=[0.0, 1.0], image_pred=[0.2, 1.2], fusion_pred=[0.0, 1.0]
    )
    absolute = paired_error_bootstrap(
        predictions, "image_only", "image_plus_spectrum",
        scale="absolute", n_resamples=100, seed=1,
    )
    normalized = paired_error_bootstrap(
        predictions, "image_only", "image_plus_spectrum",
        scale="one_plus_z", n_resamples=100, seed=1,
    )
    assert absolute["error_improvement"] == pytest.approx(0.2)
    assert normalized["error_improvement"] == pytest.approx((0.2 + 0.1) / 2)
    assert normalized["scale"] == "one_plus_z"
    assert "1 + z_true" in normalized["estimand"]


def test_paired_bootstrap_rejects_unknown_scale_and_incomplete_pairs():
    predictions = _two_condition_predictions(
        z_true=[0.1, 0.2], image_pred=[0.1, 0.3], fusion_pred=[0.1, 0.2]
    )
    with pytest.raises(ValueError, match="scale"):
        paired_error_bootstrap(
            predictions, "image_only", "image_plus_spectrum",
            scale="relative", n_resamples=10, seed=0,
        )
    incomplete = predictions.drop(index=predictions.index[-1])
    with pytest.raises(ValueError, match="incomplete"):
        paired_error_bootstrap(
            incomplete, "image_only", "image_plus_spectrum",
            scale="absolute", n_resamples=10, seed=0,
        )
