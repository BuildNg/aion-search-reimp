import numpy as np
import pandas as pd
import pytest

from spec_probes.encoders import SpectrumBatch
from spec_probes.spectra_data import (
    _zwarn_is_good,
    assert_no_split_leakage,
    extract_labels,
    extract_spectrum_batch,
    object_level_split,
    split_fingerprint,
    verify_required_columns,
)


def _object_ids(n: int):
    return [f"obj-{i:05d}" for i in range(n)]


def test_split_is_deterministic_regardless_of_input_row_order() -> None:
    ids = _object_ids(500)
    first = object_level_split(ids, seed=7, train_ratio=0.8)
    second = object_level_split(list(reversed(ids)), seed=7, train_ratio=0.8)
    assert first["object_id"].tolist() == second["object_id"].tolist()
    assert first["split"].tolist() == second["split"].tolist()


def test_split_is_identical_across_simulated_encoder_runs() -> None:
    """The split must be derived once and reused. Two independent calls with
    the same seed -- standing in for two different encoders both consuming
    the one shared split -- must agree exactly, including their fingerprint."""
    ids = _object_ids(300)
    split_a = object_level_split(ids, seed=42, train_ratio=0.8)
    split_b = object_level_split(ids, seed=42, train_ratio=0.8)
    assert split_fingerprint(split_a) == split_fingerprint(split_b)


def test_object_level_split_has_no_leakage() -> None:
    ids = _object_ids(1000)
    split = object_level_split(ids, seed=3, train_ratio=0.75)
    assert_no_split_leakage(split)  # must not raise
    train_ids = set(split.loc[split["split"] == "train", "object_id"])
    test_ids = set(split.loc[split["split"] == "test", "object_id"])
    assert train_ids & test_ids == set()
    assert train_ids | test_ids == set(ids)


def test_assert_no_split_leakage_fails_loudly_on_injected_overlap() -> None:
    frame = pd.DataFrame({"object_id": ["a", "b"], "split": ["train", "train"]})
    frame = pd.concat(
        [frame, pd.DataFrame({"object_id": ["a"], "split": ["test"]})], ignore_index=True
    )
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_split_leakage(frame)


def test_split_fingerprint_is_row_order_invariant() -> None:
    ids = _object_ids(200)
    split = object_level_split(ids, seed=11, train_ratio=0.8)
    shuffled = split.sample(frac=1.0, random_state=1)
    assert split_fingerprint(split) == split_fingerprint(shuffled)


def test_split_fingerprint_changes_with_seed() -> None:
    ids = _object_ids(200)
    split_a = object_level_split(ids, seed=1, train_ratio=0.8)
    split_b = object_level_split(ids, seed=2, train_ratio=0.8)
    assert split_fingerprint(split_a) != split_fingerprint(split_b)


def test_object_level_split_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        object_level_split(["a", "a"], seed=1, train_ratio=0.5)


def test_object_level_split_rejects_bad_train_ratio() -> None:
    with pytest.raises(ValueError, match="train_ratio"):
        object_level_split(["a", "b"], seed=1, train_ratio=1.2)


def test_verify_required_columns_passes_when_all_present() -> None:
    columns = ["object_id", "spectrum", "Z", "ZWARN", "EBV"]
    verify_required_columns(columns, "object_id", "spectrum", "Z", "ZWARN")  # must not raise


def test_verify_required_columns_fails_clearly_when_columns_missing() -> None:
    columns = ["object_id", "EBV"]
    with pytest.raises(ValueError, match="missing"):
        verify_required_columns(columns, "object_id", "spectrum", "Z", "ZWARN")


def test_extract_labels_normalizes_and_validates() -> None:
    frame = pd.DataFrame(
        {
            "object_id": [1, 2, 3],
            "Z": [0.1, 0.2, 0.3],
            "ZWARN": [True, True, False],
        }
    )
    labels = extract_labels(frame, "object_id", "Z", "ZWARN")
    assert list(labels.columns) == ["object_id", "z", "zwarn"]
    assert labels["object_id"].tolist() == ["1", "2", "3"]
    assert labels["zwarn"].tolist() == [True, True, False]


def test_extract_labels_rejects_duplicate_object_id() -> None:
    frame = pd.DataFrame({"object_id": [1, 1], "Z": [0.1, 0.2], "ZWARN": [True, True]})
    with pytest.raises(ValueError, match="Duplicate object_id"):
        extract_labels(frame, "object_id", "Z", "ZWARN")


def test_extract_labels_rejects_missing_columns() -> None:
    frame = pd.DataFrame({"object_id": [1], "Z": [0.1]})
    with pytest.raises(ValueError, match="missing columns"):
        extract_labels(frame, "object_id", "Z", "ZWARN")


def _spectrum_row(object_id: str, wave, flux, ivar, mask):
    return {
        "object_id": object_id,
        "spectrum": {"flux": flux, "ivar": ivar, "lambda": wave, "mask": mask},
    }


def test_extract_spectrum_batch_builds_shared_grid_batch() -> None:
    wave = [3600.0, 3601.0, 3602.0]
    frame = pd.DataFrame(
        [
            _spectrum_row("a", wave, [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [False, False, True]),
            _spectrum_row("b", wave, [4.0, 5.0, 6.0], [2.0, 2.0, 2.0], [False, True, False]),
        ]
    )
    batch = extract_spectrum_batch(frame, "object_id", "spectrum", "flux", "lambda", "ivar", "mask")
    assert isinstance(batch, SpectrumBatch)
    assert batch.flux.shape == (2, 3)
    np.testing.assert_array_equal(batch.wave, np.asarray(wave, dtype=np.float32))
    assert batch.mask is not None and batch.mask.dtype == bool
    assert batch.mask[0].tolist() == [False, False, True]


def test_extract_spectrum_batch_rejects_divergent_wavelength_grids() -> None:
    frame = pd.DataFrame(
        [
            _spectrum_row("a", [3600.0, 3601.0], [1.0, 2.0], [1.0, 1.0], [False, False]),
            _spectrum_row("b", [3600.0, 3699.0], [1.0, 2.0], [1.0, 1.0], [False, False]),
        ]
    )
    with pytest.raises(ValueError, match="shared wavelength grid"):
        extract_spectrum_batch(frame, "object_id", "spectrum", "flux", "lambda", "ivar", "mask")


def test_extract_spectrum_batch_rejects_missing_columns() -> None:
    frame = pd.DataFrame({"object_id": ["a"]})
    with pytest.raises(ValueError, match="missing columns"):
        extract_spectrum_batch(frame, "object_id", "spectrum", "flux", "lambda", "ivar", "mask")


def test_zwarn_is_good_true_means_no_problem() -> None:
    """MultimodalUniverse/desi stores ZWARN as a bool already inverted from
    the raw DESI bitmask (True == raw ZWARN == 0, "no problem"); see
    scripts/desi/desi.py's `example["ZWARN"] = not bool(data["ZWARN"][i])`."""
    assert _zwarn_is_good(True, zwarn_filter_value=0) is True
    assert _zwarn_is_good(False, zwarn_filter_value=0) is False


def test_zwarn_is_good_rejects_nonzero_filter_value() -> None:
    with pytest.raises(ValueError, match="zwarn_filter_value must be 0"):
        _zwarn_is_good(True, zwarn_filter_value=1)
