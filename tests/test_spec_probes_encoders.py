from typing import ClassVar

import numpy as np
import pytest

from spec_probes.encoders import (
    AionSpectrumCodecEncoder,
    AstroCLIPSpecFormerEncoder,
    PCASpectrumEncoder,
    SpectrumBatch,
    SpectrumEncoderAdapter,
    continuum_normalize_log_flux,
    resample_to_common_grid,
    resolve_device,
)


class FakeEncoder(SpectrumEncoderAdapter):
    """Minimal concrete adapter used only to exercise the shared interface."""

    name: ClassVar[str] = "fake"

    def __init__(self, output_dim: int = 6, seed: int = 0) -> None:
        self.output_dim = output_dim
        self.revision = "fake-rev-0001"
        self._rng = np.random.default_rng(seed)

    def embed(self, batch: SpectrumBatch) -> np.ndarray:
        return self._rng.normal(size=(len(batch), self.output_dim)).astype(np.float32)


def _batch(n=8, n_pix=20, seed=0):
    rng = np.random.default_rng(seed)
    object_id = np.array([f"obj-{i}" for i in range(n)])
    wave = np.linspace(3600.0, 9800.0, n_pix)
    flux = 1.0 + 0.05 * rng.normal(size=(n, n_pix))
    return SpectrumBatch(object_id=object_id, flux=flux, wave=wave)


def test_spectrum_batch_rejects_mismatched_wave_length() -> None:
    with pytest.raises(ValueError, match="wave"):
        SpectrumBatch(object_id=np.array(["a"]), flux=np.zeros((1, 5)), wave=np.zeros(4))


def test_spectrum_batch_rejects_object_id_length_mismatch() -> None:
    with pytest.raises(ValueError, match="object_id"):
        SpectrumBatch(object_id=np.array(["a", "b"]), flux=np.zeros((1, 5)), wave=np.zeros(5))


def test_spectrum_batch_rejects_mismatched_ivar_shape() -> None:
    with pytest.raises(ValueError, match="ivar"):
        SpectrumBatch(
            object_id=np.array(["a"]), flux=np.zeros((1, 5)), wave=np.zeros(5), ivar=np.zeros((1, 4))
        )


def test_spectrum_batch_len_matches_object_count() -> None:
    batch = _batch(n=5)
    assert len(batch) == 5


def test_fake_encoder_satisfies_the_shared_interface() -> None:
    encoder = FakeEncoder(output_dim=4)
    batch = _batch(n=6)
    fitted = encoder.fit(batch)  # no-op for frozen encoders, must return self
    assert fitted is encoder
    embeddings = encoder.embed(batch)
    assert embeddings.shape == (6, 4)
    assert embeddings.dtype == np.float32


def test_resample_to_common_grid_interpolates_linearly() -> None:
    flux = np.array([[0.0, 10.0]])
    wave = np.array([0.0, 1.0])
    common_grid = np.array([0.0, 0.5, 1.0])
    resampled = resample_to_common_grid(flux, wave, common_grid)
    assert resampled[0].tolist() == pytest.approx([0.0, 5.0, 10.0])


def test_continuum_normalize_log_flux_rejects_even_window() -> None:
    with pytest.raises(ValueError, match="odd"):
        continuum_normalize_log_flux(np.ones((1, 10)), continuum_window_pixels=10)


def test_continuum_normalize_flat_spectrum_is_near_zero_log_flux() -> None:
    flux = np.full((1, 21), 5.0)
    normalized = continuum_normalize_log_flux(flux, continuum_window_pixels=5)
    assert normalized == pytest.approx(np.zeros_like(normalized), abs=1e-5)


def test_pca_encoder_requires_fit_before_embed() -> None:
    encoder = PCASpectrumEncoder(n_components=3, resample_n_pixels=16)
    batch = _batch(n=10, n_pix=20)
    with pytest.raises(RuntimeError, match="before fit"):
        encoder.embed(batch)


def test_pca_encoder_fits_on_train_and_embeds_both_splits() -> None:
    train_batch = _batch(n=12, n_pix=24, seed=1)
    test_batch = _batch(n=5, n_pix=24, seed=2)
    encoder = PCASpectrumEncoder(n_components=3, resample_n_pixels=16)
    encoder.fit(train_batch)
    train_embeddings = encoder.embed(train_batch)
    test_embeddings = encoder.embed(test_batch)
    assert train_embeddings.shape == (12, 3)
    assert test_embeddings.shape == (5, 3)
    assert encoder.output_dim == 3


def test_pca_encoder_is_deterministic_given_fixed_seed() -> None:
    train_batch = _batch(n=12, n_pix=24, seed=1)
    test_batch = _batch(n=5, n_pix=24, seed=2)
    first = PCASpectrumEncoder(n_components=3, resample_n_pixels=16, seed=0)
    first.fit(train_batch)
    second = PCASpectrumEncoder(n_components=3, resample_n_pixels=16, seed=0)
    second.fit(train_batch)
    np.testing.assert_array_equal(first.embed(test_batch), second.embed(test_batch))


def test_spectrum_batch_accepts_optional_mask() -> None:
    batch = SpectrumBatch(
        object_id=np.array(["a", "b"]),
        flux=np.zeros((2, 4)),
        wave=np.zeros(4),
        mask=np.array([[False, False, True, False], [False, False, False, True]]),
    )
    assert batch.mask.dtype == bool


def test_spectrum_batch_rejects_mismatched_mask_shape() -> None:
    with pytest.raises(ValueError, match="mask"):
        SpectrumBatch(
            object_id=np.array(["a"]), flux=np.zeros((1, 5)), wave=np.zeros(5), mask=np.zeros((1, 4), dtype=bool)
        )


def test_resolve_device_cpu_stays_cpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_rejects_unknown_device() -> None:
    with pytest.raises(ValueError, match="device must be"):
        resolve_device("tpu")


def test_aion_encoder_is_constructible_without_network() -> None:
    """Finding 1/3: construction must never import `aion` or touch a
    network -- only `.embed()` (via lazy `_load()`) does, so the fake-
    adapter tests elsewhere in this suite keep working without the
    optional cluster dependency installed."""
    encoder = AionSpectrumCodecEncoder(
        repo_id="polymathic-ai/aion-base",
        revision="40541618104bab0fa85c8af68daeb867a720bb8c",
        output_dim=768,
        device="cpu",
        batch_size=8,
        dtype="float32",
        num_encoder_tokens=600,
    )
    assert encoder.output_dim == 768
    assert encoder.name == "aion_spectrumcodec"


def test_specformer_encoder_is_constructible_without_network() -> None:
    encoder = AstroCLIPSpecFormerEncoder(
        repo_id="polymathic-ai/specformer",
        revision="160d67f0c07daf33d192568ca60ff38d76c39d66",
        output_dim=768,
        device="cpu",
        batch_size=8,
        dtype="float32",
    )
    assert encoder.output_dim == 768
    assert encoder.name == "astroclip_specformer"
