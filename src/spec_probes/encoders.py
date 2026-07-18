"""Frozen spectrum-encoder adapters behind one shared interface.

Every encoder here is embed-only: none of them is trained or fine-tuned in
this package. AION-1 (frozen encoder representation) and AstroCLIP SpecFormer
wrap pretrained, frozen third-party weights; the PCA baseline fits a linear
basis on the train split only, which is the sole "training" any encoder here
performs.

Model-specific code stays in this module only (mirrors the aion_reimp
module-ownership convention in architecture.md): spectra_data.py, probes.py,
and run_probes.py never import a model package directly.

Loading paths verified 2026-07-18 (see per-class docstrings for exact
sources/commits):

- AION-1: https://huggingface.co/polymathic-ai/aion-base (model card,
  revision ``40541618104bab0fa85c8af68daeb867a720bb8c``) and
  https://github.com/PolymathicAI/AION (``aion/modalities.py``,
  ``aion/model.py``, ``aion/codecs/manager.py``, ``aion/codecs/spectrum.py``,
  README.md). Package name on PyPI is ``polymathic-aion`` (import name
  ``aion``).
- AstroCLIP SpecFormer: https://huggingface.co/polymathic-ai/specformer
  (file listing, revision ``160d67f0c07daf33d192568ca60ff38d76c39d66``) and
  https://github.com/PolymathicAI/AstroCLIP (MIT license, commit
  ``e129576a16bccd25a2794be21fab34d05c608661``). The architecture is vendored
  in ``spec_probes/specformer_model.py`` -- see that module's docstring.

The one remaining item that could not be verified without a live network
call or a data download is marked ``# VERIFY-ON-CLUSTER:``; each such marker
names the exact ``--preflight`` check (see
``scripts/run_phase6_probes_cluster.py``) that resolves it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Optional

import numpy as np
from scipy.ndimage import median_filter


@dataclass(frozen=True)
class SpectrumBatch:
    """One batch of spectra on a common wavelength grid, ready to embed."""

    object_id: np.ndarray  # (n,) str
    flux: np.ndarray  # (n, n_pix) float32, resampled to a common grid
    wave: np.ndarray  # (n_pix,) float32, the shared wavelength grid
    ivar: Optional[np.ndarray] = None  # (n, n_pix) float32, optional
    mask: Optional[np.ndarray] = None  # (n, n_pix) bool, optional; True == masked/bad pixel
    # (matches the verified MultimodalUniverse/desi `spectrum.mask` semantics:
    # spectra_data.py builds this from the streamed `spectrum` struct.)

    def __post_init__(self) -> None:
        object_id = np.asarray(self.object_id)
        flux = np.asarray(self.flux, dtype=np.float32)
        wave = np.asarray(self.wave, dtype=np.float32)
        if flux.ndim != 2:
            raise ValueError("SpectrumBatch.flux must be 2-D (n_objects, n_pix)")
        if wave.ndim != 1 or wave.shape[0] != flux.shape[1]:
            raise ValueError("SpectrumBatch.wave must be 1-D and match flux's pixel axis")
        if object_id.shape[0] != flux.shape[0]:
            raise ValueError("SpectrumBatch.object_id must have one entry per spectrum")
        if self.ivar is not None:
            ivar = np.asarray(self.ivar, dtype=np.float32)
            if ivar.shape != flux.shape:
                raise ValueError("SpectrumBatch.ivar must match flux's shape")
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.shape != flux.shape:
                raise ValueError("SpectrumBatch.mask must match flux's shape")

    def __len__(self) -> int:
        return int(np.asarray(self.object_id).shape[0])


class SpectrumEncoderAdapter(abc.ABC):
    """Shared interface every Phase 6 frozen spectrum-embedding encoder implements.

    ``fit`` is a no-op for the two pretrained encoders (AION-1, AstroCLIP
    SpecFormer); only the PCA baseline overrides it, since it must fit its
    basis on the train split only.
    """

    name: ClassVar[str]
    output_dim: int
    revision: str

    def fit(self, train_batch: SpectrumBatch) -> "SpectrumEncoderAdapter":
        return self

    @abc.abstractmethod
    def embed(self, batch: SpectrumBatch) -> np.ndarray:
        """Return (len(batch), self.output_dim) float32 embeddings, row-aligned
        to batch.object_id."""
        raise NotImplementedError


def resolve_device(declared_device: str) -> str:
    """Config-declared device with CPU fallback (Finding 3).

    ``declared_device`` is normally ``"cuda"`` or ``"cpu"`` from
    ``run.device`` in ``configs/phase6_probes.yaml``. If ``cuda`` is
    declared but unavailable (e.g. exercising this package's tests on a
    laptop with no GPU), fall back to CPU rather than raising -- the actual
    GPU requirement is enforced by the cluster preflight
    (``--preflight`` reports the resolved device and free GPU memory; the
    full run refuses to start without a passing preflight report).
    """
    if declared_device not in {"cuda", "cpu"}:
        raise ValueError(f"device must be 'cuda' or 'cpu', got {declared_device!r}")
    if declared_device == "cpu":
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _chunk_ranges(n: int, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


def resample_to_common_grid(flux: np.ndarray, wave: np.ndarray, common_grid: np.ndarray) -> np.ndarray:
    """Linear-interpolate each spectrum row onto one shared wavelength grid."""
    flux = np.asarray(flux, dtype=np.float64)
    wave = np.asarray(wave, dtype=np.float64)
    common_grid = np.asarray(common_grid, dtype=np.float64)
    if flux.ndim != 2 or wave.shape != (flux.shape[1],):
        raise ValueError("flux must be (n, n_pix) and wave must match its pixel axis")
    resampled = np.empty((flux.shape[0], common_grid.shape[0]), dtype=np.float32)
    for index in range(flux.shape[0]):
        resampled[index] = np.interp(common_grid, wave, flux[index]).astype(np.float32)
    return resampled


def continuum_normalize_log_flux(flux: np.ndarray, continuum_window_pixels: int) -> np.ndarray:
    """Per-spectrum continuum estimate via a sliding median, then log-flux.

    ``continuum_window_pixels`` must be odd, for a centered median-filter
    window (``scipy.ndimage.median_filter``, already a repo dependency).
    """
    flux = np.asarray(flux, dtype=np.float64)
    if continuum_window_pixels % 2 == 0:
        raise ValueError("continuum_window_pixels must be odd")
    continuum = median_filter(flux, size=(1, continuum_window_pixels), mode="nearest")
    continuum = np.where(continuum > 0, continuum, np.nan)
    normalized = flux / continuum
    normalized = np.nan_to_num(normalized, nan=1.0, posinf=1.0, neginf=1.0)
    floor = 1e-3
    return np.log(np.clip(normalized, floor, None)).astype(np.float32)


class PCASpectrumEncoder(SpectrumEncoderAdapter):
    """Classical baseline: PCA on resampled, continuum-normalized log-flux.

    Fit on the train split only. Anchors what "no pretrained encoder at all"
    buys, relative to the two frozen foundation/contrastive encoders below.
    CPU-only by design (scikit-learn PCA on a probe-scale sample is fast
    enough that a GPU path would be premature engineering).
    """

    name: ClassVar[str] = "pca_baseline"

    def __init__(
        self,
        n_components: int,
        resample_n_pixels: int,
        continuum_window_pixels: int = 51,
        seed: int = 0,
        name: str = "pca_baseline",
    ) -> None:
        from sklearn.decomposition import PCA

        self.name = str(name)
        self.output_dim = int(n_components)
        self.revision = "sklearn-pca-no-pretrained-revision"
        self._resample_n_pixels = int(resample_n_pixels)
        self._continuum_window_pixels = int(continuum_window_pixels)
        self._pca = PCA(n_components=n_components, random_state=seed)
        self._common_grid: Optional[np.ndarray] = None
        self._fitted = False

    def _preprocess(self, batch: SpectrumBatch) -> np.ndarray:
        if self._common_grid is None:
            low, high = float(batch.wave.min()), float(batch.wave.max())
            self._common_grid = np.linspace(low, high, self._resample_n_pixels)
        resampled = resample_to_common_grid(batch.flux, batch.wave, self._common_grid)
        return continuum_normalize_log_flux(resampled, self._continuum_window_pixels)

    def fit(self, train_batch: SpectrumBatch) -> "PCASpectrumEncoder":
        features = self._preprocess(train_batch)
        self._pca.fit(features)
        self._fitted = True
        return self

    def embed(self, batch: SpectrumBatch) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PCASpectrumEncoder.embed called before fit() (train split only)")
        features = self._preprocess(batch)
        return self._pca.transform(features).astype(np.float32)


class AionSpectrumCodecEncoder(SpectrumEncoderAdapter):
    """AION-1 frozen encoder, DESI spectrum modality.

    Verified 2026-07-18 against the model card at
    https://huggingface.co/polymathic-ai/aion-base (revision
    ``40541618104bab0fa85c8af68daeb867a720bb8c``) and against
    https://github.com/PolymathicAI/AION (pip package ``polymathic-aion``,
    import name ``aion``):

    - ``aion/modalities.py``: ``DESISpectrum(Spectrum)`` has fields
      ``flux``, ``ivar``, ``mask``, ``wavelength`` (all ``(batch, length)``
      tensors) and ``token_key = "tok_spectrum_desi"``.
    - ``aion/codecs/manager.py``: ``CodecManager(device=...).encode(modality)``
      dispatches on the modality's type and returns ``{token_key: tensor}``.
    - ``aion/codecs/spectrum.py`` (``AutoencoderSpectrumCodec._encode``):
      pads the input to ``DESISpectrum.pad_length`` (7808) internally via
      ``aion.codecs.preprocessing.spectrum.pad_spectrum`` when the modality
      has a ``pad_length`` attribute, then interpolates onto its own
      internal 8704-pixel latent grid using the supplied wavelength array --
      so this adapter passes the raw per-object DESI flux/ivar/mask/
      wavelength straight through with no manual resampling or padding.
    - ``aion/model.py`` (``AION.encode``): returns the *encoder* output
      (post ``forward_encoder`` + ``decoder_proj_context`` residual), shape
      ``(batch, num_encoder_tokens, 768)`` for aion-base -- this is the
      frozen AION-1 encoder representation the probe suite is meant to
      benchmark, not raw codec tokens.
    - AION's own README "Property Prediction" example mean-pools this
      exact tensor over the token axis (``embeddings.mean(axis=1)``, i.e.
      ``axis=1``/``dim=1``) before handing it to a downstream regressor;
      this adapter does the same.

    Reproducibility caveat (resolved by reading ``aion/codecs/manager.py``,
    not a placeholder): ``CodecManager._load_codec_from_hf`` calls
    ``codec_class.from_pretrained(HF_REPO_ID, modality=modality_type)``
    with no ``revision`` kwarg, so the spectrum codec always loads whatever
    subfolder revision ships with the *installed* ``aion`` package version,
    independent of the ``revision`` pinned here. Only the top-level
    ``AION.from_pretrained(repo_id, revision=...)`` call honors this
    adapter's pinned revision. Full reproducibility of the codec therefore
    rests on pinning ``polymathic-aion`` in ``pyproject.toml`` (done), not
    on this ``revision`` field alone.
    """

    name: ClassVar[str] = "aion_spectrumcodec"

    def __init__(
        self,
        repo_id: str,
        revision: str,
        output_dim: int = 768,
        device: str = "cpu",
        batch_size: int = 64,
        dtype: str = "float32",
        num_encoder_tokens: int = 600,
        name: str = "aion_spectrumcodec",
    ) -> None:
        self.name = str(name)
        self.output_dim = int(output_dim)
        self.revision = str(revision)
        self._repo_id = repo_id
        self._device = resolve_device(device)
        self._batch_size = int(batch_size)
        self._dtype = dtype
        self._num_encoder_tokens = int(num_encoder_tokens)
        self._model: Optional[Any] = None
        self._codec_manager: Optional[Any] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from aion import AION
        from aion.codecs import CodecManager

        model = AION.from_pretrained(self._repo_id, revision=self.revision)
        model = model.to(self._device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model
        self._codec_manager = CodecManager(device=self._device)

    def embed(self, batch: SpectrumBatch) -> np.ndarray:
        import torch

        from aion.modalities import DESISpectrum

        self._load()
        torch_dtype = getattr(torch, self._dtype)
        chunks = []
        with torch.no_grad():
            for start, end in _chunk_ranges(len(batch), self._batch_size):
                flux = torch.as_tensor(batch.flux[start:end], dtype=torch_dtype, device=self._device)
                wave_row = torch.as_tensor(batch.wave, dtype=torch_dtype, device=self._device)
                wavelength = wave_row.unsqueeze(0).expand(flux.shape[0], -1)
                if batch.ivar is not None:
                    ivar = torch.as_tensor(batch.ivar[start:end], dtype=torch_dtype, device=self._device)
                else:
                    ivar = torch.ones_like(flux)
                if batch.mask is not None:
                    mask = torch.as_tensor(batch.mask[start:end], dtype=torch.bool, device=self._device)
                else:
                    mask = torch.zeros_like(flux, dtype=torch.bool)

                spectrum = DESISpectrum(flux=flux, ivar=ivar, mask=mask, wavelength=wavelength)
                tokens = self._codec_manager.encode(spectrum)
                # VERIFY-ON-CLUSTER: confirm `model.encode(tokens,
                # num_encoder_tokens=...)` accepts this single-modality
                # tokens dict directly (matches AION's README usage) and
                # returns a (batch, num_encoder_tokens, 768) tensor for
                # aion-base with a single spectrum modality present.
                # Resolved by the `--preflight` "encode one example per
                # encoder, report output shape" check.
                encoded = self._model.encode(tokens, num_encoder_tokens=self._num_encoder_tokens)
                pooled = encoded.mean(dim=1)
                chunks.append(pooled.detach().to(torch.float32).cpu().numpy())
        embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
        if embeddings.shape[1] != self.output_dim:
            raise ValueError(
                f"AION-1 encoder produced dim={embeddings.shape[1]}, expected output_dim={self.output_dim}"
            )
        return embeddings


class AstroCLIPSpecFormerEncoder(SpectrumEncoderAdapter):
    """AstroCLIP SpecFormer, loaded from its Lightning checkpoint.

    Verified 2026-07-18: https://huggingface.co/polymathic-ai/specformer
    (revision ``160d67f0c07daf33d192568ca60ff38d76c39d66``) hosts a single
    file, ``specformer.ckpt`` (~173 MB, a ``lightning``-format checkpoint
    dict with ``hyper_parameters`` and ``state_dict`` keys) -- there is no
    ``config.json`` there, so ``transformers.AutoModel.from_pretrained``
    cannot load it (the original guess in this adapter was wrong). Its own
    HF README: "Specformer model pretrained on the cross-matched DESI
    legacy survey for the AstroCLIP alignment task."

    Loading route, following the reference recipe in
    ``downstream_tasks/property_estimation/embed_provabgs.py`` of
    https://github.com/PolymathicAI/AstroCLIP (MIT, commit
    ``e129576a16bccd25a2794be21fab34d05c608661``):
    ``huggingface_hub.hf_hub_download`` the checkpoint file, ``torch.load``
    it, construct ``spec_probes.specformer_model.SpecFormer`` from the
    checkpoint's own ``hyper_parameters`` (so this adapter never hardcodes
    an architecture that could drift from the pinned checkpoint), then
    ``load_state_dict``. The vendored ``SpecFormer.forward`` returns
    ``{"embedding": ...}``, shape ``(batch, num_sections, 768)``, mean-pooled
    over the section axis exactly like the reference recipe
    (``np.mean(specformer(x)["embedding"].cpu().numpy(), axis=1)``).

    Chosen over depending on the ``astroclip`` package itself because
    ``astroclip`` is not published to PyPI (only installable from its git
    repo) and its top-level ``astroclip.models`` package eagerly imports
    ``Moco_v2``/``astrodino`` machinery (pulling in ``dinov2`` and other
    image-model dependencies unrelated to this one encoder); vendoring the
    ~250-line SpecFormer architecture keeps this package's dependency
    footprint to ``torch`` + ``huggingface_hub``, which is more reliable to
    install on a shared cluster environment than an unpublished git
    dependency plus its full transitive image-model stack.
    """

    name: ClassVar[str] = "astroclip_specformer"

    def __init__(
        self,
        repo_id: str,
        revision: str,
        output_dim: int = 768,
        device: str = "cpu",
        batch_size: int = 64,
        dtype: str = "float32",
        name: str = "astroclip_specformer",
    ) -> None:
        self.name = str(name)
        self.output_dim = int(output_dim)
        self.revision = str(revision)
        self._repo_id = repo_id
        self._device = resolve_device(device)
        self._batch_size = int(batch_size)
        self._dtype = dtype
        self._model: Optional[Any] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from huggingface_hub import hf_hub_download

        from .specformer_model import SpecFormer

        checkpoint_path = hf_hub_download(
            repo_id=self._repo_id, filename="specformer.ckpt", revision=self.revision
        )
        # weights_only=False: the checkpoint's `hyper_parameters` entry is a
        # plain dict (originally a lightning AttributeDict), not just
        # tensors; this is the pinned, single-file, MIT-licensed checkpoint
        # verified above, not an arbitrary untrusted download.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hyper_parameters = dict(checkpoint["hyper_parameters"])
        model = SpecFormer(**hyper_parameters)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(self._device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model

    def embed(self, batch: SpectrumBatch) -> np.ndarray:
        import torch

        self._load()
        torch_dtype = getattr(torch, self._dtype)
        chunks = []
        with torch.no_grad():
            for start, end in _chunk_ranges(len(batch), self._batch_size):
                # SpecFormer.forward expects (batch, length, 1): a single
                # flux channel, no wavelength/ivar input (see
                # specformer_model.py -- preprocess() normalizes internally
                # and slices into overlapping chunks along the length axis).
                # VERIFY-ON-CLUSTER: confirm the MMU DESI flux array (length
                # 7781, native DESI coadd grid) is close enough to the
                # wavelength grid/resolution SpecFormer was pretrained on
                # (AstroCLIP's own DESI resampling) for the checkpoint's
                # position embeddings to be meaningful; SpecFormer has no
                # wavelength input to make this correction itself. Resolved
                # by the `--preflight` "encode one example per encoder,
                # report output shape" check plus a manual read of the
                # reported shape against max_len=800 sections.
                flux = torch.as_tensor(
                    batch.flux[start:end], dtype=torch_dtype, device=self._device
                ).unsqueeze(-1)
                output = self._model(flux)
                pooled = output["embedding"].mean(dim=1)
                chunks.append(pooled.detach().to(torch.float32).cpu().numpy())
        embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
        if embeddings.shape[1] != self.output_dim:
            raise ValueError(
                f"AstroCLIP SpecFormer produced dim={embeddings.shape[1]}, "
                f"expected output_dim={self.output_dim}"
            )
        return embeddings


def build_encoder(spec: Mapping[str, Any], device: str = "cpu") -> SpectrumEncoderAdapter:
    """Dispatch one resolved ``encoders[i]`` config entry to its adapter class.

    ``device`` comes from the top-level ``run.device`` config key (Finding
    3); it is resolved once (with CPU fallback) per adapter instance rather
    than at import time, so constructing an adapter never touches
    ``torch.cuda`` unless the adapter actually needs a device.
    """
    kind = spec["kind"]
    if kind == "pca_baseline":
        return PCASpectrumEncoder(
            n_components=spec["n_components"],
            resample_n_pixels=spec["resample_n_pixels"],
            name=spec["name"],
        )
    if kind == "aion_spectrumcodec":
        return AionSpectrumCodecEncoder(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            output_dim=spec["output_dim"],
            device=device,
            batch_size=spec["batch_size"],
            dtype=spec["dtype"],
            num_encoder_tokens=spec["num_encoder_tokens"],
            name=spec["name"],
        )
    if kind == "astroclip_specformer":
        return AstroCLIPSpecFormerEncoder(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            output_dim=spec["output_dim"],
            device=device,
            batch_size=spec["batch_size"],
            dtype=spec["dtype"],
            name=spec["name"],
        )
    raise ValueError(f"Unknown encoder kind: {kind!r}")
