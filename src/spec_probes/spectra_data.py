"""Phase 6 probe-scale spectrum sample selection, label extraction, and the
shared object-level train/test split used by every encoder and probe family.

Split derivation reuses ``aion_reimp.manifest.split_fraction`` (the same
SHA-256-based deterministic per-object split Phase 2/3 manifests use) and
``aion_reimp.manifest.manifest_fingerprint`` (the same row-order-invariant
content fingerprint), rather than re-deriving equivalent logic here -- both
are generic, dataset-agnostic helpers with no retrieval/training coupling.

Schema verified 2026-07-18 against
https://huggingface.co/datasets/MultimodalUniverse/desi (revision
``933b9056b93b0f3e7790ee28c43412b49c39e232``) and the dataset's own builder
script, https://github.com/MultimodalUniverse/MultimodalUniverse
``scripts/desi/desi.py`` / ``scripts/desi/build_parent_sample.py`` (Apache-2.0):

- Top-level columns: a nested ``spectrum`` struct (``flux``, ``ivar``,
  ``lambda``, ``mask``, ``lsf_sigma``, each length 7781) plus ``Z``, ``ZERR``,
  ``EBV``, ``FLUX_*``/``FIBER*FLUX_*`` photometry, ``ZWARN`` (bool), and
  ``object_id`` (string). There is no ``SPECTYPE`` column and no
  ``TARGETID`` column.
- ``object_id`` *is* the DESI ``TARGETID`` (cast to string): the builder
  script sets ``catalog["object_id"] = catalog["TARGETID"]`` before export
  (``build_parent_sample.py`` line 151). A DESI VAC crossmatch for
  ``SPECTYPE`` would therefore join on this dataset's own ``object_id``
  directly -- no ID-encoding scheme to reverse-engineer -- but the join
  itself (against an external, multi-gigabyte DESI redshift catalog file)
  is not implemented here; it is a separate, not-yet-scoped deliverable
  (see ``labels.spectral_class`` in ``configs/phase6_probes.yaml``).
- ``ZWARN`` is stored as a **boolean**, and it is already inverted from the
  raw DESI bitmask: ``desi.py``'s ``_generate_examples`` does
  ``example["ZWARN"] = not bool(data["ZWARN"][i])`` ("if flag is 0, then no
  problem"). So the stored value is ``True`` exactly when the raw DESI
  ``ZWARN == 0`` (good spectrum), and the original bitmask is not
  recoverable -- ``labels.zwarn_filter_value`` can only ever mean "0" (the
  ``ZWARN == 0`` cut this package always applies), which is why
  ``config.py`` rejects any other value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from aion_reimp.manifest import manifest_fingerprint, split_fraction

if TYPE_CHECKING:
    from .encoders import SpectrumBatch

CANONICAL_LABEL_COLUMNS = ("object_id", "z", "zwarn")


def inspect_source_columns(repo_id: str, revision: str, split: str) -> Dict[str, Any]:
    """Network-only: report the columns available in one streamed row.

    This is deliberately the first action of the Phase 6 prep pipeline: it
    peeks at exactly one streamed row -- never the full sample -- so a
    schema mismatch is discovered before anything else runs. Never called
    locally; it needs network access to Hugging Face and is exercised only
    by ``scripts/run_phase6_probes_cluster.py --preflight`` after approval.
    """
    from datasets import load_dataset

    dataset = load_dataset(repo_id, revision=revision, split=split, streaming=True)
    first_row = next(iter(dataset))
    return {
        "repo_id": repo_id,
        "revision": revision,
        "split": split,
        "columns": sorted(first_row.keys()),
        "spectrum_fields": sorted(first_row["spectrum"].keys()) if isinstance(first_row.get("spectrum"), dict) else None,
        "zwarn_dtype": type(first_row.get("ZWARN")).__name__,
    }


def verify_required_columns(
    available_columns: Sequence[str],
    object_id_column: str,
    spectrum_column: str,
    redshift_column: str,
    zwarn_column: str,
) -> None:
    """Confirm the config's declared column names are actually present.

    Fails with a clear, actionable message rather than guessing. Unlike an
    earlier version of this module, there is no inline-vs-join label-source
    branch to resolve here: SPECTYPE is not a column this package's config
    can declare at all (see module docstring), so the only thing to verify
    is that the columns this package *does* use are really there.
    """
    columns = set(available_columns)
    required = {object_id_column, spectrum_column, redshift_column, zwarn_column}
    missing = required - columns
    if missing:
        raise ValueError(
            f"Streamed columns are missing {sorted(missing)}. Available columns: {sorted(columns)}. "
            "The column names in configs/phase6_probes.yaml do not match the live schema."
        )


def _zwarn_is_good(raw_value: Any, zwarn_filter_value: int) -> bool:
    """Translate the config's ``zwarn_filter_value`` into the dataset's stored bool.

    See the module docstring: MMU DESI's ``ZWARN`` field is already a
    boolean "no problem" flag (``True`` iff the raw DESI ``ZWARN == 0``),
    not the raw bitmask. ``config.py`` only accepts
    ``zwarn_filter_value == 0``, so "keep" always means the stored boolean
    is ``True``.
    """
    if zwarn_filter_value != 0:
        raise ValueError(
            "zwarn_filter_value must be 0: MultimodalUniverse/desi's ZWARN field is a boolean "
            "'no problem' flag with the original bitmask discarded, so no other value is meaningful."
        )
    return bool(raw_value) is True


def select_probe_sample(
    repo_id: str,
    revision: str,
    split: str,
    sample_size: int,
    seed: int,
    zwarn_column: str,
    zwarn_filter_value: int = 0,
) -> pd.DataFrame:
    """Network-only: streaming-select a probe-scale DESI sample.

    Never called locally. Filters to ``ZWARN`` "no problem" rows (see
    ``_zwarn_is_good``), shuffles the stream deterministically by seed, and
    takes the first ``sample_size`` rows -- no bulk download of the full
    dataset.
    """
    # VERIFY-ON-CLUSTER: confirm HF streaming `.shuffle(seed=...,
    # buffer_size=...)` followed by manual `.take`-equivalent iteration
    # gives a reproducible sample given this dataset's shard layout.
    # Resolved by `--preflight`, which creates two fresh streams with the
    # same pinned revision/seed and requires their ordered object-ID lists
    # to match before writing a passing report.
    from datasets import load_dataset

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    dataset = load_dataset(repo_id, revision=revision, split=split, streaming=True)
    filtered = dataset.filter(lambda row: _zwarn_is_good(row[zwarn_column], zwarn_filter_value))
    shuffled = filtered.shuffle(seed=seed, buffer_size=max(sample_size * 10, 10_000))
    rows: List[Dict[str, Any]] = []
    for row in shuffled:
        rows.append(row)
        if len(rows) >= sample_size:
            break
    if len(rows) < sample_size:
        raise ValueError(
            f"Only {len(rows)} rows survived the ZWARN 'no problem' filter, "
            f"short of the requested sample_size={sample_size}"
        )
    return pd.DataFrame(rows)


def extract_labels(
    frame: pd.DataFrame,
    object_id_column: str,
    redshift_column: str,
    zwarn_column: str,
) -> pd.DataFrame:
    """Normalize raw source columns to the canonical label schema.

    Only spec-z (``Z``) and ``ZWARN`` are extracted: there is no SPECTYPE
    column in this dataset (see module docstring), so this package's only
    physical-recovery target is redshift unless/until a separate DESI VAC
    crossmatch deliverable adds a spectral-class label source.
    """
    required = {object_id_column, redshift_column, zwarn_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Label extraction source missing columns: {sorted(missing)}")
    labels = frame.loc[:, [object_id_column, redshift_column, zwarn_column]].copy()
    labels.columns = list(CANONICAL_LABEL_COLUMNS)
    labels["object_id"] = labels["object_id"].astype(str)
    labels["z"] = labels["z"].astype(np.float64)
    labels["zwarn"] = labels["zwarn"].astype(bool)
    if labels["object_id"].duplicated().any():
        duplicate = labels.loc[labels["object_id"].duplicated(), "object_id"].iloc[0]
        raise ValueError(f"Duplicate object_id in extracted labels: {duplicate}")
    return labels.sort_values("object_id").reset_index(drop=True)


def extract_spectrum_batch(
    frame: pd.DataFrame,
    object_id_column: str,
    spectrum_column: str,
    flux_field: str,
    wave_field: str,
    ivar_field: str,
    mask_field: str,
) -> "SpectrumBatch":
    """Build a ``spec_probes.encoders.SpectrumBatch`` from streamed rows.

    ``spectrum_column`` holds a nested struct per row (verified schema:
    ``{"flux": [...], "ivar": [...], "lambda": [...], "mask": [...],
    "lsf_sigma": [...]}``, each a fixed length-7781 array on DESI's shared
    coadd wavelength grid). Imports ``SpectrumBatch`` locally to avoid a
    module-level import cycle (``encoders.py`` does not import this
    module); this is data shaping, not encoder-specific code, so it stays
    consistent with this module's "probe sample selection" ownership in
    architecture.md.
    """
    from .encoders import SpectrumBatch

    required = {object_id_column, spectrum_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Spectrum batch source missing columns: {sorted(missing)}")

    object_id = frame[object_id_column].astype(str).to_numpy()

    def _field(field_name: str) -> np.ndarray:
        # VERIFY-ON-CLUSTER: confirm each streamed row's `spectrum` value is
        # a dict of fixed-length arrays (one per subfield), as the HF
        # `datasets` library materializes a `Sequence(feature=struct)`
        # column -- rather than a list of one dict per pixel. Resolved by
        # the `--preflight` "report nested structure" check, which reports
        # the literal type of one streamed row's `spectrum` value before
        # any full sample is pulled.
        return np.stack([np.asarray(row[field_name], dtype=np.float32) for row in frame[spectrum_column]])

    flux = _field(flux_field)
    wave_rows = _field(wave_field)
    ivar = _field(ivar_field)
    mask_rows = np.stack([np.asarray(row[mask_field], dtype=bool) for row in frame[spectrum_column]])

    if not np.allclose(wave_rows, wave_rows[0]):
        raise ValueError(
            "extract_spectrum_batch expected one shared wavelength grid across the sample "
            "(DESI's coadd grid is instrument-fixed); got per-row variation instead"
        )
    wave = wave_rows[0]

    return SpectrumBatch(object_id=object_id, flux=flux, wave=wave, ivar=ivar, mask=mask_rows)


def object_level_split(object_ids: Sequence[str], seed: int, train_ratio: float) -> pd.DataFrame:
    """One seeded, deterministic, leakage-free train/test split by object_id.

    Reuses ``aion_reimp.manifest.split_fraction`` so this probe split is
    derived by the same trusted SHA-256 mechanism Phase 2/3 manifests use,
    rather than a second implementation of the same idea.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between zero and one")
    ids = [str(value) for value in object_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("object_level_split requires unique object IDs")
    frame = pd.DataFrame({"object_id": ids})
    frame["_fraction"] = frame["object_id"].map(lambda value: split_fraction(value, seed))
    frame["split"] = np.where(frame["_fraction"] < train_ratio, "train", "test")
    frame = frame.drop(columns="_fraction")
    return frame.sort_values("object_id").reset_index(drop=True)


def split_fingerprint(split_frame: pd.DataFrame) -> str:
    """Row-order-invariant fingerprint of a train/test split, via aion_reimp.manifest."""
    required = {"object_id", "split"}
    missing = required - set(split_frame.columns)
    if missing:
        raise ValueError(f"Split frame missing columns: {sorted(missing)}")
    return manifest_fingerprint(split_frame.loc[:, ["object_id", "split"]])


def assert_no_split_leakage(split_frame: pd.DataFrame) -> None:
    """Object-level guard: the train and test ID sets must never intersect."""
    train_ids = set(split_frame.loc[split_frame["split"] == "train", "object_id"])
    test_ids = set(split_frame.loc[split_frame["split"] == "test", "object_id"])
    overlap = train_ids & test_ids
    if overlap:
        raise AssertionError(f"Object-level split leakage detected for {len(overlap)} objects")
