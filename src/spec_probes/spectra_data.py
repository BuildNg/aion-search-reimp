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

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


def assign_hats_partitions(
    targets: pd.DataFrame,
    partition_info: pd.DataFrame,
    *,
    ra_column: str,
    dec_column: str,
) -> pd.DataFrame:
    """Map coordinates to the adaptive HATS leaf partition containing them."""
    from hats.pixel_math import compute_spatial_index, spatial_index_to_healpix

    if not {ra_column, dec_column}.issubset(targets):
        raise ValueError("HATS target coordinates are missing")
    if not {"Norder", "Npix"}.issubset(partition_info):
        raise ValueError("HATS partition_info.csv is missing Norder/Npix")
    spatial = compute_spatial_index(
        targets[ra_column].to_numpy(dtype=float),
        targets[dec_column].to_numpy(dtype=float),
    )
    assigned_order = np.full(len(targets), -1, dtype=np.int16)
    assigned_pixel = np.full(len(targets), -1, dtype=np.int64)
    for order in sorted(partition_info["Norder"].unique(), reverse=True):
        unresolved = np.flatnonzero(assigned_order < 0)
        if not len(unresolved):
            break
        pixels = spatial_index_to_healpix(spatial[unresolved].tolist(), target_order=int(order))
        valid = set(
            partition_info.loc[partition_info["Norder"].eq(order), "Npix"].astype(int)
        )
        matches = np.asarray([int(pixel) in valid for pixel in pixels], dtype=bool)
        assigned_order[unresolved[matches]] = int(order)
        assigned_pixel[unresolved[matches]] = np.asarray(pixels, dtype=np.int64)[matches]
    if np.any(assigned_order < 0):
        raise ValueError(f"{int(np.sum(assigned_order < 0))} targets fall outside the HATS catalog")
    result = targets.copy()
    result["hats_order"] = assigned_order
    result["hats_pixel"] = assigned_pixel
    return result


def hats_partition_path(catalog_path: str, order: int, pixel: int) -> str:
    directory = (int(pixel) // 10_000) * 10_000
    return f"{catalog_path}/dataset/Norder={int(order)}/Dir={directory}/Npix={int(pixel)}.parquet"


def validate_spectrum_struct(value: Any, required_fields: Sequence[str]) -> Dict[str, int]:
    """Validate one materialized HATS spectrum before an expensive preparation continues."""
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected spectrum struct mapping, got {type(value).__name__}")
    missing = set(required_fields) - set(value)
    if missing:
        raise ValueError(f"Spectrum struct missing fields: {sorted(missing)}")
    lengths = {field: int(np.asarray(value[field]).size) for field in required_fields}
    if any(length <= 0 for length in lengths.values()) or len(set(lengths.values())) != 1:
        raise ValueError(f"Spectrum fields do not share one non-empty length: {lengths}")
    return lengths


def load_hats_target_spectra(
    repo_id: str,
    revision: str,
    catalog_path: str,
    targets: pd.DataFrame,
    cache_dir: Path,
    *,
    target_id_column: str,
    target_ra_column: str,
    target_dec_column: str,
    object_id_column: str,
    spectrum_column: str,
    redshift_column: str,
    zwarn_column: str,
    spectrum_fields: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Read only target-bearing HATS partitions, checkpointing each partition.

    A completed partition is an atomic Parquet part containing only requested
    objects. A retry validates and reuses those parts, so it never restarts a
    multi-partition download from zero.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    repo_root = f"datasets/{repo_id}@{revision}"
    with fs.open(f"{repo_root}/{catalog_path}/partition_info.csv", "rb") as handle:
        partition_info = pd.read_csv(handle)
    assigned = assign_hats_partitions(
        targets,
        partition_info,
        ra_column=target_ra_column,
        dec_column=target_dec_column,
    )
    expected_all = set(assigned[target_id_column].astype(str))
    rows: List[pd.DataFrame] = []
    partition_bytes_touched = 0
    reused_parts = 0
    for (order, pixel), group in assigned.groupby(["hats_order", "hats_pixel"], sort=True):
        expected_ids = set(group[target_id_column].astype(str))
        part_path = cache_dir / f"order{int(order)}_pixel{int(pixel)}.parquet"
        if part_path.exists():
            selected = pd.read_parquet(part_path)
            reused_parts += 1
        else:
            relative = hats_partition_path(catalog_path, int(order), int(pixel))
            remote_path = f"{repo_root}/{relative}"
            partition_bytes_touched += int(fs.info(remote_path)["size"])
            with fs.open(remote_path, "rb") as handle:
                table = pq.ParquetFile(handle).read(
                    columns=[
                        object_id_column,
                        *[f"{spectrum_column}.{field}" for field in spectrum_fields],
                        redshift_column,
                        zwarn_column,
                    ]
                )
            partition = table.to_pandas()
            partition[object_id_column] = partition[object_id_column].astype(str)
            selected = partition.loc[partition[object_id_column].isin(expected_ids)].copy()
            temporary = part_path.with_suffix(".tmp.parquet")
            selected.to_parquet(temporary, index=False)
            temporary.replace(part_path)
        selected[object_id_column] = selected[object_id_column].astype(str)
        found_ids = set(selected[object_id_column])
        if found_ids != expected_ids or selected[object_id_column].duplicated().any():
            raise ValueError(
                f"HATS partition ({order}, {pixel}) returned {len(found_ids)} unique targets; "
                f"expected {len(expected_ids)}"
            )
        validate_spectrum_struct(selected.iloc[0][spectrum_column], spectrum_fields)
        rows.append(selected)
        print(
            f"Prepared HATS partition ({int(order)}, {int(pixel)}): "
            f"{len(expected_ids)} targets ({len(rows)}/{assigned[['hats_order', 'hats_pixel']].drop_duplicates().shape[0]})",
            flush=True,
        )
    spectra = pd.concat(rows, ignore_index=True)
    if set(spectra[object_id_column].astype(str)) != expected_all:
        raise ValueError("Prepared HATS parts do not cover the exact requested target IDs")
    return (
        spectra.sort_values(object_id_column).reset_index(drop=True),
        assigned,
        {
            "partitions": int(assigned[["hats_order", "hats_pixel"]].drop_duplicates().shape[0]),
            "partition_bytes_touched_this_call": int(partition_bytes_touched),
            "reused_parts": int(reused_parts),
        },
    )


def find_stream_target_ids(
    repo_id: str,
    revision: str,
    split: str,
    candidate_ids: Sequence[str],
    object_id_column: str,
) -> List[str]:
    """Network-only: return the first candidate ID present in a streamed table.

    Scans the object-ID column alone (parquet column pruning keeps this
    cheap), and stops at the first match because preflight needs only one
    object shared between the HATS conversion and the probe-validated
    MultimodalUniverse/desi table.
    """
    from datasets import load_dataset

    candidates = {str(value) for value in candidate_ids}
    if not candidates:
        raise ValueError("candidate_ids must not be empty")
    dataset = load_dataset(repo_id, revision=revision, split=split, streaming=True)
    dataset = dataset.select_columns([object_id_column])
    found = set()
    for scanned, row in enumerate(dataset, start=1):
        object_id = str(row[object_id_column])
        if object_id in candidates:
            found.add(object_id)
            break
        if scanned % 100_000 == 0:
            print(f"Scanned {scanned:,} rows for binding overlap; found {len(found):,}", flush=True)
    return sorted(found)


def select_target_spectra(
    repo_id: str,
    revision: str,
    split: str,
    target_ids: Sequence[str],
    *,
    object_id_column: str,
    spectrum_column: str,
    redshift_column: str,
    zwarn_column: str,
) -> pd.DataFrame:
    """Fetch exact target rows from a projected streamed spectrum table."""
    from datasets import load_dataset

    expected = {str(value) for value in target_ids}
    if not expected:
        raise ValueError("target_ids must not be empty")
    columns = [object_id_column, spectrum_column, redshift_column, zwarn_column]
    dataset = load_dataset(repo_id, revision=revision, split=split, streaming=True)
    dataset = dataset.select_columns(columns)
    rows = []
    found = set()
    for row in dataset:
        object_id = str(row[object_id_column])
        if object_id in expected:
            if object_id in found:
                raise ValueError(f"Duplicate streamed target ID: {object_id}")
            rows.append(row)
            found.add(object_id)
            if found == expected:
                break
    if found != expected:
        missing = sorted(expected - found)
        raise ValueError(f"Streamed spectrum table is missing target IDs: {missing}")
    frame = pd.DataFrame(rows)
    frame[object_id_column] = frame[object_id_column].astype(str)
    return frame.sort_values(object_id_column).reset_index(drop=True)


def assert_spectrum_value_binding(
    batch: "SpectrumBatch",
    reference: "SpectrumBatch",
    object_id: str,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> Dict[str, float]:
    """Require one object's spectrum values to match across two sources.

    Binds a new spectrum source (the HATS conversion) to the source the
    encoder probes validated, per row: shared wavelength grid, flux, ivar,
    and mask must agree within float32 conversion tolerance. Returns the
    observed maximum differences for the preflight report; raises if any
    array disagrees beyond tolerance.
    """
    def _row_index(source: "SpectrumBatch", label: str) -> int:
        ids = [str(value) for value in source.object_id]
        if str(object_id) not in ids:
            raise ValueError(f"Binding object {object_id!r} missing from the {label} batch")
        return ids.index(str(object_id))

    if batch.ivar is None or reference.ivar is None or batch.mask is None or reference.mask is None:
        raise ValueError("Value binding requires ivar and mask on both batches")
    row = _row_index(batch, "candidate")
    reference_row = _row_index(reference, "reference")
    if batch.wave.shape != reference.wave.shape:
        raise ValueError(
            f"Wavelength grids differ in shape: {batch.wave.shape} vs {reference.wave.shape}"
        )
    report = {
        "max_abs_wave_diff": float(np.max(np.abs(batch.wave - reference.wave))),
        "max_abs_flux_diff": float(np.max(np.abs(batch.flux[row] - reference.flux[reference_row]))),
        "max_abs_ivar_diff": float(np.max(np.abs(batch.ivar[row] - reference.ivar[reference_row]))),
        "mask_mismatch_pixels": int(np.sum(batch.mask[row] != reference.mask[reference_row])),
    }
    mismatched = [
        name
        for name, candidate, expected in (
            ("wave", batch.wave, reference.wave),
            ("flux", batch.flux[row], reference.flux[reference_row]),
            ("ivar", batch.ivar[row], reference.ivar[reference_row]),
        )
        if not np.allclose(candidate, expected, rtol=rtol, atol=atol)
    ]
    if report["mask_mismatch_pixels"]:
        mismatched.append("mask")
    if mismatched:
        raise ValueError(
            f"Spectrum values for object {object_id!r} disagree between sources on "
            f"{mismatched}: {report}"
        )
    return report


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


def save_spectrum_batch(path: Path, batch: "SpectrumBatch") -> None:
    """Cache an exact spectrum batch without Python-object serialization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        object_id=np.asarray(batch.object_id, dtype=str),
        flux=np.asarray(batch.flux, dtype=np.float32),
        wave=np.asarray(batch.wave, dtype=np.float32),
        ivar=np.asarray(batch.ivar, dtype=np.float32),
        mask=np.asarray(batch.mask, dtype=bool),
    )


def load_spectrum_batch(path: Path) -> "SpectrumBatch":
    """Load a cache written by :func:`save_spectrum_batch`."""
    from .encoders import SpectrumBatch

    with np.load(Path(path), allow_pickle=False) as payload:
        return SpectrumBatch(
            object_id=payload["object_id"],
            flux=payload["flux"],
            wave=payload["wave"],
            ivar=payload["ivar"],
            mask=payload["mask"],
        )


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
