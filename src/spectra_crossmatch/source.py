"""Pure source-population normalization and deterministic scale selection."""

from __future__ import annotations

import hashlib
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


METADATA_COLUMNS = (
    "source_object_id",
    "source_survey",
    "source_ra",
    "source_dec",
    "source_row_id",
)

SOURCE_COLUMNS = (
    *METADATA_COLUMNS,
    "selection_reason",
    "selection_rank",
)


def normalize_source_metadata(
    frame: pd.DataFrame,
    *,
    columns: Mapping[str, str],
) -> pd.DataFrame:
    """Normalize a pinned image catalog's metadata into the crossmatch schema."""
    rename = {
        columns["object_id"]: "source_object_id",
        columns["survey"]: "source_survey",
        columns["ra"]: "source_ra",
        columns["dec"]: "source_dec",
        columns["source_row_id"]: "source_row_id",
    }
    missing = set(rename) - set(frame)
    if missing:
        raise ValueError(f"Source metadata missing columns: {sorted(missing)}")
    source = frame.rename(columns=rename).loc[:, METADATA_COLUMNS].copy()
    source["source_object_id"] = source["source_object_id"].astype(str)
    source["source_survey"] = source["source_survey"].astype(str)
    if source["source_object_id"].duplicated().any():
        duplicate = source.loc[source["source_object_id"].duplicated(), "source_object_id"].iloc[0]
        raise ValueError(f"Duplicate source object ID: {duplicate}")
    source["source_ra"] = pd.to_numeric(source["source_ra"], errors="raise")
    source["source_dec"] = pd.to_numeric(source["source_dec"], errors="raise")
    if not np.isfinite(source[["source_ra", "source_dec"]].to_numpy(dtype=float)).all():
        raise ValueError("Source coordinates contain non-finite values")
    if not source["source_ra"].between(0.0, 360.0, inclusive="left").all():
        raise ValueError("Source RA must be in [0, 360)")
    if not source["source_dec"].between(-90.0, 90.0, inclusive="both").all():
        raise ValueError("Source Dec must be in [-90, 90]")
    return source.reset_index(drop=True)


def _selection_key(object_id: str, seed: int, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{object_id}|{seed}".encode("utf-8")).hexdigest()


def select_source_population(
    metadata: pd.DataFrame,
    *,
    survey: str,
    sample_size: int,
    seed: int,
    salt: str,
    excluded_object_ids: Iterable[str],
    anchor_object_ids: Iterable[str],
) -> pd.DataFrame:
    """Expand a required anchor set to an exact, deterministic survey sample."""
    missing = set(METADATA_COLUMNS) - set(metadata)
    if missing:
        raise ValueError(f"Normalized source metadata missing columns: {sorted(missing)}")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not str(survey).strip() or not str(salt).strip():
        raise ValueError("survey and salt must be non-empty")

    excluded = {str(value) for value in excluded_object_ids}
    anchors = {str(value) for value in anchor_object_ids}
    if anchors & excluded:
        raise ValueError(f"{len(anchors & excluded)} anchor objects are benchmark exclusions")

    survey_rows = metadata.loc[metadata["source_survey"].eq(str(survey))].copy()
    survey_ids = set(survey_rows["source_object_id"])
    missing_anchors = anchors - survey_ids
    if missing_anchors:
        raise ValueError(f"{len(missing_anchors)} anchor objects are absent from survey {survey!r}")
    if len(anchors) > sample_size:
        raise ValueError(f"Anchor has {len(anchors)} rows, exceeding sample_size={sample_size}")

    eligible = survey_rows.loc[~survey_rows["source_object_id"].isin(excluded)].copy()
    if len(eligible) < sample_size:
        raise ValueError(
            f"Only {len(eligible)} eligible {survey!r} rows for sample_size={sample_size}"
        )

    additions_needed = sample_size - len(anchors)
    additions = eligible.loc[~eligible["source_object_id"].isin(anchors)].copy()
    additions["_selection_key"] = additions["source_object_id"].map(
        lambda value: _selection_key(str(value), seed, salt)
    )
    additions = additions.sort_values(["_selection_key", "source_object_id"], kind="mergesort")
    additions = additions.head(additions_needed).drop(columns="_selection_key")
    if len(additions) != additions_needed:
        raise ValueError(f"Only {len(additions)} non-anchor additions for required {additions_needed}")

    anchor_rows = eligible.loc[eligible["source_object_id"].isin(anchors)].copy()
    anchor_rows["selection_reason"] = "anchor_phase6_crossmatch_v3"
    additions["selection_reason"] = "deterministic_hsc_expansion"
    selected = pd.concat([anchor_rows, additions], ignore_index=True)
    selected["_selection_key"] = selected["source_object_id"].map(
        lambda value: _selection_key(str(value), seed, salt)
    )
    selected = selected.sort_values(["_selection_key", "source_object_id"], kind="mergesort")
    selected["selection_rank"] = np.arange(len(selected), dtype=np.int64)
    selected = selected.drop(columns="_selection_key").reset_index(drop=True)

    if len(selected) != sample_size or selected["source_object_id"].duplicated().any():
        raise AssertionError("Selected source population is not the requested exact unique sample")
    if not anchors.issubset(set(selected["source_object_id"])):
        raise AssertionError("Selected source population dropped required anchor objects")
    if set(selected["source_survey"]) != {str(survey)}:
        raise AssertionError("Selected source population contains an unexpected survey")
    return selected.loc[:, SOURCE_COLUMNS]
