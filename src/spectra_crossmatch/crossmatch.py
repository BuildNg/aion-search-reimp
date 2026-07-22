"""Pure preparation, quality, duplicate, and summary logic for the crossmatch.

Network and LSDB calls stay in the cluster entrypoint. This module receives
small pandas frames so the scientific contract can be tested locally without
opening either remote survey.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from aion_reimp.manifest import manifest_fingerprint
from .source import SOURCE_COLUMNS

CANDIDATE_COLUMNS = (
    *SOURCE_COLUMNS,
    "desi_object_id",
    "desi_ra",
    "desi_dec",
    "desi_z",
    "desi_zerr",
    "desi_zwarn",
    "separation_arcsec",
)


def source_fingerprint(source: pd.DataFrame) -> str:
    missing = set(SOURCE_COLUMNS) - set(source)
    if missing:
        raise ValueError(f"Source fingerprint missing columns: {sorted(missing)}")
    # The shared dataset-agnostic helper uses ``object_id`` as its stable
    # sort key. Adapt the crossmatch artifact's explicit left-side name at
    # this boundary instead of reimplementing the hash algorithm.
    canonical = source.loc[:, SOURCE_COLUMNS].rename(
        columns={"source_object_id": "object_id"}
    )
    return manifest_fingerprint(canonical)


def normalize_lsdb_matches(
    frame: pd.DataFrame,
    *,
    source_columns: Mapping[str, str],
    desi_columns: Mapping[str, str],
) -> pd.DataFrame:
    """Rename the explicitly suffixed LSDB result into a stable artifact schema."""
    rename = {
        f"{source_columns['object_id']}_source": "source_object_id",
        f"{source_columns['survey']}_source": "source_survey",
        f"{source_columns['ra']}_source": "source_ra",
        f"{source_columns['dec']}_source": "source_dec",
        f"{source_columns['source_row_id']}_source": "source_row_id",
        f"{source_columns['selection_reason']}_source": "selection_reason",
        f"{source_columns['selection_rank']}_source": "selection_rank",
        f"{desi_columns['object_id']}_desi": "desi_object_id",
        f"{desi_columns['ra']}_desi": "desi_ra",
        f"{desi_columns['dec']}_desi": "desi_dec",
        f"{desi_columns['redshift']}_desi": "desi_z",
        f"{desi_columns['redshift_error']}_desi": "desi_zerr",
        f"{desi_columns['zwarn']}_desi": "desi_zwarn",
        "_dist_arcsec": "separation_arcsec",
    }
    missing = set(rename) - set(frame)
    if missing:
        raise ValueError(
            f"LSDB crossmatch output missing expected columns {sorted(missing)}; "
            f"available={sorted(frame.columns)}"
        )
    # LSDB.compute() returns nested_pandas.NestedFrame, whose overridden
    # ``to_parquet`` passes pandas' ``index=`` argument through to PyArrow
    # and fails. Materialize an ordinary pandas DataFrame at this boundary;
    # none of the selected columns is nested.
    selected = frame.rename(columns=rename).loc[:, CANDIDATE_COLUMNS]
    normalized = pd.DataFrame(
        {column: selected[column].to_numpy(copy=True) for column in CANDIDATE_COLUMNS}
    )
    normalized["source_object_id"] = normalized["source_object_id"].astype(str)
    normalized["desi_object_id"] = normalized["desi_object_id"].astype(str)
    normalized["separation_arcsec"] = pd.to_numeric(normalized["separation_arcsec"], errors="raise")
    return normalized.reset_index(drop=True)


def annotate_candidates(
    candidates: pd.DataFrame,
    *,
    zwarn_good_value: bool,
    minimum_redshift: float,
    require_positive_redshift_error: bool,
) -> pd.DataFrame:
    """Apply the preregistered spectrum-quality rule and deterministic ranks."""
    missing = set(CANDIDATE_COLUMNS) - set(candidates)
    if missing:
        raise ValueError(f"Candidate matches missing columns: {sorted(missing)}")
    frame = candidates.copy()
    frame["desi_z"] = pd.to_numeric(frame["desi_z"], errors="coerce")
    frame["desi_zerr"] = pd.to_numeric(frame["desi_zerr"], errors="coerce")
    frame["desi_zwarn"] = frame["desi_zwarn"].astype(bool)

    reasons: List[str] = []
    valid_values: List[bool] = []
    for row in frame.itertuples(index=False):
        row_reasons: List[str] = []
        if bool(row.desi_zwarn) is not bool(zwarn_good_value):
            row_reasons.append("zwarn_not_zero")
        if not np.isfinite(row.desi_z):
            row_reasons.append("redshift_nonfinite")
        elif float(row.desi_z) < float(minimum_redshift):
            row_reasons.append("redshift_below_minimum")
        if not np.isfinite(row.desi_zerr):
            row_reasons.append("redshift_error_nonfinite")
        elif require_positive_redshift_error and float(row.desi_zerr) <= 0.0:
            row_reasons.append("redshift_error_not_positive")
        reasons.append(";".join(row_reasons))
        valid_values.append(not row_reasons)
    frame["quality_exclusion_reason"] = reasons
    frame["is_valid_spectrum"] = valid_values

    frame = frame.sort_values(
        ["source_object_id", "separation_arcsec", "desi_object_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    frame["candidate_rank"] = frame.groupby("source_object_id").cumcount() + 1
    valid_rank = frame.loc[frame["is_valid_spectrum"]].groupby("source_object_id").cumcount() + 1
    frame["valid_candidate_rank"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame.loc[valid_rank.index, "valid_candidate_rank"] = valid_rank.astype("Int64")
    return frame


def select_nearest_valid(candidates: pd.DataFrame, radius_arcsec: float) -> pd.DataFrame:
    eligible = candidates.loc[
        candidates["is_valid_spectrum"] & (candidates["separation_arcsec"] <= float(radius_arcsec))
    ].copy()
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values(
        ["source_object_id", "separation_arcsec", "desi_object_id"],
        kind="mergesort",
    )
    return eligible.drop_duplicates("source_object_id", keep="first").reset_index(drop=True)


def summarize_matches(
    source: pd.DataFrame,
    candidates: pd.DataFrame,
    radii_arcsec: Sequence[float],
    primary_radius_arcsec: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build overall, survey, and selection-stratum feasibility tables."""
    source_ids = set(source["source_object_id"].astype(str))
    if not set(candidates["source_object_id"].astype(str)).issubset(source_ids):
        raise ValueError("Candidate matches contain object IDs outside the source manifest")

    overall_rows: List[Dict[str, Any]] = []
    survey_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    for radius in radii_arcsec:
        within = candidates.loc[candidates["separation_arcsec"] <= float(radius)]
        valid = within.loc[within["is_valid_spectrum"]]
        selected = select_nearest_valid(candidates, float(radius))
        valid_counts = valid.groupby("source_object_id").size()
        selected_desi_counts = selected.groupby("desi_object_id").size() if not selected.empty else pd.Series(dtype=int)
        row = {
            "radius_arcsec": float(radius),
            "source_objects": int(len(source)),
            "matched_any_objects": int(within["source_object_id"].nunique()),
            "matched_valid_objects": int(valid["source_object_id"].nunique()),
            "valid_match_fraction": float(valid["source_object_id"].nunique() / len(source)),
            "valid_candidate_rows": int(len(valid)),
            "ambiguous_valid_objects": int((valid_counts > 1).sum()),
            "selected_unique_desi_objects": int(selected["desi_object_id"].nunique()),
            "shared_selected_desi_objects": int((selected_desi_counts > 1).sum()),
        }
        overall_rows.append(row)

        for survey, survey_source in source.groupby("source_survey", dropna=False):
            survey_ids = set(survey_source["source_object_id"].astype(str))
            survey_within = within.loc[within["source_object_id"].isin(survey_ids)]
            survey_valid = valid.loc[valid["source_object_id"].isin(survey_ids)]
            survey_rows.append(
                {
                    "radius_arcsec": float(radius),
                    "source_survey": str(survey),
                    "source_objects": int(len(survey_source)),
                    "matched_any_objects": int(survey_within["source_object_id"].nunique()),
                    "matched_valid_objects": int(survey_valid["source_object_id"].nunique()),
                    "valid_match_fraction": float(
                        survey_valid["source_object_id"].nunique() / len(survey_source)
                    ),
                }
            )

        for reason, reason_source in source.groupby("selection_reason", dropna=False):
            reason_ids = set(reason_source["source_object_id"].astype(str))
            reason_within = within.loc[within["source_object_id"].isin(reason_ids)]
            reason_valid = valid.loc[valid["source_object_id"].isin(reason_ids)]
            selection_rows.append(
                {
                    "radius_arcsec": float(radius),
                    "selection_reason": str(reason),
                    "source_objects": int(len(reason_source)),
                    "matched_any_objects": int(reason_within["source_object_id"].nunique()),
                    "matched_valid_objects": int(reason_valid["source_object_id"].nunique()),
                    "valid_match_fraction": float(
                        reason_valid["source_object_id"].nunique() / len(reason_source)
                    ),
                }
            )

    exclusion_counts = (
        candidates.loc[~candidates["is_valid_spectrum"], "quality_exclusion_reason"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    if float(primary_radius_arcsec) not in {float(value) for value in radii_arcsec}:
        raise ValueError("primary_radius_arcsec must be one of radii_arcsec")
    selected_primary = select_nearest_valid(candidates, float(primary_radius_arcsec))
    separations = selected_primary["separation_arcsec"].to_numpy(dtype=float)
    primary_selection = [
        row
        for row in selection_rows
        if row["radius_arcsec"] == float(primary_radius_arcsec)
    ]
    summary = {
        "source_objects": int(len(source)),
        "candidate_rows_within_max_radius": int(len(candidates)),
        "valid_candidate_rows_within_max_radius": int(candidates["is_valid_spectrum"].sum()),
        "quality_exclusion_counts": {str(key): int(value) for key, value in exclusion_counts.items()},
        "primary_radius_arcsec": float(primary_radius_arcsec),
        "primary_by_selection_reason": {
            row["selection_reason"]: {
                "source_objects": row["source_objects"],
                "matched_any_objects": row["matched_any_objects"],
                "matched_valid_objects": row["matched_valid_objects"],
                "valid_match_fraction": row["valid_match_fraction"],
            }
            for row in primary_selection
        },
        "selected_primary_separation_arcsec_quantiles": (
            {
                "min": float(np.min(separations)),
                "p50": float(np.quantile(separations, 0.50)),
                "p90": float(np.quantile(separations, 0.90)),
                "p99": float(np.quantile(separations, 0.99)),
                "max": float(np.max(separations)),
            }
            if len(separations)
            else None
        ),
    }
    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(survey_rows),
        pd.DataFrame(selection_rows),
        summary,
    )
