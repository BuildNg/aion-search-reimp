"""Strict config contract for the Phase-6 crossmatch feasibility probe."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


class CrossmatchConfigError(ValueError):
    """Raised when a crossmatch config is incomplete or contains unused keys."""


def _section(
    data: Mapping[str, Any],
    name: str,
    allowed: Iterable[str],
    required: Iterable[str],
) -> Dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise CrossmatchConfigError(f"{name} must be a mapping")
    allowed_set = set(allowed)
    required_set = set(required)
    unknown = sorted(set(value) - allowed_set)
    missing = sorted(required_set - set(value))
    if unknown:
        raise CrossmatchConfigError(f"Unknown {name} keys: {unknown}")
    if missing:
        raise CrossmatchConfigError(f"Missing {name} keys: {missing}")
    return dict(value)


def _require_commit(value: Any, field: str) -> None:
    text = str(value)
    if len(text) != 40 or any(ch not in "0123456789abcdef" for ch in text.lower()):
        raise CrossmatchConfigError(f"{field} must be a 40-character commit SHA, got {value!r}")


def validate_config(data: Mapping[str, Any]) -> Dict[str, Any]:
    top = {
        "schema_version",
        "kind",
        "run",
        "captioned_source",
        "desi_catalog",
        "crossmatch",
        "quality",
    }
    if data.get("schema_version") != 1:
        raise CrossmatchConfigError("schema_version must equal 1")
    if data.get("kind") != "phase6_crossmatch":
        raise CrossmatchConfigError(f"Unsupported config kind: {data.get('kind')!r}")
    unknown = sorted(set(data) - top)
    missing = sorted(top - set(data))
    if unknown:
        raise CrossmatchConfigError(f"Unknown top-level keys: {unknown}")
    if missing:
        raise CrossmatchConfigError(f"Missing top-level keys: {missing}")

    run = _section(
        data,
        "run",
        {"id", "output_root", "preflight_report"},
        {"id", "output_root", "preflight_report"},
    )
    if not str(run["id"]).strip() or not str(run["preflight_report"]).strip():
        raise CrossmatchConfigError("run.id and run.preflight_report must be non-empty")
    source = _section(
        data,
        "captioned_source",
        {
            "run_dir",
            "run_id",
            "manifest_path",
            "source_rows_path",
            "expected_rows",
            "object_id_column",
            "survey_column",
            "ra_column",
            "dec_column",
            "source_row_id_column",
        },
        {
            "run_dir",
            "run_id",
            "manifest_path",
            "source_rows_path",
            "expected_rows",
            "object_id_column",
            "survey_column",
            "ra_column",
            "dec_column",
            "source_row_id_column",
        },
    )
    if not isinstance(source["expected_rows"], int) or isinstance(source["expected_rows"], bool) or source["expected_rows"] <= 0:
        raise CrossmatchConfigError("captioned_source.expected_rows must be a positive integer")

    desi = _section(
        data,
        "desi_catalog",
        {
            "repo_id",
            "revision",
            "object_id_column",
            "ra_column",
            "dec_column",
            "redshift_column",
            "redshift_error_column",
            "zwarn_column",
        },
        {
            "repo_id",
            "revision",
            "object_id_column",
            "ra_column",
            "dec_column",
            "redshift_column",
            "redshift_error_column",
            "zwarn_column",
        },
    )
    _require_commit(desi["revision"], "desi_catalog.revision")

    match = _section(
        data,
        "crossmatch",
        {"radii_arcsec", "max_neighbors", "require_right_margin", "duplicate_policy"},
        {"radii_arcsec", "max_neighbors", "require_right_margin", "duplicate_policy"},
    )
    radii = match["radii_arcsec"]
    if not isinstance(radii, list) or not radii:
        raise CrossmatchConfigError("crossmatch.radii_arcsec must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in radii):
        raise CrossmatchConfigError("crossmatch.radii_arcsec must contain positive numbers")
    if radii != sorted(set(radii)):
        raise CrossmatchConfigError("crossmatch.radii_arcsec must be sorted and unique")
    if not isinstance(match["max_neighbors"], int) or isinstance(match["max_neighbors"], bool) or match["max_neighbors"] < 2:
        raise CrossmatchConfigError("crossmatch.max_neighbors must be an integer >= 2")
    if match["require_right_margin"] is not True:
        raise CrossmatchConfigError("crossmatch.require_right_margin must be true")
    if match["duplicate_policy"] != "nearest_valid_then_desi_object_id":
        raise CrossmatchConfigError(
            "crossmatch.duplicate_policy must be 'nearest_valid_then_desi_object_id'"
        )

    quality = _section(
        data,
        "quality",
        {"zwarn_good_value", "minimum_redshift", "require_positive_redshift_error"},
        {"zwarn_good_value", "minimum_redshift", "require_positive_redshift_error"},
    )
    if quality["zwarn_good_value"] is not False:
        raise CrossmatchConfigError(
            "quality.zwarn_good_value must be false: the pinned HATS catalog stores "
            "raw DESI ZWARN semantics, so False means integer ZWARN == 0"
        )
    if isinstance(quality["minimum_redshift"], bool) or not isinstance(quality["minimum_redshift"], (int, float)):
        raise CrossmatchConfigError("quality.minimum_redshift must be numeric")
    if quality["require_positive_redshift_error"] is not True:
        raise CrossmatchConfigError("quality.require_positive_redshift_error must be true")

    return dict(data)


def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise CrossmatchConfigError("Config root must be a mapping")
    return validate_config(raw)
