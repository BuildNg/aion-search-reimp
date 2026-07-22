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


def _require_sha256(value: Any, field: str) -> None:
    text = str(value)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text.lower()):
        raise CrossmatchConfigError(f"{field} must be a 64-character SHA-256, got {value!r}")


def _require_md5(value: Any, field: str) -> None:
    text = str(value)
    if len(text) != 32 or any(ch not in "0123456789abcdef" for ch in text.lower()):
        raise CrossmatchConfigError(f"{field} must be a 32-character MD5, got {value!r}")


def validate_config(data: Mapping[str, Any]) -> Dict[str, Any]:
    top = {
        "schema_version",
        "kind",
        "run",
        "source_population",
        "exclusions",
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
        "source_population",
        {
            "repo_id",
            "revision",
            "split",
            "sample_size",
            "survey_value",
            "selection_seed",
            "selection_salt",
            "object_id_column",
            "survey_column",
            "ra_column",
            "dec_column",
            "anchor",
            "morphology_priority",
        },
        {
            "repo_id",
            "revision",
            "split",
            "sample_size",
            "survey_value",
            "selection_seed",
            "selection_salt",
            "object_id_column",
            "survey_column",
            "ra_column",
            "dec_column",
            "anchor",
        },
    )
    _require_commit(source["revision"], "source_population.revision")
    if not isinstance(source["sample_size"], int) or isinstance(source["sample_size"], bool) or source["sample_size"] <= 0:
        raise CrossmatchConfigError("source_population.sample_size must be a positive integer")
    if not isinstance(source["selection_seed"], int) or isinstance(source["selection_seed"], bool):
        raise CrossmatchConfigError("source_population.selection_seed must be an integer")
    for key in ("repo_id", "split", "survey_value", "selection_salt"):
        if not str(source[key]).strip():
            raise CrossmatchConfigError(f"source_population.{key} must be non-empty")

    anchor = source["anchor"]
    if not isinstance(anchor, dict):
        raise CrossmatchConfigError("source_population.anchor must be a mapping")
    anchor_allowed = {
        "run_dir",
        "run_id",
        "manifest_path",
        "expected_manifest_rows",
        "expected_survey_rows",
        "source_fingerprint",
        "object_id_column",
        "survey_column",
        "summary_fingerprint_field",
        "selection_reason",
    }
    anchor_required = anchor_allowed - {"summary_fingerprint_field", "selection_reason"}
    unknown_anchor = sorted(set(anchor) - anchor_allowed)
    missing_anchor = sorted(anchor_required - set(anchor))
    if unknown_anchor:
        raise CrossmatchConfigError(f"Unknown source_population.anchor keys: {unknown_anchor}")
    if missing_anchor:
        raise CrossmatchConfigError(f"Missing source_population.anchor keys: {missing_anchor}")
    for key in ("summary_fingerprint_field", "selection_reason"):
        if key in anchor and not str(anchor[key]).strip():
            raise CrossmatchConfigError(f"source_population.anchor.{key} must be non-empty")
    for key in ("expected_manifest_rows", "expected_survey_rows"):
        if not isinstance(anchor[key], int) or isinstance(anchor[key], bool) or anchor[key] <= 0:
            raise CrossmatchConfigError(f"source_population.anchor.{key} must be a positive integer")
    if anchor["expected_survey_rows"] > source["sample_size"]:
        raise CrossmatchConfigError(
            "source_population.anchor.expected_survey_rows cannot exceed sample_size"
        )
    _require_sha256(anchor["source_fingerprint"], "source_population.anchor.source_fingerprint")

    morphology = source.get("morphology_priority")
    if morphology is not None:
        if not isinstance(morphology, dict):
            raise CrossmatchConfigError("source_population.morphology_priority must be a mapping")
        morphology_keys = {
            "catalog_path",
            "source_url",
            "source_md5",
            "expected_catalog_rows",
            "match_radius_arcsec",
            "reliable_fraction_threshold",
            "labels",
            "expected_priority_objects",
            "expected_priority_additions",
        }
        unknown_morphology = sorted(set(morphology) - morphology_keys)
        missing_morphology = sorted(morphology_keys - set(morphology))
        if unknown_morphology:
            raise CrossmatchConfigError(
                f"Unknown source_population.morphology_priority keys: {unknown_morphology}"
            )
        if missing_morphology:
            raise CrossmatchConfigError(
                f"Missing source_population.morphology_priority keys: {missing_morphology}"
            )
        _require_md5(
            morphology["source_md5"],
            "source_population.morphology_priority.source_md5",
        )
        for key in ("catalog_path", "source_url"):
            if not str(morphology[key]).strip():
                raise CrossmatchConfigError(
                    f"source_population.morphology_priority.{key} must be non-empty"
                )
        if (
            not isinstance(morphology["expected_catalog_rows"], int)
            or isinstance(morphology["expected_catalog_rows"], bool)
            or morphology["expected_catalog_rows"] <= 0
        ):
            raise CrossmatchConfigError(
                "source_population.morphology_priority.expected_catalog_rows must be positive"
            )
        radius = morphology["match_radius_arcsec"]
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or radius <= 0:
            raise CrossmatchConfigError(
                "source_population.morphology_priority.match_radius_arcsec must be positive"
            )
        threshold = morphology["reliable_fraction_threshold"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0.5 < threshold <= 1.0
        ):
            raise CrossmatchConfigError(
                "source_population.morphology_priority.reliable_fraction_threshold "
                "must be in (0.5, 1.0]"
            )
        allowed_labels = {
            "smooth",
            "featured_or_disk",
            "spiral",
            "barred_spiral",
            "edge_on_disk",
        }
        labels = morphology["labels"]
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) for label in labels)
            or len(labels) != len(set(labels))
            or any(label not in allowed_labels for label in labels)
        ):
            raise CrossmatchConfigError(
                "source_population.morphology_priority.labels must be a non-empty "
                "unique list of supported morphology labels"
            )
        for key in ("expected_priority_objects", "expected_priority_additions"):
            value = morphology[key]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise CrossmatchConfigError(
                    f"source_population.morphology_priority.{key} must be positive"
                )
        if morphology["expected_priority_additions"] > morphology["expected_priority_objects"]:
            raise CrossmatchConfigError(
                "source_population.morphology_priority.expected_priority_additions "
                "cannot exceed expected_priority_objects"
            )
        expected_size = (
            anchor["expected_survey_rows"] + morphology["expected_priority_additions"]
        )
        if source["sample_size"] != expected_size:
            raise CrossmatchConfigError(
                "source_population.sample_size must equal anchor expected_survey_rows plus "
                "morphology_priority.expected_priority_additions"
            )

    exclusions = _section(
        data,
        "exclusions",
        {"radius_arcsec", "caption_screen_labels", "benchmark_coordinates"},
        {"radius_arcsec", "caption_screen_labels", "benchmark_coordinates"},
    )
    if isinstance(exclusions["radius_arcsec"], bool) or not isinstance(exclusions["radius_arcsec"], (int, float)) or exclusions["radius_arcsec"] <= 0:
        raise CrossmatchConfigError("exclusions.radius_arcsec must be positive")
    if not isinstance(exclusions["benchmark_coordinates"], dict) or not exclusions["benchmark_coordinates"]:
        raise CrossmatchConfigError("exclusions.benchmark_coordinates must be a non-empty mapping")

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
        {
            "radii_arcsec",
            "primary_radius_arcsec",
            "target_valid_matches",
            "max_neighbors",
            "require_right_margin",
            "duplicate_policy",
        },
        {
            "radii_arcsec",
            "primary_radius_arcsec",
            "target_valid_matches",
            "max_neighbors",
            "require_right_margin",
            "duplicate_policy",
        },
    )
    radii = match["radii_arcsec"]
    if not isinstance(radii, list) or not radii:
        raise CrossmatchConfigError("crossmatch.radii_arcsec must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in radii):
        raise CrossmatchConfigError("crossmatch.radii_arcsec must contain positive numbers")
    if radii != sorted(set(radii)):
        raise CrossmatchConfigError("crossmatch.radii_arcsec must be sorted and unique")
    if match["primary_radius_arcsec"] not in radii:
        raise CrossmatchConfigError("crossmatch.primary_radius_arcsec must be in radii_arcsec")
    if not isinstance(match["target_valid_matches"], int) or isinstance(match["target_valid_matches"], bool) or match["target_valid_matches"] <= 0:
        raise CrossmatchConfigError("crossmatch.target_valid_matches must be a positive integer")
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
