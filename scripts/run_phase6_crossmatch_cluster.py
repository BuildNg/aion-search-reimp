"""Cluster entrypoint for deterministic or morphology-prioritized HSC × DESI crossmatch.

The left side is an exact configured HSC population selected from pinned
AION-Search source metadata. It retains its configured anchor, optionally takes
Galaxy Zoo morphology priorities, then uses stable hash fill after exclusions.
The right side is a pinned MMU DESI HATS catalog opened column-pruned through LSDB.
No image, caption, embedding, spectrum array, or model weight is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from aion_reimp.artifacts import initialize_run, tracked_run, write_json
from aion_reimp.datasets import load_pinned_dataset
from aion_reimp.manifest import (
    assert_exclusion_coverage,
    coordinate_exclusion_coverage,
    exact_exclusion_coverage,
)
from aion_reimp.utils import file_digest
from spec_probes.morphology_coverage import (
    GALAXY_ZOO_COLUMNS,
    add_reliable_morphology_labels,
    match_galaxy_zoo,
)
from spectra_crossmatch.config import load_config
from spectra_crossmatch.crossmatch import (
    annotate_candidates,
    normalize_lsdb_matches,
    select_nearest_valid,
    source_fingerprint,
    summarize_matches,
)
from spectra_crossmatch.source import SOURCE_COLUMNS, normalize_source_metadata, select_source_population


PREFLIGHT_CONTRACT_VERSION = 2
PREFLIGHT_CODE_PATHS = (
    "scripts/run_phase6_crossmatch_cluster.py",
    "src/spectra_crossmatch/config.py",
    "src/spectra_crossmatch/source.py",
    "src/spectra_crossmatch/crossmatch.py",
    "src/spec_probes/morphology_coverage.py",
)
PREFLIGHT_DISTRIBUTIONS = (
    "datasets",
    "huggingface-hub",
    "lsdb",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _installed_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for distribution in PREFLIGHT_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED"
    return versions


def _preflight_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    return {
        "contract_version": PREFLIGHT_CONTRACT_VERSION,
        "config": dict(config),
        "code_sha256": {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in PREFLIGHT_CODE_PATHS
        },
        "python_version": platform.python_version(),
        "package_versions": _installed_versions(),
    }


def _verify_dataset_revision(repo_id: str, revision: str) -> str:
    from huggingface_hub import HfApi

    resolved = HfApi().dataset_info(repo_id, revision=revision).sha
    if resolved != revision:
        raise RuntimeError(
            f"Resolved dataset revision {resolved!r} does not equal pin {revision!r} for {repo_id}"
        )
    return resolved


def _load_anchor_ids(config: Mapping[str, Any]) -> set[str]:
    source = config["source_population"]
    anchor = source["anchor"]
    run_dir = Path(anchor["run_dir"])
    status_path = run_dir / "run_status.json"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / anchor["manifest_path"]
    for path in (status_path, summary_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Anchor artifact missing: {path}")

    status = json.loads(status_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete" or summary.get("run_id") != anchor["run_id"]:
        raise ValueError("Anchor crossmatch run is not the configured completed run")
    fingerprint_field = anchor.get(
        "summary_fingerprint_field", "captioned_source_fingerprint"
    )
    if summary.get(fingerprint_field) != anchor["source_fingerprint"]:
        raise ValueError("Anchor source fingerprint differs from the configured authoritative value")

    manifest = pd.read_parquet(
        manifest_path,
        columns=[anchor["object_id_column"], anchor["survey_column"]],
    )
    if len(manifest) != int(anchor["expected_manifest_rows"]):
        raise ValueError(
            f"Anchor manifest has {len(manifest)} rows; expected {anchor['expected_manifest_rows']}"
        )
    if manifest[anchor["object_id_column"]].astype(str).duplicated().any():
        raise ValueError("Anchor manifest contains duplicate object IDs")
    selected = manifest.loc[
        manifest[anchor["survey_column"]].astype(str).eq(str(source["survey_value"])),
        anchor["object_id_column"],
    ].astype(str)
    if len(selected) != int(anchor["expected_survey_rows"]):
        raise ValueError(
            f"Anchor has {len(selected)} {source['survey_value']!r} rows; "
            f"expected {anchor['expected_survey_rows']}"
        )
    return set(selected)


def _load_morphology_priority_ids(
    metadata: pd.DataFrame,
    excluded_ids: set[str],
    config: Mapping[str, Any],
) -> Tuple[list[str], Dict[str, Any]]:
    source = config["source_population"]
    spec = source.get("morphology_priority")
    if spec is None:
        return [], {"enabled": False}

    catalog_path = Path(spec["catalog_path"])
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Galaxy Zoo catalog missing: {catalog_path}. Download the pinned file from "
            f"{spec['source_url']}"
        )
    actual_md5 = file_digest(catalog_path, "md5")
    if actual_md5 != spec["source_md5"]:
        raise ValueError(
            f"Galaxy Zoo catalog MD5 mismatch: expected {spec['source_md5']}, got {actual_md5}"
        )
    catalog = pd.read_parquet(catalog_path, columns=list(GALAXY_ZOO_COLUMNS))
    if len(catalog) != int(spec["expected_catalog_rows"]):
        raise ValueError(
            f"Expected {spec['expected_catalog_rows']} Galaxy Zoo rows, found {len(catalog)}"
        )

    candidates = metadata.loc[
        metadata["source_survey"].eq(str(source["survey_value"]))
        & ~metadata["source_object_id"].isin(excluded_ids)
    ].copy()
    manifest = candidates.rename(
        columns={
            "source_object_id": "object_id",
            "source_ra": "spectrum_ra",
            "source_dec": "spectrum_dec",
        }
    )[["object_id", "spectrum_ra", "spectrum_dec"]]
    manifest["z"] = 0.0
    matched = match_galaxy_zoo(
        manifest,
        catalog,
        radius_arcsec=float(spec["match_radius_arcsec"]),
    )
    raw_match_rows = len(matched)
    duplicated = matched["galaxy_zoo_dr8_id"].duplicated(keep=False)
    duplicate_ids = int(matched.loc[duplicated, "galaxy_zoo_dr8_id"].nunique())
    matched = (
        matched.sort_values(["separation_arcsec", "object_id"], kind="mergesort")
        .drop_duplicates("galaxy_zoo_dr8_id", keep="first")
        .reset_index(drop=True)
    )
    labelled = add_reliable_morphology_labels(
        matched,
        fraction_threshold=float(spec["reliable_fraction_threshold"]),
    )

    labels = list(spec["labels"])
    labelled["_priority_tier"] = len(labels)
    reliable_counts: Dict[str, int] = {}
    exclusive_counts: Dict[str, int] = {}
    for tier, label in enumerate(labels):
        flag = labelled[f"reliable_{label}"].astype(bool)
        reliable_counts[label] = int(flag.sum())
        choose = flag & labelled["_priority_tier"].eq(len(labels))
        labelled.loc[choose, "_priority_tier"] = tier
        exclusive_counts[label] = int(choose.sum())
    priority = labelled.loc[labelled["_priority_tier"].lt(len(labels))].copy()
    priority["_priority_key"] = priority["object_id"].map(
        lambda object_id: _fingerprint(
            {
                "salt": source["selection_salt"],
                "seed": source["selection_seed"],
                "object_id": str(object_id),
                "purpose": "morphology_priority",
            }
        )
    )
    priority = priority.sort_values(
        ["_priority_tier", "_priority_key", "object_id"], kind="mergesort"
    )
    priority_ids = priority["object_id"].astype(str).tolist()
    if not priority_ids:
        raise ValueError("Morphology targeting found no eligible priority objects")
    return priority_ids, {
        "enabled": True,
        "catalog_path": str(catalog_path),
        "source_url": spec["source_url"],
        "source_md5": actual_md5,
        "catalog_rows": int(len(catalog)),
        "candidate_source_rows": int(len(candidates)),
        "galaxy_zoo_matches_before_deduplication": int(raw_match_rows),
        "duplicate_galaxy_zoo_ids": duplicate_ids,
        "galaxy_zoo_matches": int(len(matched)),
        "match_radius_arcsec": float(spec["match_radius_arcsec"]),
        "reliable_fraction_threshold": float(spec["reliable_fraction_threshold"]),
        "label_priority": labels,
        "reliable_counts": reliable_counts,
        "exclusive_priority_counts": exclusive_counts,
        "priority_objects": int(len(priority_ids)),
    }


def _build_exclusion_coverage(
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    spec = config["exclusions"]
    canonical = metadata.rename(
        columns={
            "source_object_id": "object_id",
            "source_ra": "ra",
            "source_dec": "dec",
        }
    )
    screen = pd.read_parquet(Path(spec["caption_screen_labels"]), columns=["object_id"])
    exact = exact_exclusion_coverage(
        canonical["object_id"], "caption_screen_64", screen["object_id"]
    )
    benchmarks = {
        name: pd.read_parquet(Path(path))
        for name, path in spec["benchmark_coordinates"].items()
    }
    coordinate = coordinate_exclusion_coverage(
        canonical,
        benchmarks,
        radius_arcsec=float(spec["radius_arcsec"]),
    )
    coverage = pd.concat([exact, coordinate], ignore_index=True)
    assert_exclusion_coverage(
        coverage,
        expected_rows=len(screen) + sum(len(frame) for frame in benchmarks.values()),
    )
    return coverage


def _load_source_population(
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    spec = config["source_population"]
    dataset = load_pinned_dataset(spec["repo_id"], spec["revision"], spec["split"])
    requested = [
        spec["object_id_column"],
        spec["survey_column"],
        spec["ra_column"],
        spec["dec_column"],
    ]
    missing = set(requested) - set(dataset.column_names)
    if missing:
        raise ValueError(f"Pinned source dataset missing columns: {sorted(missing)}")
    raw = dataset.select_columns(requested).to_pandas()
    raw["_source_row_id"] = np.arange(len(raw), dtype=np.int64)
    metadata = normalize_source_metadata(
        raw,
        columns={
            "object_id": spec["object_id_column"],
            "survey": spec["survey_column"],
            "ra": spec["ra_column"],
            "dec": spec["dec_column"],
            "source_row_id": "_source_row_id",
        },
    )
    coverage = _build_exclusion_coverage(metadata, config)
    excluded_ids = set(
        coverage.loc[coverage["status"].eq("matched"), "source_object_id"].astype(str)
    )
    anchor_ids = _load_anchor_ids(config)
    priority_ids, priority_provenance = _load_morphology_priority_ids(
        metadata, excluded_ids, config
    )
    selected = select_source_population(
        metadata,
        survey=str(spec["survey_value"]),
        sample_size=int(spec["sample_size"]),
        seed=int(spec["selection_seed"]),
        salt=str(spec["selection_salt"]),
        excluded_object_ids=excluded_ids,
        anchor_object_ids=anchor_ids,
        priority_object_ids=priority_ids,
        anchor_selection_reason=str(
            spec["anchor"].get("selection_reason", "anchor_phase6_crossmatch_v3")
        ),
    )
    provenance = {
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "split": spec["split"],
        "dataset_rows": int(len(metadata)),
        "survey": spec["survey_value"],
        "survey_rows_before_exclusions": int(
            metadata["source_survey"].eq(str(spec["survey_value"])).sum()
        ),
        "matched_exclusion_objects": int(len(excluded_ids)),
        "anchor_rows": int(len(anchor_ids)),
        "selected_rows": int(len(selected)),
        "deterministic_expansion_rows": int(
            selected["selection_reason"].eq("deterministic_hsc_expansion").sum()
        ),
        "morphology_priority_rows": int(
            selected["selection_reason"].eq("morphology_priority").sum()
        ),
        "morphology_priority": priority_provenance,
        "selection_seed": int(spec["selection_seed"]),
        "selection_salt": str(spec["selection_salt"]),
        "fingerprint": source_fingerprint(selected),
    }
    return selected, coverage, provenance


def _catalog_uri(config: Mapping[str, Any]) -> str:
    spec = config["desi_catalog"]
    return f"hf://datasets/{spec['repo_id']}@{spec['revision']}"


def _desi_columns(config: Mapping[str, Any]) -> list[str]:
    spec = config["desi_catalog"]
    return [
        spec["object_id_column"],
        spec["ra_column"],
        spec["dec_column"],
        spec["redshift_column"],
        spec["redshift_error_column"],
        spec["zwarn_column"],
    ]


def _open_desi_catalog(config: Mapping[str, Any]):
    import lsdb

    return lsdb.open_catalog(_catalog_uri(config), columns=_desi_columns(config))


def _normalize(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    desi = config["desi_catalog"]
    return normalize_lsdb_matches(
        frame,
        source_columns={
            "object_id": "source_object_id",
            "survey": "source_survey",
            "ra": "source_ra",
            "dec": "source_dec",
            "source_row_id": "source_row_id",
            "selection_reason": "selection_reason",
            "selection_rank": "selection_rank",
        },
        desi_columns={
            "object_id": desi["object_id_column"],
            "ra": desi["ra_column"],
            "dec": desi["dec_column"],
            "redshift": desi["redshift_column"],
            "redshift_error": desi["redshift_error_column"],
            "zwarn": desi["zwarn_column"],
        },
    )


def _annotate(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    quality = config["quality"]
    return annotate_candidates(
        frame,
        zwarn_good_value=bool(quality["zwarn_good_value"]),
        minimum_redshift=float(quality["minimum_redshift"]),
        require_positive_redshift_error=bool(quality["require_positive_redshift_error"]),
    )


def run_preflight(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Bind the exact scale population and one real DESI self-match without a run dir."""
    report: Dict[str, Any] = {"status": "pass", "checks": {}}
    try:
        contract = _preflight_contract(config)
        report["contract"] = contract
        report["contract_fingerprint"] = _fingerprint(contract)
        missing_packages = [
            name for name, version in contract["package_versions"].items()
            if version == "NOT_INSTALLED"
        ]
        if missing_packages:
            raise RuntimeError(f"Required packages not installed: {missing_packages}")
        report["checks"]["contract"] = {"ok": True}

        source_spec = config["source_population"]
        source_revision = _verify_dataset_revision(
            source_spec["repo_id"], source_spec["revision"]
        )
        source, coverage, provenance = _load_source_population(config)
        report["checks"]["source_population"] = {
            "ok": True,
            **provenance,
            "resolved_revision": source_revision,
            "exclusion_coverage_rows": int(len(coverage)),
        }

        desi = config["desi_catalog"]
        desi_revision = _verify_dataset_revision(desi["repo_id"], desi["revision"])
        catalog = _open_desi_catalog(config)
        missing = set(_desi_columns(config)) - set(catalog.columns)
        if missing:
            raise RuntimeError(f"DESI HATS catalog missing configured columns: {sorted(missing)}")
        if config["crossmatch"]["require_right_margin"] and catalog.margin is None:
            raise RuntimeError("DESI HATS catalog has no loaded right-margin catalog")

        remote_row = catalog.head(1).iloc[0]
        synthetic_left = pd.DataFrame(
            {
                "source_object_id": ["preflight-self-match"],
                "source_survey": ["preflight"],
                "source_ra": [float(remote_row[desi["ra_column"]])],
                "source_dec": [float(remote_row[desi["dec_column"]])],
                "source_row_id": [-1],
                "selection_reason": ["preflight"],
                "selection_rank": [-1],
            }
        ).loc[:, SOURCE_COLUMNS]
        import lsdb

        left_catalog = lsdb.from_dataframe(
            synthetic_left,
            ra_column="source_ra",
            dec_column="source_dec",
            catalog_name="source",
            margin_threshold=max(config["crossmatch"]["radii_arcsec"]) + 1.0,
        )
        raw = left_catalog.crossmatch(
            catalog,
            n_neighbors=1,
            radius_arcsec=0.1,
            require_right_margin=True,
            suffixes=("_source", "_desi"),
            suffix_method="all_columns",
            log_changes=False,
        ).compute()
        normalized = _annotate(_normalize(raw, config), config)
        if len(normalized) != 1 or float(normalized.iloc[0]["separation_arcsec"]) > 1e-6:
            raise RuntimeError("Pinned DESI self-match did not return one zero-separation row")
        if not bool(normalized.iloc[0]["is_valid_spectrum"]):
            raise RuntimeError(
                "Pinned DESI self-match failed the configured quality rule: "
                f"ZWARN={normalized.iloc[0]['desi_zwarn']!r}, "
                f"Z={normalized.iloc[0]['desi_z']!r}, "
                f"ZERR={normalized.iloc[0]['desi_zerr']!r}, "
                f"reason={normalized.iloc[0]['quality_exclusion_reason']!r}"
            )
        report["checks"]["desi_catalog"] = {
            "ok": True,
            "repo_id": desi["repo_id"],
            "revision": desi_revision,
            "catalog_uri": _catalog_uri(config),
            "columns": _desi_columns(config),
            "right_margin_loaded": catalog.margin is not None,
            "self_match_separation_arcsec": float(normalized.iloc[0]["separation_arcsec"]),
            "self_match_zwarn": bool(normalized.iloc[0]["desi_zwarn"]),
            "self_match_quality_valid": True,
        }
    except Exception as error:  # noqa: BLE001 - preserve a complete failure report
        report["status"] = "fail"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
    return report


def require_passing_preflight(
    config: Mapping[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    report_path = Path(config["run"]["preflight_report"])
    if not report_path.exists():
        raise FileNotFoundError(f"Crossmatch preflight report missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise RuntimeError(f"Crossmatch preflight did not pass: status={report.get('status')!r}")
    if report.get("contract_fingerprint") != _fingerprint(_preflight_contract(config)):
        raise RuntimeError("Crossmatch preflight is stale for current config, code, or packages")
    current_source, coverage, provenance = _load_source_population(config)
    expected = report.get("checks", {}).get("source_population", {}).get("fingerprint")
    if expected != source_fingerprint(current_source) or expected != provenance["fingerprint"]:
        raise RuntimeError("Crossmatch preflight is stale for the selected source population")
    return report, current_source, coverage, provenance


def run_full(config: Mapping[str, Any]) -> Path:
    preflight, source, coverage, source_provenance = require_passing_preflight(config)
    run_dir = initialize_run(
        Path(config["run"]["output_root"]),
        config["run"]["id"],
        config,
        shlex.join(sys.argv),
    )
    write_json(run_dir / "preflight_contract.json", preflight)

    with tracked_run(run_dir, {"phase": 6, "condition": "hsc_x_desi_crossmatch"}):
        source.to_parquet(run_dir / "source_manifest.parquet", index=False)
        coverage.to_parquet(run_dir / "exclusion_coverage.parquet", index=False)
        write_json(run_dir / "source_provenance.json", source_provenance)
        catalog = _open_desi_catalog(config)
        import lsdb

        left_catalog = lsdb.from_dataframe(
            source,
            ra_column="source_ra",
            dec_column="source_dec",
            catalog_name="source",
            margin_threshold=max(config["crossmatch"]["radii_arcsec"]) + 1.0,
        )
        raw = left_catalog.crossmatch(
            catalog,
            n_neighbors=int(config["crossmatch"]["max_neighbors"]),
            radius_arcsec=float(max(config["crossmatch"]["radii_arcsec"])),
            require_right_margin=bool(config["crossmatch"]["require_right_margin"]),
            suffixes=("_source", "_desi"),
            suffix_method="all_columns",
            log_changes=False,
        ).compute()
        candidates = _annotate(_normalize(raw, config), config)
        candidates.to_parquet(run_dir / "candidate_matches.parquet", index=False)

        candidate_counts = candidates.groupby("source_object_id").size()
        limit_hits = int((candidate_counts >= int(config["crossmatch"]["max_neighbors"])).sum())
        if limit_hits:
            raise RuntimeError(
                f"{limit_hits} source objects reached max_neighbors={config['crossmatch']['max_neighbors']}; "
                "candidate preservation may be truncated, so increase the cap and use a new run ID"
            )

        primary_radius = float(config["crossmatch"]["primary_radius_arcsec"])
        selected = select_nearest_valid(candidates, primary_radius)
        selected.to_parquet(run_dir / "selected_matches.parquet", index=False)
        by_radius, by_survey, summary = summarize_matches(
            source,
            candidates,
            config["crossmatch"]["radii_arcsec"],
            primary_radius,
        )
        by_radius.to_csv(run_dir / "counts_by_radius.csv", index=False)
        by_survey.to_csv(run_dir / "counts_by_survey.csv", index=False)
        target = int(config["crossmatch"]["target_valid_matches"])
        write_json(
            run_dir / "summary.json",
            {
                **summary,
                "run_id": config["run"]["id"],
                "source_population_fingerprint": source_fingerprint(source),
                "source_population": source_provenance,
                "desi_repo_id": config["desi_catalog"]["repo_id"],
                "desi_revision": config["desi_catalog"]["revision"],
                "radii_arcsec": [float(value) for value in config["crossmatch"]["radii_arcsec"]],
                "target_valid_matches": target,
                "primary_valid_matches": int(len(selected)),
                "target_met": bool(len(selected) >= target),
                "max_neighbors": int(config["crossmatch"]["max_neighbors"]),
                "candidate_limit_reached_objects": limit_hits,
                "duplicate_policy": config["crossmatch"]["duplicate_policy"],
                "quality_policy": dict(config["quality"]),
            },
        )
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase6_crossmatch.yaml"))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.preflight:
        report = run_preflight(config)
        report_path = Path(config["run"]["preflight_report"])
        write_json(report_path, report, overwrite=report_path.exists())
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "pass" else 1
    run_dir = run_full(config)
    print(f"Crossmatch scale run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
