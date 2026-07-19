"""Cluster entrypoint for the probe-scale Legacy-Survey × DESI crossmatch.

The left side is the exact completed Phase-3 caption manifest. The right
side is a pinned HATS conversion of MMU DESI EDR SV3, opened with LSDB and
column-pruned before any partition is read. No spectrum arrays are loaded.
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
from typing import Any, Dict, Mapping

import pandas as pd
import yaml

from aion_reimp.artifacts import initialize_run, tracked_run, write_json
from spectra_crossmatch.config import load_config
from spectra_crossmatch.crossmatch import (
    annotate_candidates,
    normalize_lsdb_matches,
    prepare_captioned_source,
    select_nearest_valid,
    source_fingerprint,
    summarize_matches,
)

PREFLIGHT_CONTRACT_VERSION = 1
PREFLIGHT_CODE_PATHS = (
    "scripts/run_phase6_crossmatch_cluster.py",
    "src/spectra_crossmatch/config.py",
    "src/spectra_crossmatch/crossmatch.py",
)
PREFLIGHT_DISTRIBUTIONS = ("huggingface-hub", "lsdb", "numpy", "pandas", "pyarrow")


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


def _source_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    spec = config["captioned_source"]
    run_dir = Path(spec["run_dir"])
    return run_dir, run_dir / spec["manifest_path"], run_dir / spec["source_rows_path"]


def _load_captioned_source(config: Mapping[str, Any]) -> pd.DataFrame:
    spec = config["captioned_source"]
    run_dir, manifest_path, source_rows_path = _source_paths(config)
    status_path = run_dir / "run_status.json"
    if not status_path.exists():
        raise FileNotFoundError(f"Caption source run has no run_status.json: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError(f"Caption source run is not complete: status={status.get('status')!r}")
    source_config_path = run_dir / "config.yaml"
    if not source_config_path.exists():
        raise FileNotFoundError(f"Caption source run has no resolved config: {source_config_path}")
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    resolved_run_id = source_config.get("run", {}).get("id") if isinstance(source_config, dict) else None
    if resolved_run_id != spec["run_id"]:
        raise ValueError(
            f"Caption source run ID mismatch: config has {resolved_run_id!r}, "
            f"crossmatch requires {spec['run_id']!r}"
        )
    if not manifest_path.exists() or not source_rows_path.exists():
        raise FileNotFoundError(
            f"Caption source artifacts missing: manifest={manifest_path.exists()}, "
            f"source_rows={source_rows_path.exists()}"
        )

    object_id = spec["object_id_column"]
    manifest = pd.read_parquet(manifest_path, columns=[object_id])
    source_rows = pd.read_parquet(
        source_rows_path,
        columns=[
            object_id,
            spec["survey_column"],
            spec["ra_column"],
            spec["dec_column"],
            spec["source_row_id_column"],
        ],
    )
    return prepare_captioned_source(
        manifest,
        source_rows,
        expected_rows=int(spec["expected_rows"]),
        object_id_column=object_id,
        survey_column=spec["survey_column"],
        ra_column=spec["ra_column"],
        dec_column=spec["dec_column"],
        source_row_id_column=spec["source_row_id_column"],
    )


def _catalog_uri(config: Mapping[str, Any]) -> str:
    spec = config["desi_catalog"]
    # Open the collection root, not only its main-catalog subdirectory, so
    # LSDB also resolves the collection.properties-declared 10-arcsec margin.
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
            "object_id": "caption_object_id",
            "survey": "caption_survey",
            "ra": "caption_ra",
            "dec": "caption_dec",
            "source_row_id": "caption_source_row_id",
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
    """Validate exact local inputs and one real remote self-match without a run dir."""
    report: Dict[str, Any] = {"status": "pass", "checks": {}}
    try:
        contract = _preflight_contract(config)
        report["contract"] = contract
        report["contract_fingerprint"] = _fingerprint(contract)
        missing_packages = [
            name for name, version in contract["package_versions"].items() if version == "NOT_INSTALLED"
        ]
        if missing_packages:
            raise RuntimeError(f"Required packages not installed: {missing_packages}")
        report["checks"]["contract"] = {"ok": True}

        source = _load_captioned_source(config)
        report["checks"]["captioned_source"] = {
            "ok": True,
            "rows": int(len(source)),
            "fingerprint": source_fingerprint(source),
            "surveys": {
                str(key): int(value) for key, value in source["caption_survey"].value_counts().items()
            },
        }

        from huggingface_hub import HfApi

        desi = config["desi_catalog"]
        info = HfApi().dataset_info(desi["repo_id"], revision=desi["revision"])
        if info.sha != desi["revision"]:
            raise RuntimeError(
                f"Resolved DESI revision {info.sha!r} does not equal config pin {desi['revision']!r}"
            )
        catalog = _open_desi_catalog(config)
        missing = set(_desi_columns(config)) - set(catalog.columns)
        if missing:
            raise RuntimeError(f"DESI HATS catalog missing configured columns: {sorted(missing)}")
        if config["crossmatch"]["require_right_margin"] and catalog.margin is None:
            raise RuntimeError("DESI HATS catalog has no loaded right-margin catalog")

        remote_row = catalog.head(1).iloc[0]
        import lsdb

        synthetic_left = pd.DataFrame(
            {
                "caption_object_id": ["preflight-self-match"],
                "caption_survey": ["preflight"],
                "caption_ra": [float(remote_row[desi["ra_column"]])],
                "caption_dec": [float(remote_row[desi["dec_column"]])],
                "caption_source_row_id": [-1],
            }
        )
        left_catalog = lsdb.from_dataframe(
            synthetic_left,
            ra_column="caption_ra",
            dec_column="caption_dec",
            catalog_name="caption",
            margin_threshold=max(config["crossmatch"]["radii_arcsec"]) + 1.0,
        )
        raw = left_catalog.crossmatch(
            catalog,
            n_neighbors=1,
            radius_arcsec=0.1,
            require_right_margin=True,
            suffixes=("_caption", "_desi"),
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
            "revision": info.sha,
            "catalog_uri": _catalog_uri(config),
            "columns": _desi_columns(config),
            "right_margin_loaded": catalog.margin is not None,
            "self_match_separation_arcsec": float(normalized.iloc[0]["separation_arcsec"]),
            "self_match_zwarn": bool(normalized.iloc[0]["desi_zwarn"]),
            "self_match_quality_valid": bool(normalized.iloc[0]["is_valid_spectrum"]),
        }
    except Exception as error:  # noqa: BLE001 - preflight must preserve the failure report
        report["status"] = "fail"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
    return report


def require_passing_preflight(config: Mapping[str, Any]) -> Dict[str, Any]:
    report_path = Path(config["run"]["preflight_report"])
    if not report_path.exists():
        raise FileNotFoundError(f"Crossmatch preflight report missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise RuntimeError(f"Crossmatch preflight did not pass: status={report.get('status')!r}")
    current_fingerprint = _fingerprint(_preflight_contract(config))
    if report.get("contract_fingerprint") != current_fingerprint:
        raise RuntimeError("Crossmatch preflight is stale for the current config, code, or package versions")
    current_source = _load_captioned_source(config)
    current_source_fingerprint = source_fingerprint(current_source)
    preflight_source_fingerprint = (
        report.get("checks", {}).get("captioned_source", {}).get("fingerprint")
    )
    if preflight_source_fingerprint != current_source_fingerprint:
        raise RuntimeError(
            "Crossmatch preflight is stale for the completed Phase-3 captioned source artifacts"
        )
    return report


def run_full(config: Mapping[str, Any]) -> Path:
    preflight = require_passing_preflight(config)
    source = _load_captioned_source(config)
    run_dir = initialize_run(
        Path(config["run"]["output_root"]),
        config["run"]["id"],
        config,
        shlex.join(sys.argv),
    )
    write_json(run_dir / "preflight_contract.json", preflight)

    with tracked_run(run_dir, {"phase": 6, "condition": "captioned_legacy_x_desi_crossmatch"}):
        source.to_parquet(run_dir / "source_manifest.parquet", index=False)
        catalog = _open_desi_catalog(config)
        import lsdb

        left_catalog = lsdb.from_dataframe(
            source,
            ra_column="caption_ra",
            dec_column="caption_dec",
            catalog_name="caption",
            margin_threshold=max(config["crossmatch"]["radii_arcsec"]) + 1.0,
        )
        raw = left_catalog.crossmatch(
            catalog,
            n_neighbors=int(config["crossmatch"]["max_neighbors"]),
            radius_arcsec=float(max(config["crossmatch"]["radii_arcsec"])),
            require_right_margin=bool(config["crossmatch"]["require_right_margin"]),
            suffixes=("_caption", "_desi"),
            suffix_method="all_columns",
            log_changes=False,
        ).compute()
        candidates = _annotate(_normalize(raw, config), config)
        candidates.to_parquet(run_dir / "candidate_matches.parquet", index=False)

        candidate_counts = candidates.groupby("caption_object_id").size()
        limit_hits = int((candidate_counts >= int(config["crossmatch"]["max_neighbors"])).sum())
        if limit_hits:
            raise RuntimeError(
                f"{limit_hits} caption objects reached max_neighbors={config['crossmatch']['max_neighbors']}; "
                "candidate preservation may be truncated, so increase the cap and use a new run ID"
            )

        max_radius = float(max(config["crossmatch"]["radii_arcsec"]))
        selected = select_nearest_valid(candidates, max_radius)
        selected.to_parquet(run_dir / "selected_matches.parquet", index=False)
        by_radius, by_survey, summary = summarize_matches(
            source, candidates, config["crossmatch"]["radii_arcsec"]
        )
        by_radius.to_csv(run_dir / "counts_by_radius.csv", index=False)
        by_survey.to_csv(run_dir / "counts_by_survey.csv", index=False)
        write_json(
            run_dir / "summary.json",
            {
                **summary,
                "run_id": config["run"]["id"],
                "captioned_source_run_id": config["captioned_source"]["run_id"],
                "captioned_source_fingerprint": source_fingerprint(source),
                "desi_repo_id": config["desi_catalog"]["repo_id"],
                "desi_revision": config["desi_catalog"]["revision"],
                "radii_arcsec": [float(value) for value in config["crossmatch"]["radii_arcsec"]],
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
    print(f"Crossmatch probe complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
