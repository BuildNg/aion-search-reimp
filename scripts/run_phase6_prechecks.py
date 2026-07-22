#!/usr/bin/env python3
"""Run the two cheap checks before morphology-by-redshift retrieval."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from aion_reimp.artifacts import initialize_run, tracked_run, write_json
from aion_reimp.utils import file_digest
from spec_probes.morphology_coverage import (
    GALAXY_ZOO_COLUMNS,
    MORPHOLOGY_RULES,
    add_reliable_morphology_labels,
    match_galaxy_zoo,
    morphology_coverage_tables,
)
from spec_probes.paired import paired_redshift_readout, run_paired_redshift_comparison
from spec_probes.run_probes import embeddings_fingerprint


CONFIG_PATH = Path("configs/phase6_prechecks.yaml")


def load_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("kind") != "phase6_prechecks":
        raise ValueError(f"{path} is not a phase6_prechecks config")
    for section in ("run", "inputs", "alpha_sensitivity", "galaxy_zoo"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing config section: {section}")
    return config


def _load_manifest(config: Dict[str, Any]) -> pd.DataFrame:
    path = Path(config["inputs"]["paired_manifest"])
    if not path.exists():
        raise FileNotFoundError(f"Locked paired manifest is missing: {path}")
    manifest = pd.read_parquet(path)
    expected = int(config["inputs"]["expected_pairs"])
    if len(manifest) != expected:
        raise ValueError(f"Expected {expected} paired rows, found {len(manifest)}")
    if manifest["object_id"].astype(str).duplicated().any():
        raise ValueError("Paired manifest object_id values must be unique")
    return manifest.sort_values("object_id").reset_index(drop=True)


def run_alpha_sensitivity(config: Dict[str, Any]) -> None:
    source_dir = Path(config["inputs"]["source_run_dir"])
    source_config = yaml.safe_load((source_dir / "config.yaml").read_text(encoding="utf-8"))
    source_metrics = yaml.safe_load((source_dir / "metrics.json").read_text(encoding="utf-8"))
    source_fingerprints = yaml.safe_load(
        (source_dir / "embedding_fingerprints.json").read_text(encoding="utf-8")
    )
    manifest = _load_manifest(config)
    with np.load(source_dir / "embeddings.npz", allow_pickle=False) as cached:
        cached_ids = cached["object_id"].astype(str)
        image = np.asarray(cached["image"], dtype=np.float32)
        spectrum = np.asarray(cached["spectrum"], dtype=np.float32)
    object_ids = manifest["object_id"].astype(str).to_numpy()
    if not np.array_equal(cached_ids, object_ids):
        raise ValueError("Cached embedding IDs do not match the locked manifest order")
    fusion = np.concatenate([image, spectrum], axis=1)
    embeddings = {
        "image_only": image,
        "spectrum_only": spectrum,
        "image_plus_spectrum": fusion,
    }
    revisions = {}
    for condition, values in embeddings.items():
        metadata = source_fingerprints[condition]
        if embeddings_fingerprint(object_ids, values) != metadata["fingerprint"]:
            raise ValueError(f"Cached {condition} embedding fingerprint changed")
        revisions[condition] = metadata["revision"]

    alpha_grid = config["alpha_sensitivity"]["alpha_grid"]
    predictions, splits = run_paired_redshift_comparison(
        manifest,
        embeddings,
        revisions,
        split_seeds=source_config["split"]["seeds"],
        train_ratio=float(source_config["split"]["train_ratio"]),
        cv_folds=int(source_config["ridge"]["cv_folds"]),
        ridge_alpha_grid=alpha_grid,
        seed=int(source_config["run"]["seed"]),
    )
    metrics, tables, comparisons, alpha_selection = paired_redshift_readout(
        predictions,
        alpha_grid=alpha_grid,
        outlier_threshold=float(source_config["metrics"]["catastrophic_outlier_threshold"]),
        bootstrap_resamples=int(source_config["metrics"]["paired_bootstrap_resamples"]),
        seed=int(source_config["run"]["seed"]),
    )

    run_config = {**config, "selected_check": "alpha_sensitivity"}
    output_dir = initialize_run(
        Path(config["run"]["output_root"]),
        config["alpha_sensitivity"]["run_id"],
        run_config,
        shlex.join(sys.argv),
    )
    with tracked_run(output_dir, {"phase": 6, "condition": "cached_alpha_sensitivity"}):
        predictions.to_parquet(output_dir / "predictions.parquet", index=False)
        splits.to_parquet(output_dir / "split_assignments.parquet", index=False)
        write_json(output_dir / "metrics.json", metrics)
        tables.to_csv(output_dir / "tables.csv", index=False)
        comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
        alpha_selection.to_csv(output_dir / "alpha_selection.csv", index=False)
        source_primary = source_metrics["paired_bootstrap"]["fusion_vs_spectrum"]["one_plus_z"]
        new_primary = metrics["paired_bootstrap"]["fusion_vs_spectrum"]["one_plus_z"]
        write_json(
            output_dir / "sensitivity_summary.json",
            {
                "source_run": str(source_dir),
                "source_alpha_grid": source_config["ridge"]["alpha_grid"],
                "sensitivity_alpha_grid": alpha_grid,
                "selected_at_new_grid_max": int(alpha_selection["at_grid_max"].sum()),
                "fusion_vs_spectrum_primary_before": source_primary,
                "fusion_vs_spectrum_primary_after": new_primary,
                "error_improvement_change": float(
                    new_primary["error_improvement"] - source_primary["error_improvement"]
                ),
            },
        )


def run_galaxy_zoo_audit(config: Dict[str, Any]) -> None:
    manifest = _load_manifest(config)
    spec = config["galaxy_zoo"]
    catalog_path = Path(spec["catalog_path"])
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Galaxy Zoo DESI catalog is missing: {catalog_path}. Download the pinned file from "
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
    matches = match_galaxy_zoo(
        manifest,
        catalog,
        radius_arcsec=float(spec["match_radius_arcsec"]),
    )
    labelled = add_reliable_morphology_labels(
        matches,
        fraction_threshold=float(spec["reliable_fraction_threshold"]),
    )
    counts, crosstab, summary = morphology_coverage_tables(
        labelled,
        total_pairs=len(manifest),
        fraction_threshold=float(spec["reliable_fraction_threshold"]),
        redshift_bin_edges=spec["redshift_bin_edges"],
    )
    summary.update(
        {
            "catalog_source": spec["source_url"],
            "catalog_md5": actual_md5,
            "catalog_rows": int(len(catalog)),
            "match_radius_arcsec": float(spec["match_radius_arcsec"]),
            "morphology_rules": MORPHOLOGY_RULES,
            "label_contract": (
                "Galaxy Zoo DESI friendly predicted vote fractions; child labels require "
                "every parent branch to pass the same confidence threshold"
            ),
        }
    )

    run_config = {**config, "selected_check": "galaxy_zoo"}
    output_dir = initialize_run(
        Path(config["run"]["output_root"]),
        spec["run_id"],
        run_config,
        shlex.join(sys.argv),
    )
    with tracked_run(output_dir, {"phase": 6, "condition": "galaxy_zoo_coverage"}):
        labelled.to_parquet(output_dir / "matched_labels.parquet", index=False)
        manifest.loc[~manifest["object_id"].isin(labelled["object_id"])].to_csv(
            output_dir / "unmatched_pairs.csv", index=False
        )
        counts.to_csv(output_dir / "morphology_counts.csv", index=False)
        crosstab.to_csv(output_dir / "redshift_crosstab.csv", index=False)
        write_json(output_dir / "summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "check", choices=("alpha-sensitivity", "galaxy-zoo"), help="pre-check to run"
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check == "alpha-sensitivity":
        run_alpha_sensitivity(config)
    else:
        run_galaxy_zoo_audit(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
