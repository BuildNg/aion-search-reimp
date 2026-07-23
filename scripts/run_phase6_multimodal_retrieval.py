#!/usr/bin/env python3
"""Prepare and run the locked Phase-6 structured retrieval pilot."""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from aion_reimp.artifacts import initialize_run, tracked_run, write_json
from aion_reimp.manifest import manifest_fingerprint
from aion_reimp.multimodal_retrieval import (
    assemble_embeddings,
    query_targets,
    run_cached_distance_retrieval,
    run_joint_retrieval,
)
from spec_probes.encoders import build_encoder, resolve_device
from spec_probes.morphology_coverage import add_reliable_morphology_labels
from spec_probes.paired import (
    build_paired_manifest,
    load_image_embeddings,
    reorder_spectrum_batch,
)
from spec_probes.run_probes import embeddings_fingerprint
from spec_probes.spectra_data import (
    extract_spectrum_batch,
    load_hats_target_spectra,
    load_spectrum_batch,
    save_spectrum_batch,
)


CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "configs/phase6_multimodal_retrieval.yaml"))


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("kind") != "phase6_multimodal_retrieval":
        raise ValueError(f"{path} is not a phase6_multimodal_retrieval config")
    for section in (
        "run", "inputs", "image_source", "spectrum_source", "spectrum_encoder",
        "split", "heads", "retrieval",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing config section: {section}")
    retrieval = config["retrieval"]
    for key in ("joint_queries", "distance_queries"):
        if not isinstance(retrieval.get(key), list) or not retrieval[key]:
            raise ValueError(f"retrieval.{key} must be a non-empty list")
    if config["inputs"].get("candidate_policy") != "galaxy_zoo_matched_only":
        raise ValueError("Unknown-morphology pairs must not enter the retrieval benchmark")
    return config


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _joint_manifest(config: Dict[str, Any]) -> pd.DataFrame:
    inputs = config["inputs"]
    matches = pd.read_parquet(inputs["selected_matches"])
    if len(matches) != int(inputs["expected_pairs"]):
        raise ValueError(f"Expected {inputs['expected_pairs']} selected pairs, found {len(matches)}")
    manifest = build_paired_manifest(matches)
    selection = matches.set_index(matches["source_object_id"].astype(str))["selection_reason"]
    manifest["selection_reason"] = manifest["object_id"].map(selection)
    if manifest["selection_reason"].isna().any():
        raise ValueError("Selected pairs are missing selection provenance")

    labels = pd.read_parquet(inputs["morphology_labels"])
    if len(labels) != int(inputs["expected_labelled_pairs"]):
        raise ValueError(
            f"Expected {inputs['expected_labelled_pairs']} Galaxy Zoo-labelled pairs, found {len(labels)}"
        )
    if labels["object_id"].astype(str).duplicated().any():
        raise ValueError("Morphology labels contain duplicate object IDs")
    selected_by_id = manifest.set_index("object_id")
    label_ids = labels["object_id"].astype(str)
    if not set(label_ids).issubset(set(selected_by_id.index)):
        raise ValueError("Morphology labels contain objects outside the selected pairs")
    selected_rows = selected_by_id.loc[label_ids]
    if not np.allclose(
        selected_rows["z"].to_numpy(dtype=float),
        labels["z"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("Morphology-label redshifts disagree with the selected pairs")
    if selected_rows["selection_reason"].astype(str).tolist() != labels["selection_reason"].astype(str).tolist():
        raise ValueError("Morphology-label selection provenance disagrees with the selected pairs")
    labelled = add_reliable_morphology_labels(
        labels,
        fraction_threshold=float(config["retrieval"]["morphology_threshold"]),
    )
    for column in (
        "reliable_featured_or_disk", "reliable_spiral",
        "reliable_barred_spiral", "reliable_edge_on_disk",
    ):
        if not labelled[column].equals(labels[column].astype(bool)):
            raise ValueError(f"Saved morphology flag disagrees with the locked threshold: {column}")

    label_columns = [
        "object_id", "galaxy_zoo_dr8_id",
        "smooth-or-featured_smooth_fraction",
        "smooth-or-featured_featured-or-disk_fraction",
        "disk-edge-on_yes_fraction", "disk-edge-on_no_fraction",
        "has-spiral-arms_yes_fraction", "bar_strong_fraction", "bar_weak_fraction",
        "reliable_smooth", "reliable_featured_or_disk", "reliable_spiral",
        "reliable_barred_spiral", "reliable_edge_on_disk",
    ]
    labelled = labelled.loc[:, label_columns].copy()
    labelled["object_id"] = labelled["object_id"].astype(str)
    manifest = manifest.merge(labelled, on="object_id", how="inner", validate="one_to_one")
    if len(manifest) != int(inputs["expected_labelled_pairs"]):
        raise ValueError("Galaxy Zoo labels are not an exact subset of the selected pairs")
    for query in config["retrieval"]["joint_queries"]:
        query_targets(
            manifest,
            query,
            morphology_threshold=float(config["retrieval"]["morphology_threshold"]),
        )
    return manifest.sort_values("object_id").reset_index(drop=True)


def _spectrum_fields(source: Dict[str, Any]) -> list[str]:
    return [source["flux_field"], source["wave_field"], source["ivar_field"], source["mask_field"]]


def _base_embeddings(config: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(config["inputs"]["base_run_dir"]) / "embeddings.npz"
    with np.load(path, allow_pickle=False) as payload:
        object_ids = payload["object_id"].astype(str)
        image = payload["image"].astype(np.float32)
        spectrum = payload["spectrum"].astype(np.float32)
    expected = int(config["inputs"]["expected_base_pairs"])
    if len(object_ids) != expected or image.shape != (expected, 768) or spectrum.shape != (expected, 768):
        raise ValueError("Base paired embedding cache has the wrong shape")
    if len(set(object_ids)) != expected:
        raise ValueError("Base paired embedding cache has duplicate object IDs")
    return object_ids, image, spectrum


def prepare(config: Dict[str, Any]) -> None:
    data_dir = Path(config["run"]["data_dir"])
    summary_path = data_dir / "summary.json"
    if summary_path.exists():
        print(f"Multimodal retrieval data already complete: {data_dir}")
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = _joint_manifest(config)
    base_ids, _, _ = _base_embeddings(config)
    new = manifest.loc[~manifest["object_id"].isin(set(base_ids))].copy()
    expected_new = int(config["inputs"]["expected_new_embeddings"])
    if len(new) != expected_new or not new["selection_reason"].eq("morphology_priority").all():
        raise ValueError(f"Expected {expected_new} new morphology-priority embeddings, found {len(new)}")
    manifest.to_parquet(data_dir / "joint_manifest.parquet", index=False)

    new_image = load_image_embeddings(config["image_source"], new)
    np.save(data_dir / "new_image_embeddings.npy", new_image, allow_pickle=False)
    source = config["spectrum_source"]
    spectra, assignments, io_summary = load_hats_target_spectra(
        source["repo_id"], source["revision"], source["catalog_path"],
        new, data_dir / "parts",
        target_id_column="spectrum_object_id", target_ra_column="spectrum_ra",
        target_dec_column="spectrum_dec", object_id_column=source["object_id_column"],
        spectrum_column=source["spectrum_column"], redshift_column=source["redshift_column"],
        zwarn_column=source["zwarn_column"], spectrum_fields=_spectrum_fields(source),
    )
    batch = extract_spectrum_batch(
        spectra, source["object_id_column"], source["spectrum_column"],
        source["flux_field"], source["wave_field"], source["ivar_field"], source["mask_field"],
    )
    batch = reorder_spectrum_batch(batch, new["spectrum_object_id"].astype(str).tolist())
    source_z = spectra.set_index(source["object_id_column"]).loc[
        new["spectrum_object_id"].astype(str), source["redshift_column"]
    ].to_numpy(dtype=float)
    if not np.allclose(source_z, new["z"].to_numpy(dtype=float), rtol=0.0, atol=1e-6):
        raise ValueError("Extension spectrum redshifts disagree with the locked matches")
    if spectra[source["zwarn_column"]].astype(bool).any():
        raise ValueError("Prepared extension spectra include non-zero HATS ZWARN")
    assignments.to_parquet(data_dir / "partition_assignments.parquet", index=False)
    save_spectrum_batch(data_dir / "new_spectrum_batch.npz", batch)
    write_json(
        summary_path,
        {
            "candidate_policy": config["inputs"]["candidate_policy"],
            "selected_pairs": int(config["inputs"]["expected_pairs"]),
            "labelled_candidates": int(len(manifest)),
            "unknown_morphology_excluded": int(config["inputs"]["expected_pairs"] - len(manifest)),
            "base_candidates_reused": int(len(manifest) - len(new)),
            "new_candidates_encoded": int(len(new)),
            "manifest_fingerprint": manifest_fingerprint(manifest),
            "selection_reason_counts": {
                str(key): int(value)
                for key, value in manifest["selection_reason"].value_counts().items()
            },
            "partition_io": io_summary,
        },
    )


def run_distance(config: Dict[str, Any]) -> None:
    source = Path(config["inputs"]["base_run_dir"]) / "predictions.parquet"
    predictions = pd.read_parquet(source)
    if predictions["object_id"].nunique() != int(config["inputs"]["expected_base_pairs"]):
        raise ValueError("Cached distance predictions do not cover the locked base population")
    ranked, metrics, tables = run_cached_distance_retrieval(
        predictions,
        config["retrieval"]["distance_queries"],
        seed=int(config["run"]["seed"]),
        k=int(config["retrieval"]["k"]),
    )
    output = initialize_run(
        Path(config["run"]["output_root"]), config["run"]["distance_id"],
        {**config, "selected_run": "distance"}, shlex.join(sys.argv),
    )
    with tracked_run(output, {"phase": 6, "condition": "distance_retrieval"}):
        ranked.to_parquet(output / "ranked_rows.parquet", index=False)
        metrics.to_csv(output / "metrics_by_seed.csv", index=False)
        tables.to_csv(output / "tables.csv", index=False)
        ranked.loc[ranked["rank"].le(10)].to_csv(output / "top10.csv", index=False)
        write_json(
            output / "metrics.json",
            {
                "metric_contract": {
                    "primary": config["retrieval"]["primary_metric"],
                    "secondary": config["retrieval"]["secondary_metrics"],
                },
                "per_seed": _records(metrics),
                "aggregate": _records(tables),
            },
        )


def run_joint(config: Dict[str, Any]) -> None:
    data_dir = Path(config["run"]["data_dir"])
    summary = yaml.safe_load((data_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = pd.read_parquet(data_dir / "joint_manifest.parquet")
    if manifest_fingerprint(manifest) != summary["manifest_fingerprint"]:
        raise ValueError("Prepared joint manifest fingerprint changed")
    base_ids, base_image, base_spectrum = _base_embeddings(config)
    new_image = np.load(data_dir / "new_image_embeddings.npy", allow_pickle=False)
    batch = load_spectrum_batch(data_dir / "new_spectrum_batch.npz")
    new_ids = manifest.loc[~manifest["object_id"].isin(set(base_ids)), "object_id"].astype(str).tolist()

    output = initialize_run(
        Path(config["run"]["output_root"]), config["run"]["joint_id"],
        {**config, "selected_run": "joint"}, shlex.join(sys.argv),
    )
    with tracked_run(output, {"phase": 6, "condition": "multimodal_retrieval"}):
        device = resolve_device(config["run"]["device"])
        encoder = build_encoder(config["spectrum_encoder"], device=device)
        new_spectrum = encoder.embed(batch)
        image = assemble_embeddings(manifest["object_id"], base_ids, base_image, new_ids, new_image)
        spectrum = assemble_embeddings(
            manifest["object_id"], base_ids, base_spectrum, new_ids, new_spectrum
        )
        ranked, metrics, tables, comparisons, heads, splits = run_joint_retrieval(
            manifest, image, spectrum, config["retrieval"]["joint_queries"],
            split_seeds=config["split"]["seeds"], train_ratio=float(config["split"]["train_ratio"]),
            cv_folds=int(config["heads"]["cv_folds"]), alpha_grid=config["heads"]["alpha_grid"],
            seed=int(config["run"]["seed"]), k=int(config["retrieval"]["k"]),
            morphology_threshold=float(config["retrieval"]["morphology_threshold"]),
        )
        ranked.to_parquet(output / "ranked_rows.parquet", index=False)
        metrics.to_csv(output / "metrics_by_seed.csv", index=False)
        tables.to_csv(output / "tables.csv", index=False)
        comparisons.to_csv(output / "comparisons.csv", index=False)
        heads.to_csv(output / "head_selection.csv", index=False)
        splits.to_parquet(output / "split_assignments.parquet", index=False)
        ranked.loc[ranked["rank"].le(10)].to_csv(output / "top10.csv", index=False)
        np.savez_compressed(
            output / "embeddings.npz",
            object_id=manifest["object_id"].to_numpy(dtype=str),
            image=image,
            spectrum=spectrum,
        )
        write_json(
            output / "metrics.json",
            {
                "metric_contract": {
                    "primary": config["retrieval"]["primary_metric"],
                    "secondary": config["retrieval"]["secondary_metrics"],
                    "recall_interpretation": (
                        "Recall@k is secondary and bounded by min(k, positives) / positives; "
                        "use recall_at_k_ceiling to interpret its absolute value"
                    ),
                },
                "per_seed": _records(metrics),
                "aggregate": _records(tables),
                "comparisons": _records(comparisons),
            },
        )
        write_json(
            output / "embedding_fingerprints.json",
            {
                "image": {"shape": list(image.shape), "fingerprint": embeddings_fingerprint(manifest["object_id"], image)},
                "spectrum": {"shape": list(spectrum.shape), "fingerprint": embeddings_fingerprint(manifest["object_id"], spectrum), "revision": encoder.revision},
            },
        )


def main() -> int:
    config = load_config()
    if len(sys.argv) != 2 or sys.argv[1] not in {"distance", "prepare", "joint"}:
        raise SystemExit("usage: run_phase6_multimodal_retrieval.py {distance|prepare|joint}")
    if sys.argv[1] == "distance":
        run_distance(config)
    elif sys.argv[1] == "prepare":
        prepare(config)
    else:
        run_joint(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
