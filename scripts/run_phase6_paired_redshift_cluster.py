#!/usr/bin/env python3
"""Prepare and analyze the matched HSC-DESI redshift experiment."""

from __future__ import annotations

import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from aion_reimp.artifacts import initialize_run, tracked_run, write_json
from aion_reimp.datasets import load_pinned_dataset
from aion_reimp.manifest import manifest_fingerprint
from spec_probes.encoders import SpectrumBatch, build_encoder, resolve_device
from spec_probes.paired import (
    BOOTSTRAP_SCALES,
    build_paired_manifest,
    paired_error_bootstrap,
    run_paired_redshift_comparison,
)
from spec_probes.run_probes import (
    aggregate_seed_metrics,
    embeddings_fingerprint,
    metrics_from_predictions,
    tables_from_metrics,
)
from spec_probes.spectra_data import (
    assert_spectrum_value_binding,
    assign_hats_partitions,
    extract_spectrum_batch,
    find_stream_target_ids,
    hats_partition_path,
    load_hats_target_spectra,
    load_spectrum_batch,
    save_spectrum_batch,
    select_target_spectra,
)


CONFIG_PATH = Path("configs/phase6_paired_redshift.yaml")


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("kind") != "phase6_paired_redshift":
        raise ValueError(f"{path} is not a phase6_paired_redshift config")
    for section in (
        "run", "crossmatch", "image_source", "spectrum_source",
        "spectrum_encoder", "split", "ridge", "metrics",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing config section: {section}")
    return config


def _image_embeddings(config: Dict[str, Any], manifest: pd.DataFrame) -> np.ndarray:
    source = config["image_source"]
    dataset = load_pinned_dataset(source["repo_id"], source["revision"], source["split"])
    dataset = dataset.select_columns([source["object_id_column"], source["embedding_column"]])
    rows = dataset.select(manifest["source_row_id"].astype(int).tolist())
    object_ids = [str(row[source["object_id_column"]]) for row in rows]
    if object_ids != manifest["image_object_id"].tolist():
        raise ValueError("Image source rows do not align with the paired manifest")
    values = np.stack(
        [np.asarray(row[source["embedding_column"]], dtype=np.float32) for row in rows]
    )
    if values.shape != (len(manifest), 768):
        raise ValueError(f"Expected {(len(manifest), 768)} image embeddings, got {values.shape}")
    return values


def _reorder_batch(batch: SpectrumBatch, ordered_ids: list[str]) -> SpectrumBatch:
    row_by_id = {str(object_id): index for index, object_id in enumerate(batch.object_id)}
    indices = np.asarray([row_by_id[object_id] for object_id in ordered_ids], dtype=np.int64)
    return SpectrumBatch(
        object_id=np.asarray(batch.object_id)[indices],
        flux=batch.flux[indices],
        wave=batch.wave,
        ivar=batch.ivar[indices] if batch.ivar is not None else None,
        mask=batch.mask[indices] if batch.mask is not None else None,
    )


def _paired_manifest(config: Dict[str, Any]) -> pd.DataFrame:
    matches = pd.read_parquet(config["crossmatch"]["selected_matches"])
    manifest = build_paired_manifest(matches)
    expected = int(config["crossmatch"]["expected_pairs"])
    if len(manifest) != expected:
        raise ValueError(f"Expected {expected} locked pairs, found {len(manifest)}")
    return manifest


def _spectrum_fields(source: Dict[str, Any]) -> list[str]:
    return [
        source["flux_field"], source["wave_field"],
        source["ivar_field"], source["mask_field"],
    ]


def _redshift_distribution(z_values: np.ndarray) -> Dict[str, Any]:
    z = np.asarray(z_values, dtype=float)
    quantile_levels = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    quantiles = np.quantile(z, quantile_levels)
    counts, edges = np.histogram(z, bins=20)
    return {
        "n": int(len(z)),
        "quantiles": {
            f"p{int(level * 100):02d}": float(value)
            for level, value in zip(quantile_levels, quantiles)
        },
        "histogram": {
            "bin_edges": [float(value) for value in edges],
            "counts": [int(value) for value in counts],
        },
        "interpretation": (
            "HSC x DESI SV3 selected subset; these redshift and image-only results "
            "do not estimate performance on the full 18k HSC population"
        ),
    }


def run_preflight(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check the exact-ID HATS path, one cache round trip, and value binding.

    The binding step fetches one TARGETID shared between the HATS conversion
    and the probe-validated MultimodalUniverse/desi table and requires its
    flux/ivar/mask/wavelength values to agree, so the encoder consumes
    spectra verified against the source the Phase-6 encoder gate ran on.
    """
    report: Dict[str, Any] = {"status": "pass"}
    try:
        from huggingface_hub import HfFileSystem

        manifest = _paired_manifest(config)
        source = config["spectrum_source"]
        binding = source["value_binding"]
        fs = HfFileSystem()
        repo_root = f"datasets/{source['repo_id']}@{source['revision']}"
        with fs.open(
            f"{repo_root}/{source['catalog_path']}/partition_info.csv", "rb"
        ) as handle:
            partition_info = pd.read_csv(handle)
        assignments = assign_hats_partitions(
            manifest,
            partition_info,
            ra_column="spectrum_ra",
            dec_column="spectrum_dec",
        )
        unique_parts = assignments[["hats_order", "hats_pixel"]].drop_duplicates()
        remote_bytes = 0
        for row in unique_parts.itertuples(index=False):
            relative = hats_partition_path(source["catalog_path"], row.hats_order, row.hats_pixel)
            remote_bytes += int(fs.info(f"{repo_root}/{relative}")["size"])

        overlap_ids = find_stream_target_ids(
            binding["repo_id"], binding["revision"], binding["split"],
            manifest["spectrum_object_id"].tolist(),
            source["object_id_column"],
        )
        if not overlap_ids:
            raise ValueError(
                "No matched TARGETID appears in the probe-validated binding table; "
                "value binding requires a manual one-object comparison before launch"
            )
        bound_id = overlap_ids[0]
        reference_rows = select_target_spectra(
            binding["repo_id"], binding["revision"], binding["split"],
            [bound_id],
            object_id_column=source["object_id_column"],
            spectrum_column=source["spectrum_column"],
            redshift_column=source["redshift_column"],
            zwarn_column=source["zwarn_column"],
        )
        reference_batch = extract_spectrum_batch(
            reference_rows,
            source["object_id_column"], source["spectrum_column"],
            source["flux_field"], source["wave_field"],
            source["ivar_field"], source["mask_field"],
        )

        with tempfile.TemporaryDirectory(prefix="phase6-paired-preflight-") as temporary:
            one = manifest.loc[manifest["spectrum_object_id"].eq(bound_id)]
            spectra, _, _ = load_hats_target_spectra(
                source["repo_id"], source["revision"], source["catalog_path"],
                one, Path(temporary) / "parts",
                target_id_column="spectrum_object_id",
                target_ra_column="spectrum_ra",
                target_dec_column="spectrum_dec",
                object_id_column=source["object_id_column"],
                spectrum_column=source["spectrum_column"],
                redshift_column=source["redshift_column"],
                zwarn_column=source["zwarn_column"],
                spectrum_fields=_spectrum_fields(source),
            )
            batch = extract_spectrum_batch(
                spectra,
                source["object_id_column"], source["spectrum_column"],
                source["flux_field"], source["wave_field"],
                source["ivar_field"], source["mask_field"],
            )
            binding_report = assert_spectrum_value_binding(batch, reference_batch, bound_id)
            reference_z = float(reference_rows.iloc[0][source["redshift_column"]])
            manifest_z = float(one.iloc[0]["z"])
            if abs(reference_z - manifest_z) > 1e-6:
                raise ValueError(
                    f"Binding object {bound_id!r} redshift disagrees: "
                    f"binding table {reference_z!r} vs locked match {manifest_z!r}"
                )
            cache_path = Path(temporary) / "one_spectrum.npz"
            save_spectrum_batch(cache_path, batch)
            loaded = load_spectrum_batch(cache_path)
            if loaded.flux.shape != batch.flux.shape:
                raise ValueError("Spectrum cache round trip changed the flux shape")
        first = spectra.iloc[0]
        if bool(first[source["zwarn_column"]]) is not False:
            raise ValueError("Preflight target is not ZWARN-good in the HATS source")
        report.update(
            {
                "pairs": int(len(manifest)),
                "manifest_fingerprint": manifest_fingerprint(manifest),
                "spectrum_repo_id": source["repo_id"],
                "spectrum_revision": source["revision"],
                "spectrum_catalog_path": source["catalog_path"],
                "target_partitions": int(len(unique_parts)),
                "target_partition_bytes": int(remote_bytes),
                "binding_repo_id": binding["repo_id"],
                "binding_revision": binding["revision"],
                "binding_overlap_count": int(len(overlap_ids)),
                "bound_target_id": str(bound_id),
                "value_binding": binding_report,
                "first_flux_shape": list(batch.flux.shape),
                "first_wave_shape": list(batch.wave.shape),
                "cache_round_trip": True,
            }
        )
    except Exception as error:  # noqa: BLE001 - preflight must preserve the failure
        report = {"status": "fail", "error_type": type(error).__name__, "error": str(error)}
    return report


def _require_preflight(config: Dict[str, Any]) -> None:
    path = Path(config["run"]["preflight_report"])
    if not path.exists():
        raise FileNotFoundError(f"Run paired preflight first: {path}")
    report = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest = _paired_manifest(config)
    source = config["spectrum_source"]
    binding = source["value_binding"]
    if (
        report.get("status") != "pass"
        or int(report.get("pairs", -1)) != int(config["crossmatch"]["expected_pairs"])
        or report.get("manifest_fingerprint") != manifest_fingerprint(manifest)
        or report.get("spectrum_repo_id") != source["repo_id"]
        or report.get("spectrum_revision") != source["revision"]
        or report.get("spectrum_catalog_path") != source["catalog_path"]
        or report.get("binding_repo_id") != binding["repo_id"]
        or report.get("binding_revision") != binding["revision"]
        or not report.get("bound_target_id")
    ):
        raise RuntimeError(f"Paired preflight is missing, failed, or stale: {path}")


def prepare(config: Dict[str, Any]) -> None:
    _require_preflight(config)
    data_dir = Path(config["run"]["data_dir"])
    summary_path = data_dir / "summary.json"
    if summary_path.exists():
        print(f"Paired data already complete: {data_dir}")
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = _paired_manifest(config)
    manifest_path = data_dir / "paired_manifest.parquet"
    if manifest_path.exists():
        cached_manifest = pd.read_parquet(manifest_path)
        if manifest_fingerprint(cached_manifest) != manifest_fingerprint(manifest):
            raise ValueError("Cached paired manifest does not match the locked crossmatch")
    else:
        manifest.to_parquet(manifest_path, index=False)
    image_path = data_dir / "image_embeddings.npy"
    if image_path.exists():
        image_embeddings = np.load(image_path, allow_pickle=False)
    else:
        image_embeddings = _image_embeddings(config, manifest)
        np.save(image_path, image_embeddings, allow_pickle=False)
    if image_embeddings.shape != (len(manifest), 768):
        raise ValueError("Cached image embeddings have the wrong shape")

    source = config["spectrum_source"]
    spectra, assignments, io_summary = load_hats_target_spectra(
        source["repo_id"], source["revision"], source["catalog_path"],
        manifest, data_dir / "parts",
        target_id_column="spectrum_object_id",
        target_ra_column="spectrum_ra",
        target_dec_column="spectrum_dec",
        object_id_column=source["object_id_column"],
        spectrum_column=source["spectrum_column"],
        redshift_column=source["redshift_column"],
        zwarn_column=source["zwarn_column"],
        spectrum_fields=_spectrum_fields(source),
    )
    batch = extract_spectrum_batch(
        spectra,
        source["object_id_column"], source["spectrum_column"],
        source["flux_field"], source["wave_field"],
        source["ivar_field"], source["mask_field"],
    )
    batch = _reorder_batch(batch, manifest["spectrum_object_id"].tolist())
    streamed_z = spectra.set_index(source["object_id_column"]).loc[
        manifest["spectrum_object_id"], source["redshift_column"]
    ].to_numpy(dtype=float)
    if not np.allclose(streamed_z, manifest["z"].to_numpy(dtype=float), rtol=0.0, atol=1e-6):
        raise ValueError("Spectrum-source redshifts disagree with the locked HATS matches")
    if spectra[source["zwarn_column"]].astype(bool).any():
        raise ValueError("Prepared spectra include a non-zero HATS ZWARN value")
    assignments.to_parquet(data_dir / "partition_assignments.parquet", index=False)
    temporary_batch = data_dir / "spectrum_batch.tmp.npz"
    save_spectrum_batch(temporary_batch, batch)
    temporary_batch.replace(data_dir / "spectrum_batch.npz")
    write_json(
        summary_path,
        {
            "pairs": len(manifest),
            "manifest_fingerprint": manifest_fingerprint(manifest),
            "image_embedding_shape": list(image_embeddings.shape),
            "spectrum_flux_shape": list(batch.flux.shape),
            "partition_io": io_summary,
            "redshift_distribution": _redshift_distribution(manifest["z"].to_numpy()),
        },
    )


def analyze(config: Dict[str, Any]) -> None:
    run = config["run"]
    data_dir = Path(run["data_dir"])
    data_summary_path = data_dir / "summary.json"
    if not data_summary_path.exists():
        raise FileNotFoundError(f"Paired preparation is incomplete: {data_summary_path}")
    data_summary = yaml.safe_load(data_summary_path.read_text(encoding="utf-8"))
    manifest = pd.read_parquet(data_dir / "paired_manifest.parquet")
    image_embeddings = np.load(data_dir / "image_embeddings.npy", allow_pickle=False)
    batch = load_spectrum_batch(data_dir / "spectrum_batch.npz")
    if batch.object_id.astype(str).tolist() != manifest["spectrum_object_id"].tolist():
        raise ValueError("Cached spectrum order does not match the paired manifest")

    output_dir = initialize_run(
        Path(run["output_root"]), run["id"], config, shlex.join(sys.argv)
    )
    with tracked_run(output_dir, {"phase": 6, "condition": "paired_redshift"}):
        device = resolve_device(run["device"])
        encoder = build_encoder(config["spectrum_encoder"], device=device)
        spectrum_embeddings = encoder.embed(batch)
        fusion_embeddings = np.concatenate([image_embeddings, spectrum_embeddings], axis=1)
        embeddings = {
            "image_only": image_embeddings,
            "spectrum_only": spectrum_embeddings,
            "image_plus_spectrum": fusion_embeddings,
        }
        image_revision = f"{config['image_source']['repo_id']}@{config['image_source']['revision']}"
        spectrum_revision = f"{config['spectrum_encoder']['repo_id']}@{encoder.revision}"
        revisions = {
            "image_only": image_revision,
            "spectrum_only": spectrum_revision,
            "image_plus_spectrum": f"{image_revision}+{spectrum_revision}",
        }
        predictions, splits = run_paired_redshift_comparison(
            manifest, embeddings, revisions,
            split_seeds=config["split"]["seeds"],
            train_ratio=float(config["split"]["train_ratio"]),
            cv_folds=int(config["ridge"]["cv_folds"]),
            ridge_alpha_grid=config["ridge"]["alpha_grid"],
            seed=int(run["seed"]),
        )
        predictions.to_parquet(output_dir / "predictions.parquet", index=False)
        splits.to_parquet(output_dir / "split_assignments.parquet", index=False)
        np.savez_compressed(
            output_dir / "embeddings.npz",
            object_id=manifest["object_id"].to_numpy(dtype=str),
            image=image_embeddings,
            spectrum=spectrum_embeddings,
        )
        per_seed = metrics_from_predictions(
            predictions,
            outlier_threshold=float(config["metrics"]["catastrophic_outlier_threshold"]),
            spectype_classes=[],
        )
        aggregated = aggregate_seed_metrics(per_seed)
        comparisons = {
            "fusion_vs_image": ("image_only", "image_plus_spectrum"),
            "fusion_vs_spectrum": ("spectrum_only", "image_plus_spectrum"),
        }
        bootstrap: Dict[str, Dict[str, Any]] = {}
        bootstrap_rows = []
        for offset, (comparison, (baseline, condition)) in enumerate(comparisons.items()):
            bootstrap[comparison] = {}
            for scale_offset, scale in enumerate(BOOTSTRAP_SCALES):
                result = paired_error_bootstrap(
                    predictions, baseline, condition,
                    scale=scale,
                    n_resamples=int(config["metrics"]["paired_bootstrap_resamples"]),
                    seed=int(run["seed"]) + 2 * offset + scale_offset,
                )
                result["comparison"] = comparison
                result["is_primary"] = scale == "one_plus_z"
                bootstrap[comparison][scale] = result
                bootstrap_rows.append(result)
        write_json(
            output_dir / "metrics.json",
            {"per_seed": per_seed, "aggregated": aggregated, "paired_bootstrap": bootstrap},
        )
        tables_from_metrics(aggregated).to_csv(output_dir / "tables.csv", index=False)
        pd.DataFrame(bootstrap_rows).to_csv(output_dir / "paired_comparisons.csv", index=False)
        alpha_grid = config["ridge"]["alpha_grid"]
        alpha_selection = predictions.loc[
            predictions["probe_family"].eq("linear"),
            ["encoder", "split_seed", "hyperparameter_value"],
        ].drop_duplicates()
        alpha_selection["at_grid_min"] = alpha_selection["hyperparameter_value"].eq(min(alpha_grid))
        alpha_selection["at_grid_max"] = alpha_selection["hyperparameter_value"].eq(max(alpha_grid))
        alpha_selection.to_csv(output_dir / "alpha_selection.csv", index=False)
        write_json(output_dir / "redshift_distribution.json", data_summary["redshift_distribution"])
        write_json(
            output_dir / "embedding_fingerprints.json",
            {
                name: {
                    "shape": list(values.shape),
                    "fingerprint": embeddings_fingerprint(manifest["object_id"], values),
                    "revision": revisions[name],
                }
                for name, values in embeddings.items()
            },
        )


def main() -> int:
    config = load_config()
    if len(sys.argv) != 2 or sys.argv[1] not in {"preflight", "prepare", "analyze"}:
        raise SystemExit("usage: run_phase6_paired_redshift_cluster.py {preflight|prepare|analyze}")
    if sys.argv[1] == "preflight":
        report = run_preflight(config)
        report_path = Path(config["run"]["preflight_report"])
        write_json(report_path, report, overwrite=report_path.exists())
        print(yaml.safe_dump(report, sort_keys=False))
        return 0 if report["status"] == "pass" else 1
    if sys.argv[1] == "prepare":
        prepare(config)
    else:
        analyze(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
