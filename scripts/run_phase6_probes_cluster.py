"""Phase 6 frozen spectrum-encoder probe entrypoint.

Cluster-only: requires live network access for the Multimodal Universe DESI
streaming sample and the two pretrained encoder checkpoints. Never run on
the laptop. Mirrors the scripts/run_phase3_10k_cluster.py pattern: a thin
main() that resolves config, then delegates every scientific decision to
the spec_probes package (architecture.md decision 12: probes are a bounded
review instrument, not the retrieval/training pipeline).

Two-step launch (Finding 5 -- safe first-cluster-contact):

    python scripts/run_phase6_probes_cluster.py --preflight
    python scripts/run_phase6_probes_cluster.py

``--preflight`` streams and reports the first record's full schema, repeats
the seeded selection to verify ordered-ID stability, verifies both neural
model imports and checkpoint loads, and encodes enough real streamed rows
to cover PCA fitting and each configured neural batch size. It reports
output shapes, the resolved device, and free GPU memory -- all WITHOUT creating a
``results/<run_id>`` directory, so a failed first contact never consumes a
run ID. It writes exactly one ``preflight_report.json`` to the path named
by ``run.preflight_report`` in the config. The full run (no flag) refuses
to create its run directory unless that report exists and reports
``"status": "pass"`` for the exact current config, code hashes, package
versions, Python version, and device.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from aion_reimp.artifacts import initialize_run, tracked_run, write_json
from aion_reimp.manifest import manifest_fingerprint

from spec_probes.config import load_config
from spec_probes.encoders import SpectrumBatch, build_encoder, resolve_device
from spec_probes.probes import make_cv_folds
from spec_probes.run_probes import (
    aggregate_seed_metrics,
    embeddings_fingerprint,
    metrics_from_predictions,
    run_baseline_suite,
    run_probe_suite,
    run_probe_suite_for_encoder,
    tables_from_metrics,
)
from spec_probes.spectra_data import (
    assert_no_split_leakage,
    extract_labels,
    extract_spectrum_batch,
    inspect_source_columns,
    object_level_split,
    select_probe_sample,
    split_fingerprint,
    verify_required_columns,
)

PREFLIGHT_CONTRACT_VERSION = 2
PREFLIGHT_CODE_PATHS = (
    "scripts/run_phase6_probes_cluster.py",
    "src/spec_probes/config.py",
    "src/spec_probes/encoders.py",
    "src/spec_probes/probes.py",
    "src/spec_probes/run_probes.py",
    "src/spec_probes/spectra_data.py",
    "src/spec_probes/specformer_model.py",
)
PREFLIGHT_DISTRIBUTIONS = (
    "datasets",
    "huggingface-hub",
    "lightning",
    "numpy",
    "pandas",
    "polymathic-aion",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "torch",
)


def _canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _preflight_sample_size(config: Dict[str, Any]) -> int:
    """Cover PCA.fit and at least one full configured neural batch."""
    requirements = [2]
    for encoder_spec in config["encoders"]:
        if encoder_spec["kind"] == "pca_baseline":
            requirements.append(int(encoder_spec["n_components"]))
        else:
            requirements.append(int(encoder_spec["batch_size"]))
    return max(requirements)


def _installed_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for distribution in PREFLIGHT_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED"
    return versions


def _preflight_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    code_sha256 = {
        relative_path: hashlib.sha256((repo_root / relative_path).read_bytes()).hexdigest()
        for relative_path in PREFLIGHT_CODE_PATHS
    }
    declared_device = str(config["run"]["device"])
    return {
        "contract_version": PREFLIGHT_CONTRACT_VERSION,
        "config": config,
        "code_sha256": code_sha256,
        "python_version": platform.python_version(),
        "package_versions": _installed_versions(),
        "declared_device": declared_device,
        "resolved_device": resolve_device(declared_device),
    }


def _single_row_frame(row: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([row])


def _select_embedding_rows(
    embeddings: np.ndarray,
    row_by_object_id: Dict[str, int],
    object_ids: List[str],
) -> np.ndarray:
    """Select cached full-sample embeddings in an explicit object-ID order."""
    indices = np.asarray([row_by_object_id[object_id] for object_id in object_ids], dtype=np.int64)
    return np.asarray(embeddings)[indices]


def run_preflight(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run every first-cluster-contact check and return the report dict.

    Never creates ``results/<run_id>``. Any failure is captured in the
    report (``status: "fail"``) rather than raised, so the report is always
    written even when a check fails -- the caller decides whether to exit
    nonzero.
    """
    source = config["source_data"]
    try:
        contract = _preflight_contract(config)
    except Exception as error:  # noqa: BLE001
        return {
            "status": "fail",
            "checks": {"contract": {"ok": False, "error": f"{type(error).__name__}: {error}"}},
        }
    report: Dict[str, Any] = {
        "status": "pass",
        "contract": contract,
        "contract_fingerprint": _canonical_fingerprint(contract),
        "checks": {"contract": {"ok": True}},
    }

    try:
        column_report = inspect_source_columns(source["repo_id"], source["revision"], source["split"])
        verify_required_columns(
            column_report["columns"],
            source["object_id_column"],
            source["spectrum_column"],
            source["redshift_column"],
            source["zwarn_column"],
        )
        report["checks"]["schema"] = {"ok": True, **column_report}
    except Exception as error:  # noqa: BLE001 -- preflight must capture, not raise
        report["status"] = "fail"
        report["checks"]["schema"] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        return report

    import torch

    device = contract["resolved_device"]
    declared_device = contract["declared_device"]
    device_report: Dict[str, Any] = {
        "ok": device == declared_device,
        "declared_device": declared_device,
        "resolved_device": device,
    }
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        device_report.update(
            {
                "cuda_available": True,
                "free_gpu_memory_bytes": int(free_bytes),
                "total_gpu_memory_bytes": int(total_bytes),
            }
        )
    else:
        device_report["cuda_available"] = False
    report["checks"]["device"] = device_report
    if device != declared_device:
        report["status"] = "fail"
        device_report["error"] = (
            f"run.device requested {declared_device!r}, but the environment resolved {device!r}; "
            "preflight will not authorize a fallback device"
        )
        return report

    try:
        preflight_sample_size = _preflight_sample_size(config)
        sample = select_probe_sample(
            source["repo_id"],
            source["revision"],
            source["split"],
            preflight_sample_size,
            seed=int(config["run"]["seed"]),
            zwarn_column=source["zwarn_column"],
            zwarn_filter_value=int(config["labels"]["zwarn_filter_value"]),
        )
        batch = extract_spectrum_batch(
            sample,
            source["object_id_column"],
            source["spectrum_column"],
            source["flux_field"],
            source["wave_field"],
            source["ivar_field"],
            source["mask_field"],
        )
        repeated_sample = select_probe_sample(
            source["repo_id"],
            source["revision"],
            source["split"],
            preflight_sample_size,
            seed=int(config["run"]["seed"]),
            zwarn_column=source["zwarn_column"],
            zwarn_filter_value=int(config["labels"]["zwarn_filter_value"]),
        )
        object_id_column = source["object_id_column"]
        first_ids = sample[object_id_column].astype(str).tolist()
        repeated_ids = repeated_sample[object_id_column].astype(str).tolist()
        if first_ids != repeated_ids:
            raise RuntimeError(
                "Two fresh streaming selections with the same dataset revision and seed produced different ordered IDs"
            )
        report["checks"]["preflight_sample"] = {
            "ok": True,
            "rows": len(batch),
            "required_rows": preflight_sample_size,
            "flux_shape": list(batch.flux.shape),
            "wave_shape": list(batch.wave.shape),
            "ivar_shape": list(batch.ivar.shape) if batch.ivar is not None else None,
            "mask_shape": list(batch.mask.shape) if batch.mask is not None else None,
            "ordered_object_id_fingerprint": _canonical_fingerprint(first_ids),
            "repeat_selection_matches": True,
        }
    except Exception as error:  # noqa: BLE001
        report["status"] = "fail"
        report["checks"]["preflight_sample"] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        return report

    encoder_checks: Dict[str, Any] = {}
    for encoder_spec in config["encoders"]:
        name = encoder_spec["name"]
        try:
            encoder = build_encoder(encoder_spec, device=device)
            encoder.fit(batch)
            embeddings = encoder.embed(batch)
            encoder_checks[name] = {
                "ok": True,
                "loaded": True,
                "output_shape": list(embeddings.shape),
                "revision": encoder.revision,
            }
        except Exception as error:  # noqa: BLE001
            report["status"] = "fail"
            encoder_checks[name] = {"ok": False, "loaded": False, "error": f"{type(error).__name__}: {error}"}
    report["checks"]["encoders"] = encoder_checks

    return report


def _write_preflight_report(config: Dict[str, Any], report: Dict[str, Any]) -> Path:
    report_path = Path(config["run"]["preflight_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # write_json refuses to overwrite by default (architecture.md: retries
    # need an explicit new run ID / overwrite); a preflight re-run is
    # expected to replace the previous attempt's report, so overwrite here
    # is deliberate -- unlike results/<run_id>, "preflight/" is not itself
    # a versioned run artifact.
    write_json(report_path, report, overwrite=True)
    return report_path


def _require_passing_preflight(config: Dict[str, Any]) -> None:
    report_path = Path(config["run"]["preflight_report"])
    if not report_path.exists():
        raise RuntimeError(
            f"No preflight report at {report_path}. Run "
            "`python scripts/run_phase6_probes_cluster.py --preflight` first."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise RuntimeError(
            f"Preflight report at {report_path} does not report status 'pass' "
            f"(got {report.get('status')!r}); fix the reported failures and rerun --preflight."
        )
    expected_contract = _preflight_contract(config)
    expected_fingerprint = _canonical_fingerprint(expected_contract)
    if report.get("contract_fingerprint") != expected_fingerprint or report.get("contract") != expected_contract:
        raise RuntimeError(
            f"Preflight report at {report_path} was produced for a different config, code version, "
            "package environment, or device. Rerun --preflight before starting the full run."
        )


def run_full(config: Dict[str, Any]) -> None:
    _require_passing_preflight(config)

    source = config["source_data"]
    device = resolve_device(config["run"]["device"])
    spectral_class_enabled = bool(config["labels"]["spectral_class"]["enabled"])
    spectype_classes = sorted(config["labels"]["spectral_class"]["classes"]) if spectral_class_enabled else None

    # Run-directory creation happens only after config + preflight
    # validation (Finding 5).
    output_root = initialize_run(
        Path(config["run"]["output_root"]), config["run"]["id"], config, shlex.join(sys.argv)
    )

    with tracked_run(output_root, {"phase": 6, "condition": "spectrum_encoder_probes"}):
        column_report = inspect_source_columns(source["repo_id"], source["revision"], source["split"])
        write_json(output_root / "source_columns.json", column_report)
        verify_required_columns(
            column_report["columns"],
            source["object_id_column"],
            source["spectrum_column"],
            source["redshift_column"],
            source["zwarn_column"],
        )

        sample = select_probe_sample(
            source["repo_id"],
            source["revision"],
            source["split"],
            int(source["sample_size"]),
            seed=int(config["run"]["seed"]),
            zwarn_column=source["zwarn_column"],
            zwarn_filter_value=int(config["labels"]["zwarn_filter_value"]),
        )
        labels = extract_labels(sample, source["object_id_column"], source["redshift_column"], source["zwarn_column"])
        sample_order = pd.DataFrame(
            {
                "object_id": sample[source["object_id_column"]].astype(str).to_numpy(),
                "sample_position": np.arange(len(sample), dtype=np.int64),
            }
        )
        sample_manifest = sample_order.merge(labels, on="object_id", how="left", validate="one_to_one")
        if sample_manifest[["z", "zwarn"]].isna().any().any():
            raise ValueError("Sample manifest could not be joined one-to-one to every extracted label")
        sample_manifest.to_parquet(output_root / "sample_manifest.parquet", index=False)
        write_json(
            output_root / "sample_summary.json",
            {
                "rows": len(sample_manifest),
                "fingerprint": manifest_fingerprint(sample_manifest),
                "ordered_object_id_fingerprint": _canonical_fingerprint(sample_manifest["object_id"].tolist()),
            },
        )
        full_batch = extract_spectrum_batch(
            sample,
            source["object_id_column"],
            source["spectrum_column"],
            source["flux_field"],
            source["wave_field"],
            source["ivar_field"],
            source["mask_field"],
        )
        labels_by_id = labels.set_index("object_id")
        batch_by_id = {object_id: index for index, object_id in enumerate(full_batch.object_id)}

        def _select_batch(object_ids: List[str]) -> SpectrumBatch:
            indices = [batch_by_id[object_id] for object_id in object_ids]
            return SpectrumBatch(
                object_id=full_batch.object_id[indices],
                flux=full_batch.flux[indices],
                wave=full_batch.wave,
                ivar=full_batch.ivar[indices] if full_batch.ivar is not None else None,
                mask=full_batch.mask[indices] if full_batch.mask is not None else None,
            )

        predictions_frames: List[pd.DataFrame] = []
        embeddings_fingerprints: Dict[str, Any] = {}
        split_summaries: Dict[str, Any] = {}
        split_assignment_frames: List[pd.DataFrame] = []

        # The two pretrained encoders are frozen and split-independent. Embed
        # the 10k sample exactly once per neural encoder, retain only the CPU
        # arrays, and slice those arrays for every split seed. The previous
        # orchestration rebuilt each model and re-embedded train/test rows for
        # every seed (plus a second test pass for fingerprints), multiplying
        # the dominant inference cost without changing any scientific input.
        # PCA remains inside the split loop because its basis must be fit on
        # each outer training split only.
        frozen_neural_embeddings: Dict[str, Dict[str, Any]] = {}
        for encoder_spec in config["encoders"]:
            if encoder_spec["kind"] == "pca_baseline":
                continue
            encoder = build_encoder(encoder_spec, device=device)
            encoder.fit(full_batch)  # no-op for frozen neural encoders
            embeddings = encoder.embed(full_batch)
            if embeddings.shape != (len(full_batch), encoder.output_dim):
                raise ValueError(
                    f"{encoder.name} full-sample embeddings have shape {embeddings.shape}, "
                    f"expected {(len(full_batch), encoder.output_dim)}"
                )
            frozen_neural_embeddings[encoder.name] = {
                "embeddings": embeddings,
                "revision": encoder.revision,
                "output_dim": encoder.output_dim,
            }
            del encoder
            gc.collect()
            if device == "cuda":
                import torch

                torch.cuda.empty_cache()

        for split_seed in config["split"]["seeds"]:
            split = object_level_split(
                labels["object_id"], seed=int(split_seed), train_ratio=float(config["split"]["train_ratio"])
            )
            assert_no_split_leakage(split)
            split_fp = split_fingerprint(split)
            split_summaries[str(split_seed)] = {"fingerprint": split_fp, "rows": len(split)}
            split_with_seed = split.copy()
            split_with_seed.insert(0, "split_seed", int(split_seed))
            split_assignment_frames.append(split_with_seed)

            train_ids = split.loc[split["split"] == "train", "object_id"].sort_values().tolist()
            test_ids = split.loc[split["split"] == "test", "object_id"].sort_values().tolist()
            train_batch = _select_batch(train_ids)
            test_batch = _select_batch(test_ids)
            z_train = labels_by_id.loc[train_batch.object_id, "z"].to_numpy()
            z_test = labels_by_id.loc[test_batch.object_id, "z"].to_numpy()
            spectype_train = spectype_test = None  # spectral_class.enabled is always false today; see config.py

            cv_folds = make_cv_folds(len(train_ids), int(config["probes"]["cv_folds"]), seed=int(config["run"]["seed"]))

            predictions_frames.append(
                run_baseline_suite(
                    test_batch.object_id, z_train, z_test, int(split_seed), spectype_train, spectype_test
                )
            )

            for encoder_spec in config["encoders"]:
                if encoder_spec["kind"] == "pca_baseline":
                    encoder = build_encoder(encoder_spec, device=device)
                    predictions_frames.append(
                        run_probe_suite_for_encoder(
                            encoder,
                            train_batch,
                            test_batch,
                            z_train,
                            z_test,
                            config["probes"],
                            cv_folds,
                            int(split_seed),
                            seed=int(config["run"]["seed"]),
                            spectype_train=spectype_train,
                            spectype_test=spectype_test,
                            spectype_classes=spectype_classes,
                        )
                    )
                    embeddings_test = encoder.embed(test_batch)
                    encoder_name = encoder.name
                    encoder_revision = encoder.revision
                    output_dim = encoder.output_dim
                else:
                    encoder_name = encoder_spec["name"]
                    frozen = frozen_neural_embeddings[encoder_name]
                    embeddings_train = _select_embedding_rows(
                        frozen["embeddings"], batch_by_id, train_ids
                    )
                    embeddings_test = _select_embedding_rows(
                        frozen["embeddings"], batch_by_id, test_ids
                    )
                    encoder_revision = frozen["revision"]
                    output_dim = int(frozen["output_dim"])
                    predictions_frames.append(
                        run_probe_suite(
                            encoder_name,
                            encoder_revision,
                            embeddings_train,
                            embeddings_test,
                            test_ids,
                            z_train,
                            z_test,
                            config["probes"],
                            cv_folds,
                            int(split_seed),
                            seed=int(config["run"]["seed"]),
                            spectype_train=spectype_train,
                            spectype_test=spectype_test,
                            spectype_classes=spectype_classes,
                        )
                    )
                embeddings_fingerprints[f"{encoder_name}|{split_seed}"] = {
                    "fingerprint": embeddings_fingerprint(test_ids, embeddings_test),
                    "rows": int(embeddings_test.shape[0]),
                    "output_dim": output_dim,
                    "revision": encoder_revision,
                    "embedding_reuse": (
                        "full_sample_frozen_cache" if encoder_spec["kind"] != "pca_baseline" else "split_fitted_pca"
                    ),
                }

        write_json(output_root / "split_summaries.json", split_summaries)
        pd.concat(split_assignment_frames, ignore_index=True).to_parquet(
            output_root / "split_assignments.parquet", index=False
        )

        all_predictions = pd.concat(predictions_frames, ignore_index=True)
        all_predictions.to_parquet(output_root / "predictions.parquet", index=False)
        write_json(output_root / "embeddings_fingerprints.json", embeddings_fingerprints)

        per_seed_metrics = metrics_from_predictions(
            all_predictions,
            outlier_threshold=float(config["metrics"]["catastrophic_outlier_threshold"]),
            spectype_classes=spectype_classes or [],
        )
        aggregated_metrics = aggregate_seed_metrics(per_seed_metrics)
        write_json(output_root / "metrics.json", {"per_seed": per_seed_metrics, "aggregated": aggregated_metrics})
        tables_from_metrics(aggregated_metrics).to_csv(output_root / "tables.csv", index=False)


def main() -> int:
    config_path = Path("configs/phase6_probes.yaml")
    config = load_config(config_path)

    if "--preflight" in sys.argv[1:]:
        report = run_preflight(config)
        report_path = _write_preflight_report(config, report)
        print(f"Preflight report written to {report_path}: status={report['status']}")
        if report["status"] != "pass":
            return 1
        return 0

    run_full(config)
    return 0


def _flush_and_exit_successfully() -> None:
    """Exit cleanly after every artifact/context manager has finished.

    On THQL, the successful Phase-6 preflight wrote its passing report and
    then CPython 3.11 aborted during third-party extension finalization with
    ``PyGILState_Release: thread state ... must be current``.  ``os._exit``
    intentionally skips only interpreter finalizers.  It is called solely
    after ``main`` returns success, so files are already closed, tracked-run
    status is complete, and stdout/stderr are flushed.  Exceptions and
    explicit preflight failures keep the normal Python exit path and cannot
    be converted into success by this workaround.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        _flush_and_exit_successfully()
    raise SystemExit(exit_code)
