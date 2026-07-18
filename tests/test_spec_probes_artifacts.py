import json
from pathlib import Path

import numpy as np
import pandas as pd

from aion_reimp.artifacts import initialize_run, tracked_run, write_json
from aion_reimp.manifest import manifest_fingerprint

from spec_probes.encoders import SpectrumBatch, SpectrumEncoderAdapter
from spec_probes.probes import make_cv_folds
from spec_probes.run_probes import (
    aggregate_seed_metrics,
    embeddings_fingerprint,
    metrics_from_predictions,
    run_baseline_suite,
    run_probe_suite_for_encoder,
    tables_from_metrics,
)
from spec_probes.spectra_data import assert_no_split_leakage, object_level_split, split_fingerprint


SPECTYPE_CLASSES = ["GALAXY", "QSO", "STAR"]
PROBE_CONFIG = {
    "linear": {"ridge_alpha_grid": [0.1, 1.0, 10.0], "logistic_c_grid": [0.1, 1.0, 10.0], "logistic_max_iter": 200},
    "knn": {"k": 3, "metric": "cosine"},
}
SPLIT_SEEDS = (99, 100)


class _FakeEncoder(SpectrumEncoderAdapter):
    def __init__(self, name, output_dim=5, seed=0):
        self.name = name
        self.output_dim = output_dim
        self.revision = f"{name}-rev"
        self._rng = np.random.default_rng(seed)

    def embed(self, batch):
        return self._rng.normal(size=(len(batch), self.output_dim)).astype(np.float32)


def _batch(object_ids, seed):
    rng = np.random.default_rng(seed)
    n = len(object_ids)
    n_pix = 12
    return SpectrumBatch(
        object_id=np.array(object_ids),
        flux=1.0 + 0.05 * rng.normal(size=(n, n_pix)),
        wave=np.linspace(3600.0, 9800.0, n_pix),
    )


def test_phase6_artifact_contract_is_complete(tmp_path: Path) -> None:
    object_ids = [f"obj-{i:03d}" for i in range(40)]
    rng = np.random.default_rng(0)
    z = rng.uniform(0.1, 1.5, size=40)
    spectype = np.array(SPECTYPE_CLASSES)[rng.integers(0, 3, size=40)]
    id_to_index = {object_id: index for index, object_id in enumerate(object_ids)}

    resolved_config = {"schema_version": 1, "kind": "phase6_probes", "run": {"id": "t1"}}
    output_root = initialize_run(
        tmp_path / "results", "phase6_probes_test", resolved_config, "pytest invocation"
    )

    with tracked_run(output_root, {"phase": 6, "condition": "spectrum_encoder_probes"}):
        sample_manifest = pd.DataFrame(
            {
                "object_id": object_ids,
                "sample_position": np.arange(len(object_ids)),
                "z": z,
                "zwarn": np.ones(len(object_ids), dtype=bool),
            }
        )
        sample_manifest.to_parquet(output_root / "sample_manifest.parquet", index=False)
        write_json(
            output_root / "sample_summary.json",
            {"rows": len(sample_manifest), "fingerprint": manifest_fingerprint(sample_manifest)},
        )
        predictions_frames = []
        embeddings_fingerprints = {}
        split_summaries = {}
        split_assignment_frames = []

        for split_seed in SPLIT_SEEDS:
            # One split per seed, derived once and fingerprinted, shared by
            # every encoder within that seed (Finding 4d: the whole suite
            # reruns per split seed).
            split = object_level_split(object_ids, seed=split_seed, train_ratio=0.75)
            assert_no_split_leakage(split)
            fingerprint = split_fingerprint(split)
            split_summaries[str(split_seed)] = {"fingerprint": fingerprint, "rows": len(split)}
            split_with_seed = split.copy()
            split_with_seed.insert(0, "split_seed", split_seed)
            split_assignment_frames.append(split_with_seed)

            train_ids = split.loc[split["split"] == "train", "object_id"].tolist()
            test_ids = split.loc[split["split"] == "test", "object_id"].tolist()
            train_index = [id_to_index[o] for o in train_ids]
            test_index = [id_to_index[o] for o in test_ids]
            cv_folds = make_cv_folds(len(train_ids), cv_folds=3, seed=split_seed)

            predictions_frames.append(
                run_baseline_suite(
                    test_ids,
                    z[train_index],
                    z[test_index],
                    split_seed,
                    spectype_train=spectype[train_index],
                    spectype_test=spectype[test_index],
                )
            )

            for index, name in enumerate(("encoder_a", "encoder_b", "encoder_c")):
                encoder = _FakeEncoder(name, output_dim=5, seed=index)
                train_batch = _batch([object_ids[i] for i in train_index], seed=10 + index)
                test_batch = _batch([object_ids[i] for i in test_index], seed=20 + index)
                predictions = run_probe_suite_for_encoder(
                    encoder,
                    train_batch,
                    test_batch,
                    z[train_index],
                    z[test_index],
                    PROBE_CONFIG,
                    cv_folds,
                    split_seed,
                    seed=0,
                    spectype_train=spectype[train_index],
                    spectype_test=spectype[test_index],
                    spectype_classes=SPECTYPE_CLASSES,
                )
                predictions_frames.append(predictions)
                embeddings_fingerprints[f"{name}|{split_seed}"] = {
                    "fingerprint": embeddings_fingerprint(test_batch.object_id, encoder.embed(test_batch)),
                    "rows": len(test_batch),
                    "output_dim": encoder.output_dim,
                }

        write_json(output_root / "split_summaries.json", split_summaries)
        pd.concat(split_assignment_frames, ignore_index=True).to_parquet(
            output_root / "split_assignments.parquet", index=False
        )

        all_predictions = pd.concat(predictions_frames, ignore_index=True)
        all_predictions.to_parquet(output_root / "predictions.parquet", index=False)
        write_json(output_root / "embeddings_fingerprints.json", embeddings_fingerprints)

        per_seed_metrics = metrics_from_predictions(
            all_predictions, outlier_threshold=0.15, spectype_classes=SPECTYPE_CLASSES
        )
        aggregated_metrics = aggregate_seed_metrics(per_seed_metrics)
        write_json(output_root / "metrics.json", {"per_seed": per_seed_metrics, "aggregated": aggregated_metrics})
        tables_from_metrics(aggregated_metrics).to_csv(output_root / "tables.csv", index=False)

    expected_files = {
        "config.yaml",
        "command.txt",
        "run_status.json",
        "sample_manifest.parquet",
        "sample_summary.json",
        "split_summaries.json",
        "split_assignments.parquet",
        "predictions.parquet",
        "embeddings_fingerprints.json",
        "metrics.json",
        "tables.csv",
    }
    present = {path.name for path in output_root.iterdir()}
    assert expected_files <= present

    status = json.loads((output_root / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"

    # Metrics must be recomputable from row-level predictions alone.
    reloaded_predictions = pd.read_parquet(output_root / "predictions.parquet")
    recomputed_per_seed = metrics_from_predictions(
        reloaded_predictions, outlier_threshold=0.15, spectype_classes=SPECTYPE_CLASSES
    )
    assert recomputed_per_seed == per_seed_metrics
    assert aggregate_seed_metrics(recomputed_per_seed) == aggregated_metrics

    # Every real encoder's predictions, in every split seed, must key off
    # exactly that seed's test IDs from the shared split -- no per-encoder
    # split drift -- and the trivial baseline must appear alongside them.
    for split_seed in SPLIT_SEEDS:
        split = object_level_split(object_ids, seed=split_seed, train_ratio=0.75)
        test_ids = set(split.loc[split["split"] == "test", "object_id"])
        seed_predictions = all_predictions[all_predictions["split_seed"] == split_seed]
        assert set(seed_predictions["encoder"]) >= {"encoder_a", "encoder_b", "encoder_c", "trivial_baseline"}
        for name in ("encoder_a", "encoder_b", "encoder_c", "trivial_baseline"):
            rows = seed_predictions[seed_predictions["encoder"] == name]
            assert set(rows["object_id"]) == test_ids
