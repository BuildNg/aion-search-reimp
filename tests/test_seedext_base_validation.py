import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from aion_reimp.config import load_config
from aion_reimp.manifest import build_manifest, manifest_fingerprint, write_manifest
from aion_reimp.seedext import build_combined_summary
from aion_reimp.smoke import validate_base_run


ROOT = Path(__file__).resolve().parents[1]
BASE_SEEDS = [13, 21, 33]
EXTENSION_SEEDS = [45, 57]


def _load_configs():
    base_config = load_config(ROOT / "configs" / "phase3_10k.yaml")
    extension_config = load_config(ROOT / "configs" / "phase3_10k_seedext.yaml")
    return base_config, extension_config


def _write_base_run(
    tmp_path: Path,
    base_config: dict,
    status: str = "complete",
    summary_seeds=BASE_SEEDS,
    with_data: bool = False,
) -> Path:
    base_output_root = tmp_path / "phase3_10k_v1"
    base_output_root.mkdir(parents=True)
    (base_output_root / "config.yaml").write_text(
        yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8"
    )
    (base_output_root / "run_status.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )
    (base_output_root / "phase3_10k_summary.json").write_text(
        json.dumps({"seeds": summary_seeds, "condition_seed_gates": {}}), encoding="utf-8"
    )
    if with_data:
        manifest_source = pd.DataFrame(
            {
                "object_id": ["a", "b", "c"],
                "survey": ["legacy", "legacy", "hsc"],
                "ra": [1.0, 2.0, 3.0],
                "dec": [-1.0, -2.0, -3.0],
                "source_row_id": [0, 1, 2],
            }
        )
        manifest = build_manifest(manifest_source, {}, seed=7)
        write_manifest(manifest, base_output_root / "data" / "manifest.parquet")
        source = pd.DataFrame(
            {
                "object_id": manifest["object_id"],
                "split": manifest["split"],
                "image_embedding": [[0.1, 0.2]] * len(manifest),
            }
        )
        source_path = base_output_root / "data" / "source_rows.parquet"
        source.to_parquet(source_path, index=False)
        source_path.with_suffix(source_path.suffix + ".meta.json").write_text(
            json.dumps(
                {"rows": len(source), "content_fingerprint": manifest_fingerprint(source)}
            ),
            encoding="utf-8",
        )
    return base_output_root


def test_validate_base_run_passing_case(tmp_path) -> None:
    base_config, extension_config = _load_configs()
    base_output_root = _write_base_run(tmp_path, base_config, with_data=True)

    result = validate_base_run(base_output_root, extension_config)

    assert result["run_status"]["status"] == "complete"
    assert result["summary"]["seeds"] == BASE_SEEDS
    assert set(result["source"]["object_id"]) == {"a", "b", "c"}
    assert set(result["manifest"]["object_id"]) == {"a", "b", "c"}
    assert (
        result["source_provenance"]["source_content_fingerprint_status"]
        == "verified_against_recorded_fingerprint"
    )
    assert result["compared_sections"][0] == "run.seed"
    for section in (
        "queries",
        "prerequisites",
        "source_data",
        "exclusions",
        "captioning",
        "text_embedding",
        "caches",
        "model",
        "training",
        "conditions",
        "benchmarks",
    ):
        assert section in result["compared_sections"]


def test_validate_base_run_missing_artifact_fails_loudly(tmp_path) -> None:
    base_config, extension_config = _load_configs()
    base_output_root = _write_base_run(tmp_path, base_config)
    (base_output_root / "run_status.json").unlink()

    with pytest.raises(FileNotFoundError, match="Base run validation requires"):
        validate_base_run(base_output_root, extension_config)


def test_validate_base_run_incomplete_run_status_is_fatal(tmp_path) -> None:
    base_config, extension_config = _load_configs()
    base_output_root = _write_base_run(tmp_path, base_config, status="failed")

    with pytest.raises(ValueError, match="not 'complete'"):
        validate_base_run(base_output_root, extension_config)


def test_validate_base_run_wrong_base_seeds_is_fatal(tmp_path) -> None:
    base_config, extension_config = _load_configs()
    base_output_root = _write_base_run(tmp_path, base_config, summary_seeds=[13, 21])

    with pytest.raises(ValueError, match=r"actually ran seeds \[13, 21\]"):
        validate_base_run(base_output_root, extension_config)


def test_validate_base_run_run_seed_mismatch_is_fatal(tmp_path) -> None:
    base_config, extension_config = _load_configs()
    tampered_base_config = copy.deepcopy(base_config)
    tampered_base_config["run"]["seed"] = 1
    base_output_root = _write_base_run(tmp_path, tampered_base_config)

    with pytest.raises(ValueError, match=r"diverge at run\.seed"):
        validate_base_run(base_output_root, extension_config)


def test_validate_base_run_names_divergent_scalar_key(tmp_path) -> None:
    base_config, extension_config = _load_configs()
    tampered_base_config = copy.deepcopy(base_config)
    tampered_base_config["source_data"]["sample_size"] = 5000
    base_output_root = _write_base_run(tmp_path, tampered_base_config)

    with pytest.raises(ValueError, match=r"diverge at source_data\.sample_size"):
        validate_base_run(base_output_root, extension_config)


def test_validate_base_run_names_divergent_key_inside_a_list(tmp_path) -> None:
    base_config, extension_config = _load_configs()
    tampered_base_config = copy.deepcopy(base_config)
    tampered_base_config["benchmarks"][0]["revision"] = "1" * 40
    base_output_root = _write_base_run(tmp_path, tampered_base_config)

    with pytest.raises(ValueError, match=r"diverge at benchmarks\[0\]\.revision"):
        validate_base_run(base_output_root, extension_config)


def test_validate_base_run_names_divergent_training_key(tmp_path) -> None:
    base_config, extension_config = _load_configs()
    tampered_base_config = copy.deepcopy(base_config)
    tampered_base_config["training"]["batch_size"] = 1
    base_output_root = _write_base_run(tmp_path, tampered_base_config)

    with pytest.raises(ValueError, match=r"diverge at training\.batch_size"):
        validate_base_run(base_output_root, extension_config)


def _gate(recall: float, spiral: float, merger: float, lens: float) -> dict:
    return {
        "validation_recall_at_10": recall,
        "benchmark_metrics": {
            "spiral": {"ndcg@10": spiral},
            "merger": {"ndcg@10": merger},
            "lens": {"ndcg@10": lens},
        },
    }


def test_build_combined_summary_pools_five_seeds_hand_computed() -> None:
    conditions = [{"name": "R-OAI"}]
    base_summary = {
        "condition_seed_gates": {
            "R-OAI|seed=13": _gate(0.10, 0.10, 0.20, 0.00),
            "R-OAI|seed=21": _gate(0.20, 0.20, 0.30, 0.00),
            "R-OAI|seed=33": _gate(0.30, 0.30, 0.40, 0.00),
        }
    }
    extension_seed_condition_gates = {
        "R-OAI|seed=45": _gate(0.40, 0.40, 0.50, 1.00),
        "R-OAI|seed=57": _gate(0.50, 0.50, 0.60, 0.00),
    }

    combined = build_combined_summary(
        base_summary=base_summary,
        base_run_id="phase3_10k_v1",
        reused_seeds=BASE_SEEDS,
        extension_seed_condition_gates=extension_seed_condition_gates,
        extension_run_id="phase3_10k_v1_seedext",
        extension_seeds=EXTENSION_SEEDS,
        conditions=conditions,
    )

    recall = combined["R-OAI"]["validation_recall_at_10"]
    # (0.10 + 0.20 + 0.30 + 0.40 + 0.50) / 5 = 0.30
    assert recall["mean"] == pytest.approx(0.30)
    assert recall["min"] == pytest.approx(0.10)
    assert recall["max"] == pytest.approx(0.50)
    assert recall["n"] == 5
    # population std of [0.10, 0.20, 0.30, 0.40, 0.50]
    assert recall["std"] == pytest.approx(0.14142135623730951)

    lens = combined["R-OAI"]["lens_ndcg_at_10"]
    # (0 + 0 + 0 + 1.00 + 0) / 5 = 0.20
    assert lens["mean"] == pytest.approx(0.20)

    assert combined["R-OAI"]["seeds"] == [13, 21, 33, 45, 57]
    per_seed = {row["seed"]: row for row in combined["R-OAI"]["per_seed"]}
    assert per_seed[13]["source_run_id"] == "phase3_10k_v1"
    assert per_seed[45]["source_run_id"] == "phase3_10k_v1_seedext"
    assert per_seed[45]["validation_recall_at_10"] == pytest.approx(0.40)
    assert per_seed[57]["lens_ndcg_at_10"] == pytest.approx(0.0)


def test_build_combined_summary_covers_all_conditions() -> None:
    conditions = [{"name": "R-OAI"}, {"name": "R-QWEN"}]
    base_summary = {
        "condition_seed_gates": {
            f"{name}|seed={seed}": _gate(0.1, 0.1, 0.1, 0.1)
            for name in ("R-OAI", "R-QWEN")
            for seed in BASE_SEEDS
        }
    }
    extension_seed_condition_gates = {
        f"{name}|seed={seed}": _gate(0.2, 0.2, 0.2, 0.2)
        for name in ("R-OAI", "R-QWEN")
        for seed in EXTENSION_SEEDS
    }

    combined = build_combined_summary(
        base_summary=base_summary,
        base_run_id="phase3_10k_v1",
        reused_seeds=BASE_SEEDS,
        extension_seed_condition_gates=extension_seed_condition_gates,
        extension_run_id="phase3_10k_v1_seedext",
        extension_seeds=EXTENSION_SEEDS,
        conditions=conditions,
    )

    assert set(combined) == {"R-OAI", "R-QWEN"}
    for name in combined:
        assert combined[name]["validation_recall_at_10"]["n"] == 5
        assert len(combined[name]["per_seed"]) == 5
