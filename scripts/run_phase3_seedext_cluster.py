"""Review-gated Phase 3 seed-extension entrypoint: training and evaluation only.

Runs two additional seeds (see ``configs/phase3_10k_seedext.yaml``) on top of the
three already completed by ``phase3_10k_v1``. This script never re-downloads
images, never constructs a Qwen captioner, and never re-embeds documents: the
common manifest, captions, and text-embedding caches are loaded verbatim from
``source_run.output_root/source_run.run_id`` (cache-only mode) and every loader
fails loudly on a missing file or a fingerprint mismatch instead of silently
regenerating. The only model inference this script performs is projecting the
already-trained checkpoints and embedding the small canonical/paraphrase query
set for evaluation.

Before any of that loading, ``validate_base_run`` checks that the base run is
actually usable as a source: it completed, it ran the seeds this config claims
it reused, and its saved config agrees with this one everywhere data identity
and training behavior are defined. See ``manifest.json`` under this run's output
directory for the recorded outcome, including the source-content-fingerprint
provenance. After the per-seed training and evaluation loop, this script also
writes ``combined_summary.json``: the pooled five-seed estimate (three reused
plus two new) built from the base run's own summary plus this run's own gates,
alongside the two-seed-only ``phase3_10k_seedext_summary.json`` kept for
comparison.
"""

from __future__ import annotations

import gc
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch

from aion_reimp.artifacts import initialize_run, tracked_run, write_json
from aion_reimp.config import load_config
from aion_reimp.datasets import load_query_rows
from aion_reimp.evaluate import assert_ten_folds, evaluate_released_benchmarks, load_hf_frame
from aion_reimp.launch_contract import build_launch_contract, require_launch_allowed
from aion_reimp.metrics import summary_statistics
from aion_reimp.model import AIONSearchModel, ModelConfig
from aion_reimp.seedext import build_combined_summary
from aion_reimp.smoke import (
    load_cached_common_set,
    load_cached_text_embedding_caches,
    validate_base_run,
)
from aion_reimp.text_embeddings import QwenEmbedder, embedding_frame
from aion_reimp.training import TrainingSpec, assemble_condition_rows, train_condition


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> AIONSearchModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = AIONSearchModel(
        model_config,
        temperature_initial_scale=checkpoint["temperature_initial_scale"],
        temperature_max_scale=checkpoint["temperature_max_scale"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _preflight_check_folds(benchmark_specs) -> None:
    """Fail fast on a missing/malformed kfold column, before any training runs."""
    for benchmark in benchmark_specs:
        frame = load_hf_frame(benchmark["repo_id"], benchmark["revision"], ["kfold"])
        assert_ten_folds(frame["kfold"].to_numpy())


def main() -> None:
    config_path = Path("configs/phase3_10k_seedext.yaml")
    config = load_config(config_path)
    _preflight_check_folds(config["benchmarks"])
    prerequisites = config["prerequisites"]
    launch_contract = build_launch_contract(
        Path(prerequisites["phase0_reference_gate"]),
        Path(prerequisites["phase1_qwen_caption_audit_rows"]),
        Path(prerequisites["phase1_gpt_caption_audit_rows"]),
        bootstrap_samples=prerequisites["bootstrap_samples"],
        seed=config["run"]["seed"],
    )
    require_launch_allowed(launch_contract)

    output_root = initialize_run(
        Path(config["run"]["output_root"]),
        config["run"]["id"],
        config,
        shlex.join(sys.argv),
    )
    (output_root / "launch_contract.json").write_text(
        json.dumps(launch_contract, indent=2, sort_keys=True), encoding="utf-8"
    )

    source_run = config["source_run"]
    base_output_root = Path(source_run["output_root"]) / source_run["run_id"]

    with tracked_run(output_root, {"phase": 3, "condition": "10k_decision_pilot_seed_extension"}):
        # Fail fast, before any cache is loaded, unless the base run actually
        # completed, ran the seeds it claims to have reused, and its saved config
        # agrees with this one on data identity and training behavior. This also
        # loads (and content-fingerprint-verifies) the base run's source rows and
        # manifest, so they are not loaded a second time below.
        validation = validate_base_run(base_output_root, config)
        write_json(
            output_root / "manifest.json",
            {
                "base_run_id": source_run["run_id"],
                "base_output_root": str(base_output_root),
                "base_run_status": validation["run_status"].get("status"),
                "compared_config_sections": validation["compared_sections"],
                "source_cache_provenance": validation["source_provenance"],
            },
        )
        source, manifest = validation["source"], validation["manifest"]

        # Cache-only: reuses the exact 10k manifest (same run.seed as phase3_10k_v1),
        # captions, and embedding caches that run already wrote. Every loader fails
        # loudly on a missing file or fingerprint mismatch rather than regenerating.
        captions, common_source, common_manifest, caption_stats = load_cached_common_set(
            base_output_root, source
        )
        r_oai, r_qwen, q_qwen, embedding_spec = load_cached_text_embedding_caches(
            base_output_root, config["text_embedding"]
        )

        # Both Qwen-text conditions (R-QWEN, Q-QWEN) share one text encoder, so the
        # canonical query set only needs embedding once in Qwen space. This is the
        # only text-embedding inference this script performs: a handful of query
        # strings, not the 10k-row document caches loaded above from cache.
        query_rows = load_query_rows(Path(config["queries"]["file"]))
        query_texts = [row["text"] for row in query_rows]
        embedder = QwenEmbedder(embedding_spec)
        qwen_query_vectors = embedder.encode(query_texts, "query")
        qwen_query_embeddings = embedding_frame(
            [row["object_id"] for row in query_rows],
            query_texts,
            qwen_query_vectors,
            "query",
            embedding_spec,
        )
        qwen_query_cache = pd.DataFrame(query_rows).merge(
            qwen_query_embeddings.loc[:, ["object_id", "embedding"]],
            on="object_id",
            how="left",
            validate="one_to_one",
        )
        del embedder
        gc.collect()
        torch.cuda.empty_cache()

        oai_query_cache = pd.read_parquet(Path(config["queries"]["openai_cache"]))

        caches = {
            "released_summary_openai": r_oai,
            "released_summary_qwen": r_qwen,
            "qwen_caption_qwen": q_qwen,
        }
        query_caches = {
            "released_summary_openai": oai_query_cache,
            "released_summary_qwen": qwen_query_cache,
            "qwen_caption_qwen": qwen_query_cache,
        }
        training_spec = TrainingSpec.from_mapping(config["training"])
        model_shared = config["model"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seeds = config["seeds"]

        seed_condition_gates: Dict[str, Dict[str, Any]] = {}
        for seed in seeds:
            for condition in config["conditions"]:
                model_config = ModelConfig.from_shared_and_condition(model_shared, condition)
                condition_rows = assemble_condition_rows(
                    common_source, caches[condition["text_source"]]
                )
                condition_name = condition["name"]
                condition_dir = (
                    output_root
                    / "seeds"
                    / str(seed)
                    / "conditions"
                    / condition_name.lower().replace("-", "_")
                )
                gate = train_condition(
                    condition_rows,
                    model_config,
                    training_spec,
                    condition_dir,
                    seed=seed,
                    temperature_initial_scale=float(model_shared["temperature_initial_scale"]),
                    temperature_max_scale=float(model_shared["temperature_max_scale"]),
                )
                model = _load_checkpoint_model(condition_dir / "best_checkpoint.pt", device)
                benchmark_metrics = evaluate_released_benchmarks(
                    model,
                    query_caches[condition["text_source"]],
                    config["benchmarks"],
                    condition_dir / "benchmark_evaluation",
                    fold_column="kfold",
                )
                del model
                gc.collect()
                torch.cuda.empty_cache()
                gate["benchmark_metrics"] = benchmark_metrics
                seed_condition_gates[f"{condition_name}|seed={seed}"] = gate

        condition_summaries: Dict[str, Any] = {}
        for condition in config["conditions"]:
            name = condition["name"]
            gates = [seed_condition_gates[f"{name}|seed={seed}"] for seed in seeds]
            condition_summaries[name] = {
                "seeds": seeds,
                "validation_recall_at_10": summary_statistics(
                    [gate["validation_recall_at_10"] for gate in gates]
                ),
                "spiral_ndcg_at_10": summary_statistics(
                    [gate["benchmark_metrics"]["spiral"]["ndcg@10"] for gate in gates]
                ),
                "merger_ndcg_at_10": summary_statistics(
                    [gate["benchmark_metrics"]["merger"]["ndcg@10"] for gate in gates]
                ),
                "lens_ndcg_at_10": summary_statistics(
                    [gate["benchmark_metrics"]["lens"]["ndcg@10"] for gate in gates]
                ),
                "all_seeds_passed": all(gate["passed"] for gate in gates),
            }

        summary = {
            "note": (
                "Two-seed-only summary (45, 57). See combined_summary.json for the "
                "pooled five-seed estimate with the three reused seeds (13, 21, 33)."
            ),
            "extends_run_id": source_run["run_id"],
            "reused_seeds": source_run["reused_seeds"],
            "caption_generation": caption_stats,
            "common_rows": len(common_source),
            "seeds": seeds,
            "condition_seed_gates": seed_condition_gates,
            "condition_summaries": condition_summaries,
            "all_conditions_passed": all(
                item["all_seeds_passed"] for item in condition_summaries.values()
            ),
        }
        (output_root / "phase3_10k_seedext_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

        # The promised five-seed estimate: pool the three reused seeds' raw values
        # (read from the base run's own summary, already provenance-checked by
        # validate_base_run above) with this run's two new seeds' raw values.
        combined_conditions = build_combined_summary(
            base_summary=validation["summary"],
            base_run_id=source_run["run_id"],
            reused_seeds=source_run["reused_seeds"],
            extension_seed_condition_gates=seed_condition_gates,
            extension_run_id=config["run"]["id"],
            extension_seeds=seeds,
            conditions=config["conditions"],
        )
        combined_summary = {
            "base_run": {
                "run_id": source_run["run_id"],
                "output_root": str(base_output_root),
                "reused_seeds": source_run["reused_seeds"],
                "manifest_fingerprint": validation["source_provenance"]["manifest_fingerprint"],
                "source_content_fingerprint": validation["source_provenance"][
                    "source_content_fingerprint"
                ],
                "source_content_fingerprint_status": validation["source_provenance"][
                    "source_content_fingerprint_status"
                ],
            },
            "extension_run": {
                "run_id": config["run"]["id"],
                "output_root": str(output_root),
                "seeds": seeds,
            },
            "all_five_seeds": [*source_run["reused_seeds"], *seeds],
            "conditions": combined_conditions,
        }
        (output_root / "combined_summary.json").write_text(
            json.dumps(combined_summary, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
