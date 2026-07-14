"""Review-gated Phase 3 ten-thousand-row decision pilot entrypoint."""

from __future__ import annotations

import gc
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch

from aion_reimp.artifacts import initialize_run, tracked_run
from aion_reimp.cache import (
    NormalizationPolicy,
    cache_fingerprint,
    ingest_released_embeddings,
    validate_embedding_cache,
    write_embedding_cache,
)
from aion_reimp.captioning import QwenCaptioner, append_caption_results
from aion_reimp.config import load_config
from aion_reimp.datasets import load_query_rows
from aion_reimp.evaluate import assert_ten_folds, evaluate_released_benchmarks, load_hf_frame
from aion_reimp.launch_contract import build_launch_contract, require_launch_allowed
from aion_reimp.manifest import write_manifest
from aion_reimp.metrics import summary_statistics
from aion_reimp.model import AIONSearchModel, ModelConfig
from aion_reimp.smoke import prepare_smoke_source
from aion_reimp.text_embeddings import EmbeddingSpec, QwenEmbedder, embedding_frame
from aion_reimp.training import TrainingSpec, assemble_condition_rows, train_condition


def _load_subset(path: Path, object_ids) -> pd.DataFrame:
    identifiers = [str(value) for value in object_ids]
    frame = pd.read_parquet(path, filters=[("object_id", "in", identifiers)])
    if set(frame["object_id"].astype(str)) != set(identifiers):
        raise ValueError(f"Cache subset from {path} does not match requested object IDs")
    return frame


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
    """Fail fast on a missing/malformed kfold column, before any captioning or training runs."""
    for benchmark in benchmark_specs:
        frame = load_hf_frame(benchmark["repo_id"], benchmark["revision"], ["kfold"])
        assert_ten_folds(frame["kfold"].to_numpy())


def main() -> None:
    config_path = Path("configs/phase3_10k.yaml")
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

    with tracked_run(output_root, {"phase": 3, "condition": "10k_decision_pilot"}):
        # The data seed (run.seed) picks the common 10k manifest once; it is independent
        # of the per-condition training seeds in config["seeds"], which vary model init
        # and dataloader shuffling only, per the "one common manifest, several seeds" plan.
        source, manifest = prepare_smoke_source(
            config["source_data"],
            config["exclusions"],
            output_root / "data",
            seed=config["run"]["seed"],
        )

        caption_spec = config["captioning"]
        captioner = QwenCaptioner(
            caption_spec["model_id"],
            caption_spec["revision"],
            Path(caption_spec["prompt_file"]).read_text(encoding="utf-8"),
            dtype=caption_spec["dtype"],
            max_new_tokens=caption_spec["max_new_tokens"],
        )
        captions_path = output_root / "q_qwen_captions.jsonl"
        error_path = output_root / "errors.jsonl"
        caption_stats = append_caption_results(
            captioner,
            source.loc[:, ["object_id", "image_path"]].to_dict(orient="records"),
            captions_path,
            error_jsonl=error_path,
            max_error_rate=float(caption_spec["max_error_rate"]),
        )
        (output_root / "caption_generation.json").write_text(
            json.dumps(caption_stats, indent=2, sort_keys=True), encoding="utf-8"
        )
        del captioner
        gc.collect()
        torch.cuda.empty_cache()

        captions = pd.read_json(captions_path, lines=True)
        successful_ids = set(captions["object_id"].astype(str))
        common_source = source[source["object_id"].astype(str).isin(successful_ids)].copy()
        common_manifest = manifest[
            manifest["object_id"].astype(str).isin(successful_ids)
        ].copy()
        if len(common_source) != len(captions):
            raise AssertionError("Successful captions do not join one-to-one with the Phase 3 source")
        write_manifest(common_manifest, output_root / "common_manifest.parquet")

        policy = NormalizationPolicy(
            required=True, atol=float(config["text_embedding"]["normalization_atol"])
        )
        r_oai = ingest_released_embeddings(
            common_source.rename(
                columns={
                    "released_summary": "summary",
                    "released_openai_embedding": "summary_text_embedding",
                }
            ),
            text_column="summary",
            embedding_column="summary_text_embedding",
            normalization_policy=policy,
        )
        write_embedding_cache(
            r_oai,
            output_root / "r_oai_embeddings.parquet",
            normalization_policy=policy,
            metadata={"transform": "released_verbatim_subset"},
        )

        r_qwen_source_path = (
            Path(config["caches"]["phase1_normalized_dir"])
            / config["caches"]["released_summary_qwen"]
        )
        r_qwen = _load_subset(r_qwen_source_path, common_source["object_id"])
        validate_embedding_cache(r_qwen, policy)
        r_qwen_source_meta = json.loads(
            r_qwen_source_path.with_suffix(
                r_qwen_source_path.suffix + ".meta.json"
            ).read_text(encoding="utf-8")
        )
        write_embedding_cache(
            r_qwen,
            output_root / "r_qwen_embeddings.parquet",
            normalization_policy=policy,
            metadata={
                "source_path": str(r_qwen_source_path),
                "source_fingerprint": r_qwen_source_meta["fingerprint"],
                "subset_fingerprint": cache_fingerprint(r_qwen),
                "transform": "manifest_subset",
            },
        )

        embedding_spec = EmbeddingSpec.from_mapping(config["text_embedding"])
        embedder = QwenEmbedder(embedding_spec)
        q_vectors = embedder.encode(captions["description"].astype(str).tolist(), "document")
        q_qwen = embedding_frame(
            captions["object_id"].astype(str).tolist(),
            captions["description"].astype(str).tolist(),
            q_vectors,
            "document",
            embedding_spec,
        )
        write_embedding_cache(
            q_qwen,
            output_root / "q_qwen_embeddings.parquet",
            normalization_policy=embedding_spec.normalization_policy,
            metadata={"caption_prompt": caption_spec["prompt_file"]},
        )

        # Both Qwen-text conditions (R-QWEN, Q-QWEN) share one text encoder, so the
        # canonical query set only needs embedding once in Qwen space.
        query_rows = load_query_rows(Path(config["queries"]["file"]))
        query_texts = [row["text"] for row in query_rows]
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
                model_config = ModelConfig.from_mapping(
                    {
                        "image_input_dim": model_shared["image_input_dim"],
                        "text_input_dim": condition["text_input_dim"],
                        "embedding_dim": model_shared["embedding_dim"],
                        "image_hidden_dim": model_shared["image_hidden_dim"],
                        "text_hidden_dim": model_shared["text_hidden_dim"],
                        "dropout": model_shared["dropout"],
                        "use_mean_embeddings": model_shared["use_mean_embeddings"],
                    }
                )
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
            "caption_generation": caption_stats,
            "common_rows": len(common_source),
            "seeds": seeds,
            "condition_seed_gates": seed_condition_gates,
            "condition_summaries": condition_summaries,
            "all_conditions_passed": all(
                item["all_seeds_passed"] for item in condition_summaries.values()
            ),
        }
        (output_root / "phase3_10k_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
