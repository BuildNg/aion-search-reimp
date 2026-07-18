"""Review-gated Phase 2 one-thousand-row end-to-end smoke entrypoint."""

from __future__ import annotations

import gc
import json
import shlex
import sys
from pathlib import Path

import torch

from aion_reimp.artifacts import initialize_run, tracked_run
from aion_reimp.config import load_config
from aion_reimp.launch_contract import build_launch_contract, require_launch_allowed
from aion_reimp.model import ModelConfig
from aion_reimp.smoke import (
    build_common_text_embedding_caches,
    generate_qwen_captions_and_common_set,
    prepare_smoke_source,
)
from aion_reimp.training import (
    TrainingSpec,
    assemble_condition_rows,
    train_condition,
)


def main() -> None:
    config_path = Path("configs/phase2_smoke.yaml")
    config = load_config(config_path)
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

    with tracked_run(output_root, {"phase": 2, "condition": "1k_end_to_end_smoke"}):
        source, manifest = prepare_smoke_source(
            config["source_data"],
            config["exclusions"],
            output_root / "data",
            seed=config["run"]["seed"],
        )

        caption_spec = config["captioning"]
        captions, common_source, common_manifest, caption_stats = (
            generate_qwen_captions_and_common_set(
                caption_spec,
                source,
                manifest,
                output_root,
                "Successful captions do not join one-to-one with smoke source",
            )
        )

        r_oai, r_qwen, q_qwen, embedding_spec, embedder = build_common_text_embedding_caches(
            common_source,
            captions,
            config["text_embedding"],
            config["caches"],
            caption_spec["prompt_file"],
            output_root,
        )
        del embedder
        gc.collect()
        torch.cuda.empty_cache()

        caches = {
            "released_summary_openai": r_oai,
            "released_summary_qwen": r_qwen,
            "qwen_caption_qwen": q_qwen,
        }
        training_spec = TrainingSpec.from_mapping(config["training"])
        model_shared = config["model"]
        condition_gates = {}
        for condition in config["conditions"]:
            model_config = ModelConfig.from_shared_and_condition(model_shared, condition)
            condition_rows = assemble_condition_rows(
                common_source, caches[condition["text_source"]]
            )
            condition_name = condition["name"]
            condition_gates[condition_name] = train_condition(
                condition_rows,
                model_config,
                training_spec,
                output_root / "conditions" / condition_name.lower().replace("-", "_"),
                seed=config["run"]["seed"],
                temperature_initial_scale=float(model_shared["temperature_initial_scale"]),
                temperature_max_scale=float(model_shared["temperature_max_scale"]),
            )
        summary = {
            "caption_generation": caption_stats,
            "common_rows": len(common_source),
            "conditions": condition_gates,
            "all_conditions_passed": all(gate["passed"] for gate in condition_gates.values()),
        }
        (output_root / "phase2_smoke_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
