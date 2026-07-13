"""Run the replacement Phase 1 free-form caption and extraction audit on THQL."""

from __future__ import annotations

import gc
import hashlib
import json
import shlex
import shutil
import sys
from pathlib import Path

import pandas as pd

from aion_reimp.artifacts import initialize_run, tracked_run
from aion_reimp.caption_audit import paired_accuracy_delta, write_audit
from aion_reimp.captioning import QwenCaptioner, append_caption_results
from aion_reimp.config import load_config
from aion_reimp.datasets import materialize_caption_screen
from aion_reimp.morphology import (
    GemmaMorphologyExtractor,
    append_morphology_results,
    calibration_metrics,
)


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _description_stats(rows: pd.DataFrame) -> dict:
    word_counts = rows["description"].astype(str).map(lambda value: len(value.split()))
    return {
        "descriptions": int(len(word_counts)),
        "mean_words": float(word_counts.mean()),
        "median_words": float(word_counts.median()),
        "max_words": int(word_counts.max()),
    }


def main() -> None:
    config = load_config(Path("configs/phase1.yaml"))
    output_root = initialize_run(
        Path(config["run"]["output_root"]),
        config["run"]["id"],
        config,
        shlex.join(sys.argv),
    )
    benchmark = config["benchmark"]
    screen_dir = Path(benchmark["input_dir"])
    labels_path = screen_dir / "caption_screen_labels.parquet"
    if not labels_path.exists():
        materialize_caption_screen(
            benchmark["repo_id"], benchmark["revision"], benchmark["split"], screen_dir
        )
    labels = pd.read_parquet(labels_path).sort_values("object_id").reset_index(drop=True)
    if len(labels) != 64 or labels["object_id"].astype(str).nunique() != 64:
        raise ValueError("Phase 1 requires exactly 64 unique benchmark objects")
    labels["image_path"] = labels["object_id"].astype(str).map(
        lambda object_id: str((screen_dir / "images" / f"{object_id}.png").resolve())
    )
    missing_images = [path for path in labels["image_path"] if not Path(path).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing caption-screen image: {missing_images[0]}")

    gpt_source = Path(config["artifacts"]["gpt_descriptions"])
    gpt_descriptions = _read_jsonl(gpt_source)
    if len(gpt_descriptions) != 64 or gpt_descriptions["object_id"].astype(str).nunique() != 64:
        raise ValueError("Frozen GPT free-form descriptions must contain exactly 64 objects")
    if set(gpt_descriptions["object_id"].astype(str)) != set(labels["object_id"].astype(str)):
        raise ValueError("GPT free-form descriptions do not match the benchmark object set")

    with tracked_run(
        output_root,
        {
            "phase": 1,
            "protocol": "freeform_caption_shared_gemma_extractor",
        },
    ):
        qwen_spec = config["captioners"]["qwen"]
        caption_prompt_path = Path(qwen_spec["prompt_file"])
        qwen = QwenCaptioner(
            qwen_spec["model_id"],
            qwen_spec["revision"],
            caption_prompt_path.read_text(encoding="utf-8"),
            dtype=qwen_spec["dtype"],
            max_new_tokens=qwen_spec["max_new_tokens"],
        )
        qwen_path = output_root / "qwen_descriptions.jsonl"
        append_caption_results(
            qwen,
            labels.to_dict(orient="records"),
            qwen_path,
            error_jsonl=output_root / "qwen_caption_errors.jsonl",
            max_error_rate=0.0,
            max_words=config["caption_policy"]["max_words"],
            truncate_over_limit=True,
        )
        del qwen
        gc.collect()
        import torch

        torch.cuda.empty_cache()

        frozen_gpt_path = output_root / "gpt41mini_descriptions.jsonl"
        shutil.copy2(gpt_source, frozen_gpt_path)
        qwen_descriptions = _read_jsonl(qwen_path)

        extractor_spec = config["extractor"]
        extractor_prompt_path = Path(extractor_spec["prompt_file"])
        extractor = GemmaMorphologyExtractor(
            model_path=extractor_spec["model_path"],
            prompt_template=extractor_prompt_path.read_text(encoding="utf-8"),
            dtype=extractor_spec["dtype"],
            max_new_tokens=extractor_spec["max_new_tokens"],
            enable_thinking=extractor_spec["enable_thinking"],
        )
        calibration = calibration_metrics(extractor, Path(extractor_spec["calibration_file"]))
        (output_root / "extractor_calibration.json").write_text(
            json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8"
        )
        if float(calibration["answer_accuracy"]) < float(
            extractor_spec["calibration_min_answer_accuracy"]
        ):
            raise RuntimeError(
                "Gemma extractor failed its preregistered synthetic calibration gate"
            )

        audit_metrics = {}
        audit_rows = {}
        for name, descriptions in (
            ("qwen3vl_8b", qwen_descriptions),
            ("gpt41mini", gpt_descriptions),
        ):
            extraction_path = output_root / f"{name}_extractions.jsonl"
            append_morphology_results(
                extractor,
                descriptions.to_dict(orient="records"),
                extraction_path,
                error_jsonl=output_root / f"{name}_errors.jsonl",
                max_error_rate=extractor_spec["max_error_rate"],
            )
            extractions = _read_jsonl(extraction_path)
            audit_dir = output_root / f"{name}_audit"
            write_audit(
                extractions,
                labels,
                audit_dir,
                bootstrap_samples=config["audit"]["bootstrap_samples"],
                seed=config["run"]["seed"],
            )
            audit_metrics[name] = json.loads(
                (audit_dir / "caption_audit_metrics.json").read_text(encoding="utf-8")
            )
            audit_rows[name] = pd.read_csv(audit_dir / "caption_audit_rows.csv")

        paired_delta = paired_accuracy_delta(
            audit_rows["qwen3vl_8b"],
            audit_rows["gpt41mini"],
            bootstrap_samples=config["audit"]["bootstrap_samples"],
            seed=config["run"]["seed"],
        )

        comparison = {
            "protocol": "free-form descriptions followed by the same text-only Gemma extractor",
            "caption_prompt_sha256": _sha256(caption_prompt_path),
            "extractor_prompt_sha256": _sha256(extractor_prompt_path),
            "extractor_model": {
                "model_id": extractor_spec["model_id"],
                "revision": extractor_spec["revision"],
                "enable_thinking": extractor_spec["enable_thinking"],
            },
            "gpt41mini": audit_metrics["gpt41mini"],
            "qwen3vl_8b": audit_metrics["qwen3vl_8b"],
            "description_lengths": {
                "gpt41mini": _description_stats(gpt_descriptions),
                "qwen3vl_8b": _description_stats(qwen_descriptions),
            },
            "caption_compliance": {
                "gpt41mini": {
                    "failures": int(
                        gpt_descriptions.get("compliance_failure", pd.Series(dtype=bool))
                        .fillna(False)
                        .astype(bool)
                        .sum()
                    ),
                    "readout": "secondary_truncated_fallback"
                    if "compliance_failure" in gpt_descriptions
                    and gpt_descriptions["compliance_failure"].fillna(False).astype(bool).any()
                    else "primary_matched",
                },
                "qwen3vl_8b": {
                    "failures": int(
                        qwen_descriptions.get("compliance_failure", pd.Series(dtype=bool))
                        .fillna(False)
                        .astype(bool)
                        .sum()
                    ),
                    "readout": "secondary_truncated_fallback"
                    if "compliance_failure" in qwen_descriptions
                    and qwen_descriptions["compliance_failure"].fillna(False).astype(bool).any()
                    else "primary_matched",
                },
            },
            "paired_accuracy_delta_gpt_minus_qwen": paired_delta,
            "gpt_source_sha256": _sha256(gpt_source),
            "qwen_source_sha256": _sha256(qwen_path),
        }
        (output_root / "comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
