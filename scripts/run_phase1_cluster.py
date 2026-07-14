"""Run the replacement Phase 1 free-form caption and extraction audit on THQL."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import sys
from pathlib import Path

import pandas as pd

from aion_reimp.artifacts import initialize_run, tracked_run
from aion_reimp.caption_audit import (
    paired_accuracy_delta,
    paired_path_score_delta,
    write_audit,
    write_path_audit,
)
from aion_reimp.config import load_config
from aion_reimp.datasets import materialize_caption_screen
from aion_reimp.morphology import (
    GemmaMorphologyExtractor,
    append_morphology_results,
    calibration_metrics,
    resolve_schema,
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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _description_stats(rows: pd.DataFrame) -> dict:
    word_counts = rows["description"].astype(str).map(lambda value: len(value.split()))
    return {
        "descriptions": int(len(word_counts)),
        "mean_words": float(word_counts.mean()),
        "median_words": float(word_counts.median()),
        "max_words": int(word_counts.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
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

    qwen_source = Path(config["artifacts"]["qwen_descriptions"])
    qwen_descriptions = _read_jsonl(qwen_source)
    if len(qwen_descriptions) != 64 or qwen_descriptions["object_id"].astype(str).nunique() != 64:
        raise ValueError("Frozen Qwen descriptions must contain exactly 64 objects")

    gpt_source = Path(config["artifacts"]["gpt_descriptions"])
    gpt_descriptions = _read_jsonl(gpt_source)
    if len(gpt_descriptions) != 64 or gpt_descriptions["object_id"].astype(str).nunique() != 64:
        raise ValueError("Frozen GPT free-form descriptions must contain exactly 64 objects")
    if set(gpt_descriptions["object_id"].astype(str)) != set(labels["object_id"].astype(str)):
        raise ValueError("GPT free-form descriptions do not match the benchmark object set")
    if set(qwen_descriptions["object_id"].astype(str)) != set(labels["object_id"].astype(str)):
        raise ValueError("Qwen free-form descriptions do not match the benchmark object set")

    with tracked_run(
        output_root,
        {
            "phase": 1,
            "protocol": "freeform_caption_shared_gemma_extractor",
        },
    ):
        caption_prompt_path = Path(config["captioners"]["qwen"]["prompt_file"])
        qwen_path = output_root / "qwen_descriptions.jsonl"
        shutil.copy2(qwen_source, qwen_path)

        frozen_gpt_path = output_root / "gpt41mini_descriptions.jsonl"
        shutil.copy2(gpt_source, frozen_gpt_path)

        extractor_spec = config["extractor"]
        schema_variant = extractor_spec.get("schema_variant", "flat")
        schema_json, schema_sha256, parse_fn = resolve_schema(schema_variant)
        extractor_prompt_path = Path(extractor_spec["prompt_file"])
        extractor = GemmaMorphologyExtractor(
            model_path=extractor_spec["model_path"],
            prompt_template=extractor_prompt_path.read_text(encoding="utf-8"),
            dtype=extractor_spec["dtype"],
            max_new_tokens=extractor_spec["max_new_tokens"],
            enable_thinking=extractor_spec["enable_thinking"],
            schema_json=schema_json,
        )
        actual_prompt_sha256 = _sha256_text(extractor.prompt_template)
        if actual_prompt_sha256 != extractor_spec["prompt_sha256"]:
            raise RuntimeError("Released GalaxyBench judge prompt hash mismatch")
        calibration = calibration_metrics(
            extractor,
            Path(extractor_spec["calibration_file"]),
            response_jsonl=output_root / "extractor_calibration_responses.jsonl",
            error_jsonl=output_root / "extractor_calibration_errors.jsonl",
            parse_fn=parse_fn,
        )
        (output_root / "extractor_calibration.json").write_text(
            json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8"
        )
        if float(calibration["answer_accuracy"]) < float(
            extractor_spec["calibration_min_answer_accuracy"]
        ):
            raise RuntimeError(
                "Gemma extractor failed its preregistered synthetic calibration gate"
            )

        question_metrics = {}
        question_rows = {}
        path_metrics = {}
        path_rows = {}
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
                parse_fn=parse_fn,
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
            write_path_audit(
                extractions,
                labels,
                audit_dir,
                bootstrap_samples=config["audit"]["bootstrap_samples"],
                seed=config["run"]["seed"],
            )
            question_metrics[name] = json.loads(
                (audit_dir / "caption_audit_metrics.json").read_text(encoding="utf-8")
            )
            question_rows[name] = pd.read_csv(audit_dir / "caption_audit_rows.csv")
            path_metrics[name] = json.loads(
                (audit_dir / "judge_path_audit_metrics.json").read_text(encoding="utf-8")
            )
            path_rows[name] = pd.read_csv(audit_dir / "judge_path_audit_rows.csv")

        paired_question_delta = paired_accuracy_delta(
            question_rows["qwen3vl_8b"],
            question_rows["gpt41mini"],
            bootstrap_samples=config["audit"]["bootstrap_samples"],
            seed=config["run"]["seed"],
        )
        paired_path_delta = paired_path_score_delta(
            path_rows["qwen3vl_8b"],
            path_rows["gpt41mini"],
            bootstrap_samples=config["audit"]["bootstrap_samples"],
            seed=config["run"]["seed"],
        )

        comparison = {
            "protocol": (
                "free-form descriptions judged by the same schema-constrained Gemma model "
                "using the released GalaxyBench prompt and path score"
            ),
            "primary_metric": config["audit"]["primary_metric"],
            "secondary_metric": config["audit"]["secondary_metric"],
            "caption_prompt_sha256": _sha256(caption_prompt_path),
            "extractor_prompt_file_sha256": _sha256(extractor_prompt_path),
            "extractor_prompt_sha256": actual_prompt_sha256,
            "extractor_model": {
                "model_id": extractor_spec["model_id"],
                "revision": extractor_spec["revision"],
                "enable_thinking": extractor_spec["enable_thinking"],
                "structured_output_engine": extractor_spec["structured_output_engine"],
                "schema_variant": schema_variant,
                "schema_sha256": schema_sha256,
            },
            "released_path_overlap": {
                "gpt41mini": path_metrics["gpt41mini"],
                "qwen3vl_8b": path_metrics["qwen3vl_8b"],
                "paired_delta_gpt_minus_qwen": paired_path_delta,
            },
            "per_question_diagnostic": {
                "gpt41mini": question_metrics["gpt41mini"],
                "qwen3vl_8b": question_metrics["qwen3vl_8b"],
                "paired_delta_gpt_minus_qwen": paired_question_delta,
            },
            "description_lengths": {
                "gpt41mini": _description_stats(gpt_descriptions),
                "qwen3vl_8b": _description_stats(qwen_descriptions),
            },
            "gpt_source_sha256": _sha256(gpt_source),
            "qwen_source_sha256": _sha256(qwen_source),
        }
        (output_root / "comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
