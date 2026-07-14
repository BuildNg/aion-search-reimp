"""Run and persist all four Gemma extractor calibration responses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aion_reimp.config import load_config
from aion_reimp.morphology import (
    GemmaMorphologyExtractor,
    calibration_metrics,
    resolve_schema,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase1_extractor_diag_v4"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    spec = config["extractor"]
    schema_variant = spec.get("schema_variant", "flat")
    schema_json, schema_sha256, parse_fn = resolve_schema(schema_variant)
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"Diagnostic output already exists: {output}")
    output.mkdir(parents=True)
    print(f"schema_variant={schema_variant} schema_sha256={schema_sha256}", flush=True)
    extractor = GemmaMorphologyExtractor(
        model_path=spec["model_path"],
        prompt_template=Path(spec["prompt_file"]).read_text(encoding="utf-8"),
        dtype=spec["dtype"],
        max_new_tokens=spec["max_new_tokens"],
        enable_thinking=spec["enable_thinking"],
        schema_json=schema_json,
    )
    actual_prompt_sha256 = hashlib.sha256(
        extractor.prompt_template.encode("utf-8")
    ).hexdigest()
    if actual_prompt_sha256 != spec["prompt_sha256"]:
        raise RuntimeError("Released GalaxyBench judge prompt hash mismatch")
    metrics = calibration_metrics(
        extractor,
        Path(spec["calibration_file"]),
        response_jsonl=output / "responses.jsonl",
        error_jsonl=output / "errors.jsonl",
        continue_on_error=True,
        parse_fn=parse_fn,
    )
    (output / "summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if metrics["parse_errors"] or metrics["answer_accuracy"] < float(
        spec["calibration_min_answer_accuracy"]
    ):
        raise RuntimeError("Gemma extractor diagnostic did not pass calibration")


if __name__ == "__main__":
    main()
