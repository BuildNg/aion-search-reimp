"""Generate the 64 GPT-4.1-mini free-form descriptions locally."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from PIL import Image

from aion_reimp.cache import sha256_text
from aion_reimp.captioning import OpenRouterCaptioner, parse_caption_response
from aion_reimp.config import load_config
from aion_reimp.utils import append_jsonl, read_jsonl


def _read_env_value(path: Path, name: str) -> str:
    existing = os.environ.get(name)
    if existing:
        return existing
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    raise RuntimeError(f"{name} is absent from the environment and {path}")


def _estimated_cost(usage: Dict[str, Any], cost_config: Dict[str, Any]) -> float:
    reported = usage.get("reported_cost_usd")
    if isinstance(reported, (int, float)):
        return float(reported)
    return (
        int(usage["prompt_tokens"]) * float(cost_config["input_usd_per_million"])
        + int(usage["completion_tokens"]) * float(cost_config["output_usd_per_million"])
    ) / 1_000_000.0


def _image_preflight(labels: pd.DataFrame, output_path: Path) -> None:
    dimensions = []
    for row in labels.itertuples(index=False):
        with Image.open(row.image_path) as image:
            dimensions.append(
                {
                    "object_id": str(row.object_id),
                    "width": int(image.width),
                    "height": int(image.height),
                }
            )
    over_limit = [
        row for row in dimensions if row["width"] > 512 or row["height"] > 512
    ]
    record = {
        "images": len(dimensions),
        "max_width": max(row["width"] for row in dimensions),
        "max_height": max(row["height"] for row in dimensions),
        "all_at_most_512_px": not over_limit,
        "dimension_counts": {
            f"{width}x{height}": sum(
                row["width"] == width and row["height"] == height
                for row in dimensions
            )
            for width, height in sorted(
                {(row["width"], row["height"]) for row in dimensions}
            )
        },
        "over_limit": over_limit,
    }
    if over_limit:
        raise RuntimeError(
            "GPT detail=low preflight failed because at least one source image exceeds 512 px"
        )
    serialized = json.dumps(record, indent=2, sort_keys=True)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != record:
            raise ValueError(f"Image preflight artifact changed: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path("../../.env"))
    args = parser.parse_args()

    config = load_config(args.config)
    gpt = config["captioners"]["gpt"]
    benchmark = config["benchmark"]
    artifacts = config["artifacts"]
    cost_spec = config["cost"]
    prompt = Path(gpt["prompt_file"]).read_text(encoding="utf-8")
    api_key = _read_env_value(args.env_file, gpt["api_key_env"])

    input_dir = Path(benchmark["input_dir"])
    labels = pd.read_parquet(input_dir / "caption_screen_labels.parquet")
    labels = labels.sort_values("object_id").reset_index(drop=True)
    if len(labels) != 64 or labels["object_id"].astype(str).nunique() != 64:
        raise ValueError("Phase 1 requires exactly 64 unique caption-screen objects")
    labels["image_path"] = labels["object_id"].astype(str).map(
        lambda object_id: str((input_dir / "images" / f"{object_id}.png").resolve())
    )
    missing = [path for path in labels["image_path"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing caption-screen image: {missing[0]}")
    _image_preflight(labels, Path(config["artifacts"]["image_preflight"]))

    descriptions_path = Path(artifacts["gpt_descriptions"])
    usage_path = Path(artifacts["gpt_usage"])
    cost_path = Path(artifacts["gpt_cost"])
    errors_path = descriptions_path.with_name(descriptions_path.stem + "_errors.jsonl")
    if cost_path.exists():
        raise FileExistsError(f"Completed GPT cost artifact already exists: {cost_path}")
    completed_records = read_jsonl(descriptions_path, missing_ok=True)
    completed = {str(record["object_id"]) for record in completed_records}
    usage_records = read_jsonl(usage_path, missing_ok=True)
    if completed != {str(record["object_id"]) for record in usage_records}:
        raise ValueError("GPT description and usage resume sets differ")
    if completed - set(labels["object_id"].astype(str)):
        raise ValueError("GPT resume artifact contains an unexpected object")

    total_cost = sum(float(record["effective_cost_usd"]) for record in usage_records)
    total_prompt_tokens = sum(int(record["prompt_tokens"]) for record in usage_records)
    total_completion_tokens = sum(int(record["completion_tokens"]) for record in usage_records)
    captioner = OpenRouterCaptioner(
        model_id=gpt["model_id"],
        provider=gpt["provider"],
        base_url=gpt["base_url"],
        prompt=prompt,
        api_key=api_key,
        max_output_tokens=gpt["max_output_tokens"],
        temperature=gpt["temperature"],
        image_detail=gpt["image_detail"],
    )

    for row in labels.itertuples(index=False):
        object_id = str(row.object_id)
        if object_id in completed:
            continue
        if total_cost + float(cost_spec["reserve_per_request_usd"]) > float(
            cost_spec["hard_cap_usd"]
        ):
            raise RuntimeError("GPT cost reserve would exceed the configured hard cap")
        response = None
        usage = None
        try:
            response, usage = captioner.generate_response_with_metadata(Path(row.image_path))
            result = parse_caption_response(object_id, response)
        except Exception as error:
            append_jsonl(
                errors_path,
                {
                    "object_id": object_id,
                    "image_path": str(row.image_path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "raw_response": response,
                    "word_count": len(response.split()) if isinstance(response, str) else None,
                },
            )
            raise
        request_cost = _estimated_cost(usage, cost_spec)
        total_cost += request_cost
        total_prompt_tokens += int(usage["prompt_tokens"])
        total_completion_tokens += int(usage["completion_tokens"])
        if total_cost > float(cost_spec["hard_cap_usd"]):
            raise RuntimeError("Observed GPT cost exceeded the configured hard cap")
        append_jsonl(descriptions_path, result.as_record())
        append_jsonl(
            usage_path,
            {"object_id": object_id, **usage, "effective_cost_usd": request_cost},
        )
        completed.add(object_id)
        print(f"completed={len(completed)}/64 cost_usd={total_cost:.6f}", flush=True)

    if len(completed) != 64:
        raise RuntimeError(f"GPT description cache is incomplete: {len(completed)}/64")
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    cost_path.write_text(
        json.dumps(
            {
                "requests": 64,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "effective_cost_usd": total_cost,
                "hard_cap_usd": float(cost_spec["hard_cap_usd"]),
                "model_id": gpt["model_id"],
                "provider": gpt["provider"],
                "prompt_sha256": sha256_text(prompt),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
