"""Review-gated Phase 0 cluster evaluation; query vectors must already be frozen locally."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import pandas as pd

from aion_reimp.config import load_config
from aion_reimp.artifacts import initialize_run, tracked_run
from aion_reimp.datasets import materialize_benchmark_coordinates
from aion_reimp.evaluate import evaluate_released_benchmarks
from aion_reimp.reference import (
    assert_reference_equivalence,
    load_author_reference,
    load_reimplemented_reference,
    resolve_released_files,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase0_reference.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    query_cache = Path(config["queries"]["openai_cache"])
    if not query_cache.exists():
        raise FileNotFoundError(
            f"R-OAI query cache is absent: {query_cache}. Freeze it locally and sync the cache artifacts."
        )

    output_root = initialize_run(
        Path(config["run"]["output_root"]),
        config["run"]["id"],
        config,
        shlex.join(sys.argv),
    )
    with tracked_run(output_root, {"phase": 0, "condition": "released_checkpoint"}):
        spec = config["reference_model"]
        config_path, weights_path = resolve_released_files(
            spec["repo_id"], spec["revision"], spec["config_file"], spec["weights_file"]
        )
        reimplemented = load_reimplemented_reference(config_path, weights_path)
        author = load_author_reference(Path(spec["orig_repo"]), config_path, weights_path)
        maxima = assert_reference_equivalence(reimplemented, author, seed=config["run"]["seed"])
        (output_root / "reference_equivalence.json").write_text(
            json.dumps(maxima, indent=2, sort_keys=True), encoding="utf-8"
        )

        materialize_benchmark_coordinates(config["benchmarks"], output_root / "benchmark_coordinates")
        evaluate_released_benchmarks(
            reimplemented,
            pd.read_parquet(query_cache),
            config["benchmarks"],
            output_root / "released_evaluation",
            reference_gate=config["reference_gate"],
        )


if __name__ == "__main__":
    main()
