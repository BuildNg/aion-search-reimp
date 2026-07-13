"""Review-gated preparation of manifests, exclusions, and released R-OAI cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from aion_reimp.cache import (
    NormalizationPolicy,
    ingest_released_embeddings,
    write_embedding_cache,
)
from aion_reimp.config import load_config
from aion_reimp.datasets import materialize_benchmark_coordinates, materialize_caption_screen
from aion_reimp.manifest import build_manifest, coordinate_exclusion_table, write_manifest


def main() -> None:
    phase0 = load_config(Path("configs/phase0_reference.yaml"))
    phase1 = load_config(Path("configs/phase1.yaml"))
    output_dir = Path("data/phase0")
    output_dir.mkdir(parents=True, exist_ok=False)

    training = phase0["training_data"]
    dataset = load_dataset(
        training["repo_id"], revision=training["revision"], split=training["split"]
    )
    required = {"object_id", "survey", "ra", "dec", "summary", "summary_text_embedding"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Training dataset missing required columns: {sorted(missing)}")
    source = dataset.select_columns(sorted(required)).to_pandas()
    source_coordinates = source.loc[:, ["object_id", "survey", "ra", "dec"]].copy()
    source_coordinates["source_row_id"] = np.arange(len(source_coordinates), dtype=np.int64)
    source_coordinates.to_parquet(output_dir / "source_coordinates.parquet", index=False)

    audit = phase1["benchmark"]
    screen_dir = Path(audit["input_dir"])
    labels_path = screen_dir / "caption_screen_labels.parquet"
    if not labels_path.exists():
        materialize_caption_screen(audit["repo_id"], audit["revision"], audit["split"], screen_dir)
    screen = pd.read_parquet(labels_path)

    coordinate_paths = materialize_benchmark_coordinates(
        phase0["benchmarks"], output_dir / "benchmark_coordinates"
    )
    benchmark_coordinates = {
        name: pd.read_parquet(path) for name, path in coordinate_paths.items()
    }
    coordinate_exclusions = coordinate_exclusion_table(
        source_coordinates,
        benchmark_coordinates,
        radius_arcsec=1.0,
    )
    coordinate_exclusions.to_csv(output_dir / "retrieval_coordinate_exclusions.csv", index=False)

    manifest = build_manifest(
        source_coordinates,
        {
            "retrieval_benchmarks_1arcsec": coordinate_exclusions["object_id"].astype(str),
            "caption_screen_64": screen["object_id"].astype(str),
        },
        seed=phase0["run"]["seed"],
    )
    write_manifest(manifest, output_dir / "object_manifest.parquet")

    policy = NormalizationPolicy(required=True, atol=1e-3)
    released_cache = ingest_released_embeddings(source, normalization_policy=policy)
    write_embedding_cache(
        released_cache,
        output_dir / "released_summary_openai_embeddings.parquet",
        normalization_policy=policy,
        metadata={
            "source_repo": training["repo_id"],
            "source_revision": training["revision"],
        },
    )


if __name__ == "__main__":
    main()
