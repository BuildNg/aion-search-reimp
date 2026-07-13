"""One-thousand-row smoke data preparation with audited exclusions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from .datasets import _to_image, load_pinned_dataset
from .manifest import (
    assert_exclusion_coverage,
    assert_no_benchmark_leakage,
    build_manifest,
    coordinate_exclusion_coverage,
    exact_exclusion_coverage,
    write_manifest,
)


def _selection_key(object_id: str, seed: int) -> str:
    return hashlib.sha256(f"smoke|{object_id}|{seed}".encode("utf-8")).hexdigest()


def select_smoke_rows(manifest: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    assert_no_benchmark_leakage(manifest)
    eligible = manifest[manifest["split"].isin(["train", "validation"])].copy()
    if len(eligible) < sample_size:
        raise ValueError(f"Only {len(eligible)} eligible rows for sample_size={sample_size}")
    eligible["selection_key"] = eligible["object_id"].map(
        lambda value: _selection_key(str(value), seed)
    )
    selected = eligible.sort_values(["selection_key", "object_id"]).head(sample_size).copy()
    selected["selection_rank"] = np.arange(len(selected), dtype=np.int64)
    selected = selected.drop(columns="selection_key")
    if set(selected["split"]) != {"train", "validation"}:
        raise AssertionError("Smoke sample must contain train and validation rows")
    return selected.reset_index(drop=True)


def prepare_smoke_source(
    source_spec: Mapping[str, Any],
    exclusion_spec: Mapping[str, Any],
    output_dir: Path,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize one common image/vector set and complete exclusion coverage."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Smoke data output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_pinned_dataset(
        source_spec["repo_id"], source_spec["revision"], source_spec["split"]
    )
    source_columns = {
        source_spec["object_id_column"],
        source_spec["survey_column"],
        source_spec["ra_column"],
        source_spec["dec_column"],
        source_spec["image_column"],
        source_spec["image_embedding_column"],
        source_spec["released_text_column"],
        source_spec["released_embedding_column"],
    }
    missing = source_columns - set(dataset.column_names)
    if missing:
        raise ValueError(f"Smoke source dataset missing columns: {sorted(missing)}")

    metadata_columns = [
        source_spec["object_id_column"],
        source_spec["survey_column"],
        source_spec["ra_column"],
        source_spec["dec_column"],
    ]
    metadata = dataset.select_columns(metadata_columns).to_pandas()
    metadata = metadata.rename(
        columns={
            source_spec["object_id_column"]: "object_id",
            source_spec["survey_column"]: "survey",
            source_spec["ra_column"]: "ra",
            source_spec["dec_column"]: "dec",
        }
    )
    metadata["object_id"] = metadata["object_id"].astype(str)
    metadata["source_row_id"] = np.arange(len(metadata), dtype=np.int64)

    screen = pd.read_parquet(Path(exclusion_spec["caption_screen_labels"]))
    if "object_id" not in screen.columns:
        raise ValueError("Caption-screen exclusion artifact requires object_id")
    exact_coverage = exact_exclusion_coverage(
        metadata["object_id"], "caption_screen_64", screen["object_id"].astype(str)
    )
    benchmark_frames = {
        name: pd.read_parquet(Path(path))
        for name, path in exclusion_spec["benchmark_coordinates"].items()
    }
    coordinate_coverage = coordinate_exclusion_coverage(
        metadata,
        benchmark_frames,
        radius_arcsec=float(exclusion_spec["radius_arcsec"]),
    )
    coverage = pd.concat([exact_coverage, coordinate_coverage], ignore_index=True)
    expected_coverage_rows = len(screen) + sum(len(frame) for frame in benchmark_frames.values())
    assert_exclusion_coverage(coverage, expected_rows=expected_coverage_rows)
    coverage.to_parquet(output_dir / "exclusion_coverage.parquet", index=False)

    matched = coverage.loc[coverage["status"].eq("matched"), "source_object_id"].astype(str)
    manifest = build_manifest(
        metadata,
        {"caption_screen_and_retrieval_benchmarks": matched},
        seed=seed,
        train_ratio=float(source_spec["train_ratio"]),
        image_embedding_version=f"{source_spec['repo_id']}@{source_spec['revision']}",
    )
    selected_manifest = select_smoke_rows(manifest, int(source_spec["sample_size"]), seed)
    write_manifest(selected_manifest, output_dir / "manifest.parquet")

    image_dir = output_dir / "images"
    image_dir.mkdir()
    selected_rows = dataset.select(selected_manifest["source_row_id"].astype(int).tolist())
    manifest_by_id = selected_manifest.set_index("object_id")
    records = []
    for row in selected_rows:
        object_id = str(row[source_spec["object_id_column"]])
        image_name = hashlib.sha256(object_id.encode("utf-8")).hexdigest() + ".png"
        image_path = image_dir / image_name
        _to_image(row[source_spec["image_column"]]).save(image_path)
        manifest_row = manifest_by_id.loc[object_id]
        image_embedding = np.asarray(
            row[source_spec["image_embedding_column"]], dtype=np.float32
        )
        released_embedding = np.asarray(
            row[source_spec["released_embedding_column"]], dtype=np.float32
        )
        if image_embedding.size != 768:
            raise ValueError(f"Expected 768 AION dimensions for object_id={object_id}")
        if released_embedding.size != 3072:
            raise ValueError(f"Expected 3072 OpenAI dimensions for object_id={object_id}")
        records.append(
            {
                "object_id": object_id,
                "survey": str(row[source_spec["survey_column"]]),
                "ra": float(row[source_spec["ra_column"]]),
                "dec": float(row[source_spec["dec_column"]]),
                "source_row_id": int(manifest_row["source_row_id"]),
                "split": str(manifest_row["split"]),
                "image_path": str(image_path.resolve()),
                "image_embedding": image_embedding.tolist(),
                "released_summary": str(row[source_spec["released_text_column"]]),
                "released_openai_embedding": released_embedding.tolist(),
            }
        )
    source_frame = pd.DataFrame(records).sort_values("object_id").reset_index(drop=True)
    if set(source_frame["object_id"]) != set(selected_manifest["object_id"]):
        raise AssertionError("Materialized source rows do not match the smoke manifest")
    source_frame.to_parquet(output_dir / "source_rows.parquet", index=False)
    (output_dir / "data_summary.json").write_text(
        json.dumps(
            {
                "source_rows": len(source_frame),
                "train_rows": int(source_frame["split"].eq("train").sum()),
                "validation_rows": int(source_frame["split"].eq("validation").sum()),
                "exclusion_rows": len(coverage),
                "matched_exclusions": int(coverage["status"].eq("matched").sum()),
                "absent_exclusions": int(coverage["status"].eq("absent").sum()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return source_frame, selected_manifest
