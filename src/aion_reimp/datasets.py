"""Bounded data materialization for the frozen 64-image caption screen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from PIL import Image


def load_pinned_dataset(repo_id: str, revision: str, split: str):
    """Load a pinned Hub dataset snapshot without requiring network access."""
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_files_only=True,
    )
    return load_dataset(str(snapshot), split=split)


def load_query_rows(path: Path) -> List[Dict[str, str]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rows: List[Dict[str, str]] = []
    for category, values in data["queries"].items():
        for index, text in enumerate(values):
            rows.append(
                {
                    "object_id": f"{category}:{index}",
                    "category": str(category),
                    "variant": "canonical" if index == 0 else f"paraphrase_{index}",
                    "text": str(text),
                }
            )
    return rows


def _to_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in {3, 4} and array.shape[-1] not in {3, 4}:
        array = np.moveaxis(array[:3], 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        maximum = float(np.nanmax(array)) if array.size else 1.0
        if maximum <= 1.0:
            array = array * 255.0
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")


def materialize_caption_screen(
    repo_id: str,
    revision: str,
    split: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Cluster-only: downloads the frozen 64-row dataset and writes PNGs/labels."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Caption screen output is not empty: {output_dir}")
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_pinned_dataset(repo_id, revision, split)
    if len(dataset) != 64:
        raise ValueError(f"Expected 64 caption-screen rows, found {len(dataset)}")
    records = []
    for row in dataset:
        object_id = str(row["object_id"])
        image_value = row.get("image_rgb", row.get("image_array"))
        if image_value is None:
            raise ValueError(f"No image field for {object_id}")
        image_path = image_dir / f"{object_id}.png"
        _to_image(image_value).save(image_path)
        decision_tree = row["decision_tree"]
        if not isinstance(decision_tree, str):
            decision_tree = json.dumps(decision_tree)
        records.append(
            {
                "object_id": object_id,
                "image_path": str(image_path.resolve()),
                "decision_tree": decision_tree,
                "ra": row.get("ra"),
                "dec": row.get("dec"),
            }
        )
    frame = pd.DataFrame(records).sort_values("object_id").reset_index(drop=True)
    frame.to_parquet(output_dir / "caption_screen_labels.parquet", index=False)
    frame.loc[:, ["object_id"]].assign(reason="caption_screen_64").to_csv(
        output_dir / "caption_screen_exclusions.csv", index=False
    )
    return frame


def materialize_benchmark_coordinates(
    benchmarks: List[Dict[str, str]],
    output_dir: Path,
) -> Dict[str, Path]:
    """Cluster-only: freeze benchmark coordinate tables used for exclusion."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    for benchmark in benchmarks:
        dataset = load_pinned_dataset(
            benchmark["repo_id"], benchmark["revision"], "train"
        )
        missing = {"ra", "dec"} - set(dataset.column_names)
        if missing:
            raise ValueError(f"{benchmark['name']} missing coordinate columns: {sorted(missing)}")
        frame = dataset.select_columns(["ra", "dec"]).to_pandas()
        frame.insert(0, "benchmark_row", np.arange(len(frame), dtype=np.int64))
        path = output_dir / f"{benchmark['name']}_coordinates.parquet"
        if path.exists():
            raise FileExistsError(f"Benchmark coordinate artifact exists: {path}")
        frame.to_parquet(path, index=False)
        outputs[benchmark["name"]] = path
    return outputs
