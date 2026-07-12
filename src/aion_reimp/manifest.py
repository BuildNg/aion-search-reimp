"""Immutable object manifests, split assignment, and leakage assertions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import pandas as pd


CORE_COLUMNS = ("object_id", "survey", "ra", "dec", "source_row_id")


def split_fraction(object_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{object_id}|{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def manifest_fingerprint(frame: pd.DataFrame) -> str:
    stable = frame.sort_values("object_id").reset_index(drop=True)
    payload = stable.to_json(orient="records", double_precision=15, force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def combine_exclusion_sets(exclusion_sets: Mapping[str, Iterable[str]]) -> Dict[str, str]:
    reasons: Dict[str, Set[str]] = {}
    for name, values in exclusion_sets.items():
        for value in values:
            reasons.setdefault(str(value), set()).add(str(name))
    return {object_id: ";".join(sorted(names)) for object_id, names in reasons.items()}


def coordinate_exclusion_table(
    source: pd.DataFrame,
    benchmark_coordinates: Mapping[str, pd.DataFrame],
    radius_arcsec: float = 1.0,
) -> pd.DataFrame:
    """Return source object IDs matching any benchmark coordinate catalog."""
    import numpy as np
    from scipy.spatial import cKDTree

    def unit_vectors(frame: pd.DataFrame) -> np.ndarray:
        ra = np.deg2rad(frame["ra"].to_numpy(dtype=float))
        dec = np.deg2rad(frame["dec"].to_numpy(dtype=float))
        cos_dec = np.cos(dec)
        return np.column_stack((cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)))

    required_source = {"object_id", "ra", "dec"}
    if required_source - set(source.columns):
        raise ValueError("Source coordinate table requires object_id, ra, and dec")
    if radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be positive")
    source_vectors = unit_vectors(source)
    records: List[Dict[str, object]] = []
    for name, benchmark in benchmark_coordinates.items():
        if {"ra", "dec"} - set(benchmark.columns):
            raise ValueError(f"Benchmark {name} requires ra and dec")
        if benchmark.empty:
            continue
        chord_distance, _ = cKDTree(unit_vectors(benchmark)).query(source_vectors, k=1)
        angular_radians = 2.0 * np.arcsin(np.clip(chord_distance / 2.0, 0.0, 1.0))
        separation_arcsec = np.rad2deg(angular_radians) * 3600.0
        matched = separation_arcsec <= radius_arcsec
        for object_id, distance in zip(source.loc[matched, "object_id"], separation_arcsec[matched]):
            records.append(
                {
                    "object_id": str(object_id),
                    "reason": str(name),
                    "separation_arcsec": float(distance),
                }
            )
    if not records:
        return pd.DataFrame(columns=["object_id", "reason", "separation_arcsec"])
    matches = pd.DataFrame(records).sort_values(["object_id", "separation_arcsec"])
    grouped = matches.groupby("object_id", as_index=False).agg(
        reason=("reason", lambda values: ";".join(sorted(set(values)))),
        separation_arcsec=("separation_arcsec", "min"),
    )
    return grouped


def assert_no_benchmark_leakage(frame: pd.DataFrame) -> None:
    required = {"object_id", "split", "benchmark_exclusion"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing leakage-check columns: {sorted(missing)}")
    leaked = frame[(frame["benchmark_exclusion"]) & (frame["split"].isin(["train", "validation"]))]
    if not leaked.empty:
        raise AssertionError(f"Benchmark leakage detected for {len(leaked)} objects")


def build_manifest(
    source: pd.DataFrame,
    exclusion_sets: Mapping[str, Iterable[str]],
    seed: int = 42,
    train_ratio: float = 0.8,
    image_embedding_version: str = "polymathic-ai/aion-base",
) -> pd.DataFrame:
    missing = set(CORE_COLUMNS) - set(source.columns)
    if missing:
        raise ValueError(f"Source rows missing columns: {sorted(missing)}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between zero and one")

    frame = source.loc[:, CORE_COLUMNS].copy()
    frame["object_id"] = frame["object_id"].astype(str)
    if frame["object_id"].duplicated().any():
        duplicate = frame.loc[frame["object_id"].duplicated(), "object_id"].iloc[0]
        raise ValueError(f"Duplicate object_id: {duplicate}")

    exclusions = combine_exclusion_sets(exclusion_sets)
    frame["benchmark_exclusion_reason"] = frame["object_id"].map(exclusions).fillna("")
    frame["benchmark_exclusion"] = frame["benchmark_exclusion_reason"].ne("")
    fractions = frame["object_id"].map(lambda value: split_fraction(value, seed))
    frame["split"] = "validation"
    frame.loc[fractions < train_ratio, "split"] = "train"
    frame.loc[frame["benchmark_exclusion"], "split"] = "excluded"
    frame["split_seed"] = int(seed)
    frame["image_embedding_version"] = str(image_embedding_version)
    assert_no_benchmark_leakage(frame)
    return frame.sort_values("object_id").reset_index(drop=True)


def common_object_ids(
    manifests: Sequence[pd.DataFrame],
    split: Optional[str] = None,
) -> Set[str]:
    if not manifests:
        return set()
    sets = []
    for frame in manifests:
        selected = frame if split is None else frame[frame["split"] == split]
        sets.append(set(selected["object_id"].astype(str)))
    return set.intersection(*sets)


def write_manifest(frame: pd.DataFrame, path: Path, overwrite: bool = False) -> str:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    fingerprint = manifest_fingerprint(frame)
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps({"rows": len(frame), "fingerprint": fingerprint}, indent=2),
        encoding="utf-8",
    )
    return fingerprint
