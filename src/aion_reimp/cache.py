"""Versioned caption and embedding caches with strict provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd


EMBEDDING_COLUMNS = {
    "object_id",
    "text",
    "embedding",
    "role",
    "model_id",
    "revision",
    "instruction",
    "instruction_hash",
    "source_checksum",
    "output_dim",
    "normalized",
}


@dataclass(frozen=True)
class NormalizationPolicy:
    """Validation policy recorded with every embedding cache."""

    required: bool
    atol: float = 1e-3

    def __post_init__(self) -> None:
        if self.atol <= 0:
            raise ValueError("Normalization tolerance must be positive")

    def as_dict(self) -> Dict[str, Any]:
        return {"required": self.required, "atol": self.atol}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_normalized(vector: np.ndarray, policy: NormalizationPolicy) -> bool:
    return bool(np.isclose(np.linalg.norm(vector), 1.0, atol=policy.atol, rtol=0.0))


def vector_checksum(value: Any) -> str:
    """Hash canonical float32 vector bytes, including shape."""
    vector = np.asarray(value, dtype="<f4")
    payload = vector.shape.__repr__().encode("ascii") + b"\0" + vector.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _row_fingerprint(row: Any) -> bytes:
    payload = {
        "object_id": str(row.object_id),
        "text_hash": sha256_text(str(row.text)),
        "role": str(row.role),
        "model_id": str(row.model_id),
        "revision": str(row.revision),
        "instruction_hash": str(row.instruction_hash),
        "source_checksum": str(row.source_checksum),
        "output_dim": int(row.output_dim),
        "normalized": bool(row.normalized),
        "vector_checksum": vector_checksum(row.embedding),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).digest()


def _combine_row_fingerprints(row_fingerprints: Any) -> str:
    digest = hashlib.sha256()
    for row_fingerprint in sorted(row_fingerprints):
        digest.update(row_fingerprint)
    return digest.hexdigest()


def validate_embedding_cache(
    frame: pd.DataFrame,
    normalization_policy: NormalizationPolicy,
) -> None:
    missing = EMBEDDING_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Embedding cache missing columns: {sorted(missing)}")
    if frame["object_id"].astype(str).duplicated().any():
        raise ValueError("Embedding cache contains duplicate object_id values")
    invalid_roles = set(frame["role"].astype(str)) - {"document", "query"}
    if invalid_roles:
        raise ValueError(f"Invalid embedding roles: {sorted(invalid_roles)}")
    documents = frame[frame["role"] == "document"]
    instructed_documents = documents[documents["instruction"].fillna("").astype(str).ne("")]
    if not instructed_documents.empty:
        raise ValueError("Document embedding cache contains a query instruction")
    for row in frame.itertuples(index=False):
        vector = np.asarray(row.embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size != int(row.output_dim):
            raise ValueError(f"Embedding dimension mismatch for object_id={row.object_id}")
        measured_normalized = _is_normalized(vector, normalization_policy)
        if bool(row.normalized) != measured_normalized:
            raise ValueError(
                "Stored normalized flag contradicts measured norm for "
                f"object_id={row.object_id}: flag={bool(row.normalized)} "
                f"norm={np.linalg.norm(vector):.9f} atol={normalization_policy.atol}"
            )
        if normalization_policy.required and not measured_normalized:
            raise ValueError(
                f"Embedding is not normalized for object_id={row.object_id}: "
                f"norm={np.linalg.norm(vector):.9f} atol={normalization_policy.atol}"
            )
        expected_hash = sha256_text(str(row.instruction or ""))
        if str(row.instruction_hash) != expected_hash:
            raise ValueError(f"Instruction hash mismatch for object_id={row.object_id}")


def load_cache_subset(path: Path, object_ids: Any) -> pd.DataFrame:
    """Read exactly the requested object_id rows from a parquet cache."""
    identifiers = [str(value) for value in object_ids]
    frame = pd.read_parquet(path, filters=[("object_id", "in", identifiers)])
    if set(frame["object_id"].astype(str)) != set(identifiers):
        raise ValueError(f"Cache subset from {path} does not match requested object IDs")
    return frame


def cache_fingerprint(frame: pd.DataFrame) -> str:
    return _combine_row_fingerprints(
        _row_fingerprint(row) for row in frame.itertuples(index=False)
    )


def write_embedding_cache(
    frame: pd.DataFrame,
    path: Path,
    normalization_policy: NormalizationPolicy,
    metadata: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
) -> str:
    validate_embedding_cache(frame, normalization_policy)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Embedding cache exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    fingerprint = cache_fingerprint(frame)
    meta: Dict[str, Any] = {
        "rows": len(frame),
        "fingerprint": fingerprint,
        "normalization_policy": normalization_policy.as_dict(),
    }
    meta.update(dict(metadata or {}))
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return fingerprint


def ingest_released_embeddings(
    source: pd.DataFrame,
    model_id: str = "text-embedding-3-large",
    revision: str = "released-vector-no-endpoint-revision",
    text_column: str = "summary",
    embedding_column: str = "summary_text_embedding",
    normalization_policy: NormalizationPolicy = NormalizationPolicy(required=True),
) -> pd.DataFrame:
    required = {"object_id", text_column, embedding_column}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Released embedding source missing columns: {sorted(missing)}")
    records = []
    for row in source.loc[:, ["object_id", text_column, embedding_column]].itertuples(index=False):
        object_id, text, embedding = row
        vector = np.asarray(embedding, dtype=np.float32)
        records.append(
            {
                "object_id": str(object_id),
                "text": str(text),
                "embedding": vector.tolist(),
                "role": "document",
                "model_id": model_id,
                "revision": revision,
                "instruction": "",
                "instruction_hash": sha256_text(""),
                "source_checksum": sha256_text(str(text)),
                "output_dim": int(vector.size),
                "normalized": _is_normalized(vector, normalization_policy),
            }
        )
    frame = pd.DataFrame(records)
    validate_embedding_cache(frame, normalization_policy)
    return frame


def derive_fp32_normalized_cache(
    source_path: Path,
    output_path: Path,
    normalization_policy: NormalizationPolicy = NormalizationPolicy(required=True),
) -> str:
    """Create a new fp32-renormalized cache with explicit source lineage."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_path = Path(source_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Derived cache exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    source_policy = NormalizationPolicy(required=False, atol=normalization_policy.atol)
    source_row_fingerprints = []
    derived_row_fingerprints = []
    seen_object_ids = set()
    writer = None
    rows = 0
    try:
        parquet_file = pq.ParquetFile(source_path)
        for batch in parquet_file.iter_batches(batch_size=1024):
            source = batch.to_pandas()
            batch_object_ids = set(source["object_id"].astype(str))
            overlap = seen_object_ids & batch_object_ids
            if overlap:
                raise ValueError(
                    f"Embedding cache contains duplicate object_id={sorted(overlap)[0]}"
                )
            seen_object_ids.update(batch_object_ids)
            source_row_fingerprints.extend(
                _row_fingerprint(row) for row in source.itertuples(index=False)
            )
            source_for_validation = source.copy()
            source_for_validation["normalized"] = source_for_validation["embedding"].map(
                lambda value: _is_normalized(
                    np.asarray(value, dtype=np.float32), source_policy
                )
            )
            validate_embedding_cache(source_for_validation, source_policy)

            derived = source.copy()
            matrix = np.stack(
                [np.asarray(value, dtype=np.float32) for value in derived["embedding"]]
            )
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            invalid = ~np.isfinite(norms[:, 0]) | (norms[:, 0] == 0.0)
            if invalid.any():
                object_id = derived.loc[np.flatnonzero(invalid)[0], "object_id"]
                raise ValueError(f"Cannot normalize invalid vector for object_id={object_id}")
            matrix /= norms
            derived["embedding"] = list(matrix)
            derived["normalized"] = True
            validate_embedding_cache(derived, normalization_policy)
            derived_row_fingerprints.extend(
                _row_fingerprint(row) for row in derived.itertuples(index=False)
            )

            table = pa.Table.from_pandas(derived, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary_path, table.schema)
            writer.write_table(table)
            rows += len(derived)
    except Exception:
        if writer is not None:
            writer.close()
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    else:
        if writer is None:
            raise ValueError(f"Source cache has no rows: {source_path}")
        writer.close()

    source_fingerprint = _combine_row_fingerprints(source_row_fingerprints)
    fingerprint = _combine_row_fingerprints(derived_row_fingerprints)
    temporary_path.replace(output_path)

    legacy_meta_path = source_path.with_suffix(source_path.suffix + ".meta.json")
    legacy_fingerprint = None
    if legacy_meta_path.exists():
        legacy_fingerprint = json.loads(legacy_meta_path.read_text(encoding="utf-8")).get(
            "fingerprint"
        )
    metadata: Dict[str, Any] = {
        "rows": rows,
        "fingerprint": fingerprint,
        "normalization_policy": normalization_policy.as_dict(),
        "source_path": str(source_path),
        "source_fingerprint": source_fingerprint,
        "source_legacy_fingerprint": legacy_fingerprint,
        "transform": "fp32_l2_renormalize_v1",
    }
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return fingerprint
