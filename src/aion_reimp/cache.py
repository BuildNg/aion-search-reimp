"""Versioned caption and embedding caches with strict provenance."""

from __future__ import annotations

import hashlib
import json
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_embedding_cache(frame: pd.DataFrame) -> None:
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
        expected_hash = sha256_text(str(row.instruction or ""))
        if str(row.instruction_hash) != expected_hash:
            raise ValueError(f"Instruction hash mismatch for object_id={row.object_id}")


def cache_fingerprint(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.sort_values("object_id").itertuples(index=False):
        rows.append(
            {
                "object_id": str(row.object_id),
                "text_hash": sha256_text(str(row.text)),
                "role": str(row.role),
                "model_id": str(row.model_id),
                "revision": str(row.revision),
                "instruction_hash": str(row.instruction_hash),
                "source_checksum": str(row.source_checksum),
                "output_dim": int(row.output_dim),
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_embedding_cache(
    frame: pd.DataFrame,
    path: Path,
    metadata: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
) -> str:
    validate_embedding_cache(frame)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Embedding cache exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    fingerprint = cache_fingerprint(frame)
    meta: Dict[str, Any] = {"rows": len(frame), "fingerprint": fingerprint}
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
                "normalized": bool(np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3)),
            }
        )
    frame = pd.DataFrame(records)
    validate_embedding_cache(frame)
    return frame
