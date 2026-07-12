"""Released-checkpoint loading, query freezing, and equivalence checks."""

from __future__ import annotations

import json
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .model import AIONSearchModel, ModelConfig


def resolve_released_files(
    repo_id: str,
    revision: str,
    config_file: str = "config.json",
    weights_file: str = "model.safetensors",
) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id, config_file, revision=revision)
    weights_path = hf_hub_download(repo_id, weights_file, revision=revision)
    return Path(config_path), Path(weights_path)


def load_reimplemented_reference(
    config_path: Path,
    weights_path: Path,
    device: str = "cpu",
) -> AIONSearchModel:
    from safetensors.torch import load_file

    config_data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = ModelConfig.from_mapping(config_data)
    model = AIONSearchModel(config)
    model.load_state_dict(load_file(str(weights_path)), strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


def load_author_reference(
    orig_repo: Path,
    config_path: Path,
    weights_path: Path,
) -> torch.nn.Module:
    from safetensors.torch import load_file

    repo = str(Path(orig_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from aionsearch.clip_model import AIONSearchClipModel

    config_data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    ModelConfig.from_mapping(config_data)
    model = AIONSearchClipModel(**config_data)
    model.load_state_dict(load_file(str(weights_path)), strict=True)
    return model.to(device="cpu", dtype=torch.float32).eval()


def assert_reference_equivalence(
    reimplemented: AIONSearchModel,
    author_model: torch.nn.Module,
    seed: int = 20260712,
    batch_size: int = 7,
    atol: float = 1e-6,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    config = reimplemented.config
    image = torch.randn(batch_size, config.image_input_dim, dtype=torch.float32) * 3.0 + 1.5
    text = torch.randn(batch_size, config.text_input_dim, dtype=torch.float32) * 2.0 - 0.5
    batch = {"image_embedding": image, "text_embedding": text}
    with torch.inference_mode():
        expected = author_model(batch)
        actual = reimplemented(batch)
    maxima: Dict[str, float] = {}
    for key in ("image_features", "text_features", "logits_per_image", "logits_per_text", "logit_scale"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=atol)
        maxima[key] = float(torch.max(torch.abs(actual[key] - expected[key])).item())
    return maxima


def freeze_openai_queries(
    query_rows: Sequence[Mapping[str, str]],
    output_path: Path,
    model: str = "text-embedding-3-large",
) -> None:
    """Make the single preregistered R-OAI call through OpenRouter, locally."""
    if Path(output_path).exists():
        raise FileExistsError(f"Query cache already exists: {output_path}")
    api_key = os.environ.get("OPEN_ROUTER_KEY")
    if not api_key:
        raise RuntimeError("OPEN_ROUTER_KEY is required to freeze R-OAI queries")
    from openai import OpenAI

    texts = [row["text"] for row in query_rows]
    api_model = f"openai/{model}"
    response = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    ).embeddings.create(
        input=texts,
        model=api_model,
        extra_body={
            "provider": {
                "order": ["openai"],
                "allow_fallbacks": False,
                "data_collection": "deny",
            }
        },
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    records = []
    response_items = sorted(response.data, key=lambda item: item.index)
    if len(response_items) != len(query_rows):
        raise RuntimeError(
            f"Embedding response row mismatch: expected {len(query_rows)}, got {len(response_items)}"
        )
    indices = [int(item.index) for item in response_items]
    if indices != list(range(len(query_rows))):
        raise RuntimeError(f"Embedding response indices are not contiguous: {indices}")
    for row, item in zip(query_rows, response_items):
        vector = np.asarray(item.embedding, dtype=np.float32)
        if vector.size != 3072:
            raise RuntimeError(f"Expected 3072-dimensional OpenAI embedding, got {vector.size}")
        records.append(
            {
                **dict(row),
                "provider": "openai_via_openrouter",
                "model": model,
                "api_model": api_model,
                "requested_at": timestamp,
                "response_model": getattr(response, "model", None),
                "response_id": getattr(response, "id", None),
                "dimension": int(vector.size),
                "embedding": vector.tolist(),
            }
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_parquet(output_path, index=False)
    query_manifest = [
        {
            "object_id": record["object_id"],
            "text_sha256": hashlib.sha256(record["text"].encode("utf-8")).hexdigest(),
        }
        for record in records
    ]
    usage = getattr(response, "usage", None)
    if usage is not None and hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "rows": len(frame),
                "gateway": "openrouter",
                "provider_policy": {
                    "order": ["openai"],
                    "allow_fallbacks": False,
                    "data_collection": "deny",
                },
                "requested_model": model,
                "api_model": api_model,
                "response_model": getattr(response, "model", None),
                "response_id": getattr(response, "id", None),
                "requested_at": timestamp,
                "usage": usage,
                "query_manifest": query_manifest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
