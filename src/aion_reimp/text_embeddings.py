"""Qwen3 document/query encoding with explicit instruction asymmetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .cache import sha256_text, validate_embedding_cache


def format_query(text: str, instruction: str) -> str:
    if not instruction.strip():
        raise ValueError("Query instruction must not be empty")
    return f"Instruct: {instruction.strip()}\nQuery: {text}"


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = bool(attention_mask[:, -1].sum().item() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]


@dataclass(frozen=True)
class EmbeddingSpec:
    model_id: str
    revision: str
    dimension: int = 1024
    normalize: bool = True
    max_length: int = 8192
    query_instruction: str = ""

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EmbeddingSpec":
        return cls(
            model_id=values["model_id"],
            revision=values["revision"],
            dimension=values["dimension"],
            normalize=values["normalize"],
            max_length=values["max_length"],
            query_instruction=values["query_instruction"],
        )


class QwenEmbedder:
    """Cluster-only wrapper. Construction loads model weights."""

    def __init__(self, spec: EmbeddingSpec, dtype: str = "bfloat16") -> None:
        from transformers import AutoModel, AutoTokenizer

        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            padding_side="left",
        )
        self.model = AutoModel.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            torch_dtype=getattr(torch, dtype),
            device_map="auto",
        ).eval()

    def encode(self, texts: Sequence[str], role: str) -> np.ndarray:
        if role not in {"document", "query"}:
            raise ValueError("role must be document or query")
        encoded_texts = list(texts)
        if role == "query":
            encoded_texts = [format_query(text, self.spec.query_instruction) for text in texts]
        tokens = self.tokenizer(
            encoded_texts,
            padding=True,
            truncation=True,
            max_length=self.spec.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.model.device) for key, value in tokens.items()}
        with torch.inference_mode():
            outputs = self.model(**tokens)
            embeddings = last_token_pool(outputs.last_hidden_state, tokens["attention_mask"])
            if self.spec.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.float().cpu().numpy()
        if embeddings.shape[1] != self.spec.dimension:
            raise ValueError(
                f"Expected {self.spec.dimension} embedding dimensions, got {embeddings.shape[1]}"
            )
        return embeddings


def embedding_frame(
    object_ids: Sequence[str],
    texts: Sequence[str],
    vectors: np.ndarray,
    role: str,
    spec: EmbeddingSpec,
) -> pd.DataFrame:
    if len(object_ids) != len(texts) or len(object_ids) != len(vectors):
        raise ValueError("object_ids, texts, and vectors must have equal rows")
    instruction = "" if role == "document" else spec.query_instruction
    records: List[dict] = []
    for object_id, text, vector in zip(object_ids, texts, vectors):
        array = np.asarray(vector, dtype=np.float32)
        records.append(
            {
                "object_id": str(object_id),
                "text": str(text),
                "embedding": array.tolist(),
                "role": role,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "instruction": instruction,
                "instruction_hash": sha256_text(instruction),
                "source_checksum": sha256_text(str(text)),
                "output_dim": int(array.size),
                "normalized": bool(np.isclose(np.linalg.norm(array), 1.0, atol=1e-3)),
            }
        )
    frame = pd.DataFrame(records)
    validate_embedding_cache(frame)
    return frame
