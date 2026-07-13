"""Component-faithful mean-embedding AION-Search projection model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


REQUIRED_MODEL_KEYS = {
    "image_input_dim",
    "text_input_dim",
    "embedding_dim",
    "image_hidden_dim",
    "text_hidden_dim",
    "dropout",
    "use_mean_embeddings",
}


@dataclass(frozen=True)
class ModelConfig:
    image_input_dim: int
    text_input_dim: int
    embedding_dim: int
    image_hidden_dim: int
    text_hidden_dim: int
    dropout: float
    use_mean_embeddings: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModelConfig":
        missing = REQUIRED_MODEL_KEYS - set(values)
        unknown = set(values) - REQUIRED_MODEL_KEYS
        if missing:
            raise ValueError(f"Released model config missing required keys: {sorted(missing)}")
        if unknown:
            raise ValueError(f"Released model config has unsupported keys: {sorted(unknown)}")
        if values["use_mean_embeddings"] is not True:
            raise ValueError("Only use_mean_embeddings=true is in scope")
        return cls(**{key: values[key] for key in REQUIRED_MODEL_KEYS})


class ResidualProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        normalize_input: bool,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.normalize_input = normalize_input
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            values = F.normalize(values, dim=-1, eps=1e-6)
        hidden = self.fc_in(values)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return F.normalize(self.fc_out(hidden), dim=-1, eps=1e-3)


class AIONSearchModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        temperature_initial_scale: float = 1.0 / 0.07,
        temperature_max_scale: float = 100.0,
    ) -> None:
        super().__init__()
        if temperature_initial_scale <= 0 or temperature_max_scale <= 0:
            raise ValueError("Temperature scales must be positive")
        self.config = config
        self.temperature_max_scale = float(temperature_max_scale)
        self.image_projector = ResidualProjector(
            config.image_input_dim,
            config.image_hidden_dim,
            config.embedding_dim,
            config.dropout,
            normalize_input=True,
        )
        self.text_projector = ResidualProjector(
            config.text_input_dim,
            config.text_hidden_dim,
            config.embedding_dim,
            config.dropout,
            normalize_input=False,
        )
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(float(temperature_initial_scale), dtype=torch.float32))
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_features = self.image_projector(batch["image_embedding"])
        text_features = self.text_projector(batch["text_embedding"])
        logit_scale = self.logit_scale.exp().clamp(max=self.temperature_max_scale)
        logits_per_image = logit_scale * image_features @ text_features.T
        return {
            "image_features": image_features,
            "text_features": text_features,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_image.T,
            "logit_scale": logit_scale,
        }

    @staticmethod
    def contrastive_loss(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        batch_size = outputs["logits_per_image"].shape[0]
        labels = torch.arange(batch_size, device=outputs["logits_per_image"].device)
        image_loss = F.cross_entropy(outputs["logits_per_image"], labels)
        text_loss = F.cross_entropy(outputs["logits_per_text"], labels)
        return (image_loss + text_loss) / 2.0
