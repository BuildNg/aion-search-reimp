"""Small paired-embedding trainer with Recall@10 checkpoint ownership."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .metrics import recall_at_k
from .model import AIONSearchModel, ModelConfig


@dataclass(frozen=True)
class TrainingSpec:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    gradient_clip_max_norm: float
    num_workers: int
    minimum_steps_per_epoch: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TrainingSpec":
        return cls(
            batch_size=int(values["batch_size"]),
            epochs=int(values["epochs"]),
            learning_rate=float(values["learning_rate"]),
            weight_decay=float(values["weight_decay"]),
            gradient_clip_max_norm=float(values["gradient_clip_max_norm"]),
            num_workers=int(values["num_workers"]),
            minimum_steps_per_epoch=int(values["minimum_steps_per_epoch"]),
        )


class _EmbeddingPairs(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.images = torch.tensor(
            np.asarray(frame["image_embedding"].tolist(), dtype=np.float32)
        )
        self.texts = torch.tensor(
            np.asarray(frame["text_embedding"].tolist(), dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "image_embedding": self.images[index],
            "text_embedding": self.texts[index],
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assemble_condition_rows(source: pd.DataFrame, text_cache: pd.DataFrame) -> pd.DataFrame:
    required_source = {"object_id", "split", "image_embedding"}
    required_text = {"object_id", "embedding"}
    if required_source - set(source.columns):
        raise ValueError("Condition source is missing object_id, split, or image_embedding")
    if required_text - set(text_cache.columns):
        raise ValueError("Text cache is missing object_id or embedding")
    if source["object_id"].astype(str).duplicated().any():
        raise ValueError("Condition source contains duplicate object IDs")
    if text_cache["object_id"].astype(str).duplicated().any():
        raise ValueError("Text cache contains duplicate object IDs")

    source_rows = source.loc[:, ["object_id", "split", "image_embedding"]].copy()
    source_rows["object_id"] = source_rows["object_id"].astype(str)
    text_rows = text_cache.loc[:, ["object_id", "embedding"]].copy()
    text_rows["object_id"] = text_rows["object_id"].astype(str)
    text_rows = text_rows.rename(columns={"embedding": "text_embedding"})
    joined = source_rows.merge(text_rows, on="object_id", how="left", validate="one_to_one")
    if joined["text_embedding"].isna().any():
        missing = joined.loc[joined["text_embedding"].isna(), "object_id"].tolist()
        raise ValueError(f"Text cache missing manifest objects: {missing[:5]}")
    if len(joined) != len(source_rows):
        raise AssertionError("Condition join changed the source row count")
    return joined.sort_values("object_id").reset_index(drop=True)


def _project_validation(
    model: AIONSearchModel,
    frame: pd.DataFrame,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    batch = {
        "image_embedding": torch.tensor(
            np.asarray(frame["image_embedding"].tolist(), dtype=np.float32), device=device
        ),
        "text_embedding": torch.tensor(
            np.asarray(frame["text_embedding"].tolist(), dtype=np.float32), device=device
        ),
    }
    with torch.inference_mode():
        outputs = model(batch)
        loss = float(model.contrastive_loss(outputs).item())
    images = outputs["image_features"].cpu().numpy()
    texts = outputs["text_features"].cpu().numpy()
    recall = recall_at_k(texts, images, np.arange(len(frame)), ks=(10,))[10]
    return loss, recall, images, texts


def _noncollapsed(features: np.ndarray) -> bool:
    if features.shape[0] < 2:
        return False
    mean_feature_std = float(np.std(features, axis=0).mean())
    similarities = features @ features.T
    mean_off_diagonal = float(similarities.sum() - np.trace(similarities)) / float(
        features.shape[0] * (features.shape[0] - 1)
    )
    return mean_feature_std > 1e-4 and mean_off_diagonal < 0.99


def train_condition(
    condition_rows: pd.DataFrame,
    model_config: ModelConfig,
    training_spec: TrainingSpec,
    output_dir: Path,
    seed: int,
    temperature_initial_scale: float,
    temperature_max_scale: float,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Condition output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = condition_rows[condition_rows["split"].eq("train")].reset_index(drop=True)
    validation_rows = condition_rows[
        condition_rows["split"].eq("validation")
    ].reset_index(drop=True)
    if train_rows.empty or validation_rows.empty:
        raise ValueError("Training requires non-empty train and validation splits")
    steps_per_epoch = math.ceil(len(train_rows) / training_spec.batch_size)
    if steps_per_epoch < training_spec.minimum_steps_per_epoch:
        raise ValueError(
            f"Only {steps_per_epoch} optimizer steps per epoch; "
            f"minimum is {training_spec.minimum_steps_per_epoch}"
        )

    seed_everything(seed)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = AIONSearchModel(
        model_config,
        temperature_initial_scale=temperature_initial_scale,
        temperature_max_scale=temperature_max_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_spec.learning_rate,
        weight_decay=training_spec.weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        _EmbeddingPairs(train_rows),
        batch_size=training_spec.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=training_spec.num_workers,
        generator=generator,
    )

    history = []
    best_recall = -1.0
    best_validation_loss = float("inf")
    best_epoch = None
    checkpoint_path = output_dir / "best_checkpoint.pt"
    for epoch in range(1, training_spec.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss = model.contrastive_loss(outputs)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_spec.gradient_clip_max_norm
            )
            optimizer.step()
            losses.append(float(loss.item()))
        validation_loss, validation_recall, _, _ = _project_validation(
            model, validation_rows, device
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            "validation_recall_at_10": validation_recall,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "logit_scale": float(
                model.logit_scale.detach().exp().clamp(max=temperature_max_scale).item()
            ),
            "optimizer_steps": len(losses),
        }
        history.append(row)
        improved = validation_recall > best_recall or (
            validation_recall == best_recall and validation_loss < best_validation_loss
        )
        if improved:
            best_recall = validation_recall
            best_validation_loss = validation_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config.__dict__,
                    "seed": seed,
                    "validation_recall_at_10": validation_recall,
                    "validation_loss": validation_loss,
                    "temperature_initial_scale": temperature_initial_scale,
                    "temperature_max_scale": temperature_max_scale,
                },
                checkpoint_path,
            )

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_loss, best_recall, image_features, text_features = _project_validation(
        model, validation_rows, device
    )
    random_recall = min(10 / len(validation_rows), 1.0)
    gate = {
        "finite_loss": bool(np.isfinite(history_frame["train_loss"]).all()),
        "image_embeddings_noncollapsed": _noncollapsed(image_features),
        "text_embeddings_noncollapsed": _noncollapsed(text_features),
        "validation_recall_at_10": best_recall,
        "random_recall_at_10": random_recall,
        "validation_above_random": bool(best_recall > random_recall),
        "best_validation_loss": best_loss,
        "best_epoch": best_epoch,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "steps_per_epoch": steps_per_epoch,
        "joined_rows": len(condition_rows),
    }
    gate["passed"] = all(
        gate[key]
        for key in (
            "finite_loss",
            "image_embeddings_noncollapsed",
            "text_embeddings_noncollapsed",
            "validation_above_random",
        )
    )
    (output_dir / "smoke_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    return gate
