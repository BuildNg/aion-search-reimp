"""Direct 64-image structured-answer audit against human decision paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


def decision_tree_answers(value: Any) -> Dict[str, str]:
    nodes = json.loads(value) if isinstance(value, str) else value
    if not isinstance(nodes, list):
        raise ValueError("decision_tree must decode to a list")
    answers: Dict[str, str] = {}
    for node in nodes:
        answers[str(node["question"])] = str(node["answer"])
    return answers


def audit_rows(captions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    required_captions = {"object_id", "answers_json"}
    required_labels = {"object_id", "decision_tree"}
    if required_captions - set(captions.columns):
        raise ValueError("Caption rows require object_id and answers_json")
    if required_labels - set(labels.columns):
        raise ValueError("Label rows require object_id and decision_tree")
    joined = labels.loc[:, ["object_id", "decision_tree"]].merge(
        captions.loc[:, ["object_id", "answers_json"]],
        on="object_id",
        how="left",
        validate="one_to_one",
    )
    if joined["answers_json"].isna().any():
        missing = joined.loc[joined["answers_json"].isna(), "object_id"].tolist()
        raise ValueError(f"Missing captions for label objects: {missing[:5]}")

    records = []
    for row in joined.itertuples(index=False):
        gold = decision_tree_answers(row.decision_tree)
        predicted = json.loads(row.answers_json)
        for question, answer in gold.items():
            prediction = predicted.get(question, "missing")
            records.append(
                {
                    "object_id": str(row.object_id),
                    "question": question,
                    "gold": answer,
                    "prediction": prediction,
                    "correct": bool(prediction == answer),
                    "abstained": bool(prediction == "uncertain"),
                }
            )
    return pd.DataFrame(records)


def _cluster_bootstrap_interval(
    rows: pd.DataFrame,
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    object_ids = rows["object_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    means = []
    grouped = {object_id: group for object_id, group in rows.groupby("object_id")}
    for _ in range(samples):
        draw = rng.choice(object_ids, size=len(object_ids), replace=True)
        values = np.concatenate([grouped[object_id]["correct"].to_numpy(dtype=float) for object_id in draw])
        means.append(float(values.mean()))
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def audit_metrics(rows: pd.DataFrame, bootstrap_samples: int = 2000, seed: int = 0) -> Dict[str, Any]:
    if rows.empty:
        raise ValueError("Caption audit has no scored rows")
    low, high = _cluster_bootstrap_interval(rows, bootstrap_samples, seed)
    result: Dict[str, Any] = {
        "objects": int(rows["object_id"].nunique()),
        "answers": int(len(rows)),
        "accuracy": float(rows["correct"].mean()),
        "accuracy_ci95": [low, high],
        "abstention_rate": float(rows["abstained"].mean()),
        "by_question": {},
    }
    for question, group in rows.groupby("question", sort=True):
        q_low, q_high = _cluster_bootstrap_interval(group, bootstrap_samples, seed)
        result["by_question"][question] = {
            "answers": int(len(group)),
            "accuracy": float(group["correct"].mean()),
            "accuracy_ci95": [q_low, q_high],
            "abstention_rate": float(group["abstained"].mean()),
        }
    return result


def write_audit(
    captions: pd.DataFrame,
    labels: pd.DataFrame,
    output_dir: Path,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = audit_rows(captions, labels)
    rows.to_csv(output_dir / "caption_audit_rows.csv", index=False)
    (output_dir / "caption_audit_metrics.json").write_text(
        json.dumps(audit_metrics(rows, bootstrap_samples, seed), indent=2, sort_keys=True),
        encoding="utf-8",
    )
