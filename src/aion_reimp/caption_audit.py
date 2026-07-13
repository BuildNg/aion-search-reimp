"""Score text-extracted Galaxy Zoo answers against human decision paths."""

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


def decision_tree_path(value: Any) -> list[str]:
    """Extract the released human-volunteer decision path."""
    nodes = json.loads(value) if isinstance(value, str) else value
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("decision_tree must decode to a non-empty list")
    path = []
    for node in nodes:
        if "node" in node:
            path.append(str(node["node"]))
        elif "question" in node and "answer" in node:
            path.append(f"{node['question']}_{node['answer']}")
        else:
            raise ValueError("decision_tree node lacks node or question/answer")
    return path


def path_audit_rows(captions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the released path-overlap score against human paths."""
    required_captions = {"object_id", "judge_path_json"}
    required_labels = {"object_id", "decision_tree"}
    if required_captions - set(captions.columns):
        raise ValueError("Caption rows require object_id and judge_path_json")
    if required_labels - set(labels.columns):
        raise ValueError("Label rows require object_id and decision_tree")
    joined = labels.loc[:, ["object_id", "decision_tree"]].merge(
        captions.loc[:, ["object_id", "judge_path_json"]],
        on="object_id",
        how="left",
        validate="one_to_one",
    )
    if joined["judge_path_json"].isna().any():
        missing = joined.loc[joined["judge_path_json"].isna(), "object_id"].tolist()
        raise ValueError(f"Missing judge paths for label objects: {missing[:5]}")
    records = []
    for row in joined.itertuples(index=False):
        human_path = decision_tree_path(row.decision_tree)
        judge_path = json.loads(row.judge_path_json)
        if not isinstance(judge_path, list):
            raise ValueError("judge_path_json must decode to a list")
        matches = len(set(judge_path).intersection(human_path))
        records.append(
            {
                "object_id": str(row.object_id),
                "human_path_json": json.dumps(human_path),
                "judge_path_json": json.dumps(judge_path),
                "matched_nodes": matches,
                "human_nodes": len(human_path),
                "score": matches / len(human_path),
            }
        )
    return pd.DataFrame(records)


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
                    "abstained": bool(prediction == "not-stated"),
                }
            )
    return pd.DataFrame(records)


def _cluster_bootstrap_interval(
    rows: pd.DataFrame,
    samples: int,
    seed: int,
    value_column: str = "correct",
) -> Tuple[float, float]:
    object_ids = rows["object_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    means = []
    grouped = {object_id: group for object_id, group in rows.groupby("object_id")}
    for _ in range(samples):
        draw = rng.choice(object_ids, size=len(object_ids), replace=True)
        values = np.concatenate(
            [grouped[object_id][value_column].to_numpy(dtype=float) for object_id in draw]
        )
        means.append(float(values.mean()))
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def audit_metrics(rows: pd.DataFrame, bootstrap_samples: int = 2000, seed: int = 0) -> Dict[str, Any]:
    if rows.empty:
        raise ValueError("Caption audit has no scored rows")
    low, high = _cluster_bootstrap_interval(rows, bootstrap_samples, seed)
    abstention_low, abstention_high = _cluster_bootstrap_interval(
        rows, bootstrap_samples, seed, value_column="abstained"
    )
    result: Dict[str, Any] = {
        "objects": int(rows["object_id"].nunique()),
        "answers": int(len(rows)),
        "accuracy": float(rows["correct"].mean()),
        "accuracy_ci95": [low, high],
        "abstention_rate": float(rows["abstained"].mean()),
        "abstention_ci95": [abstention_low, abstention_high],
        "by_question": {},
    }
    for question, group in rows.groupby("question", sort=True):
        q_low, q_high = _cluster_bootstrap_interval(group, bootstrap_samples, seed)
        q_abstention_low, q_abstention_high = _cluster_bootstrap_interval(
            group, bootstrap_samples, seed, value_column="abstained"
        )
        result["by_question"][question] = {
            "answers": int(len(group)),
            "accuracy": float(group["correct"].mean()),
            "accuracy_ci95": [q_low, q_high],
            "abstention_rate": float(group["abstained"].mean()),
            "abstention_ci95": [q_abstention_low, q_abstention_high],
        }
    return result


def path_audit_metrics(
    rows: pd.DataFrame, bootstrap_samples: int = 2000, seed: int = 0
) -> Dict[str, Any]:
    if rows.empty:
        raise ValueError("Path audit has no scored rows")
    low, high = _cluster_bootstrap_interval(
        rows, bootstrap_samples, seed, value_column="score"
    )
    return {
        "metric": "released_decision_path_overlap",
        "objects": int(rows["object_id"].nunique()),
        "mean_score": float(rows["score"].mean()),
        "mean_score_ci95": [low, high],
        "mean_human_path_length": float(rows["human_nodes"].mean()),
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
    }


def paired_accuracy_delta(
    qwen_rows: pd.DataFrame,
    gpt_rows: pd.DataFrame,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Paired object-cluster bootstrap for GPT-minus-Qwen answer accuracy."""
    keys = ["object_id", "question"]
    required = set(keys) | {"gold", "correct"}
    for name, rows in (("qwen", qwen_rows), ("gpt", gpt_rows)):
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"{name} audit rows missing columns: {sorted(missing)}")
        if rows.duplicated(keys).any():
            raise ValueError(f"{name} audit rows contain duplicate object-question pairs")
    paired = qwen_rows.loc[:, keys + ["gold", "correct"]].merge(
        gpt_rows.loc[:, keys + ["gold", "correct"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_qwen", "_gpt"),
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Qwen and GPT audits do not contain identical object-question pairs")
    if not paired["gold_qwen"].eq(paired["gold_gpt"]).all():
        raise ValueError("Qwen and GPT audit rows disagree on human labels")
    paired["delta"] = (
        paired["correct_gpt"].astype(float) - paired["correct_qwen"].astype(float)
    )
    object_ids = paired["object_id"].drop_duplicates().to_numpy()
    grouped = {
        object_id: group["delta"].to_numpy(dtype=float)
        for object_id, group in paired.groupby("object_id")
    }
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap_samples):
        sampled = rng.choice(object_ids, size=len(object_ids), replace=True)
        draws.append(float(np.concatenate([grouped[value] for value in sampled]).mean()))
    return {
        "point": float(paired["delta"].mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "objects": int(len(object_ids)),
        "answers": int(len(paired)),
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
    }


def paired_path_score_delta(
    qwen_rows: pd.DataFrame,
    gpt_rows: pd.DataFrame,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Paired object bootstrap for GPT-minus-Qwen released path score."""
    required = {"object_id", "human_path_json", "score"}
    for name, rows in (("qwen", qwen_rows), ("gpt", gpt_rows)):
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"{name} path rows missing columns: {sorted(missing)}")
        if rows["object_id"].duplicated().any():
            raise ValueError(f"{name} path rows contain duplicate objects")
    paired = qwen_rows.loc[:, ["object_id", "human_path_json", "score"]].merge(
        gpt_rows.loc[:, ["object_id", "human_path_json", "score"]],
        on="object_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_qwen", "_gpt"),
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Qwen and GPT path audits do not contain identical objects")
    if not paired["human_path_json_qwen"].eq(paired["human_path_json_gpt"]).all():
        raise ValueError("Qwen and GPT path audits disagree on human paths")
    deltas = paired["score_gpt"].astype(float) - paired["score_qwen"].astype(float)
    rng = np.random.default_rng(seed)
    draws = [
        float(rng.choice(deltas.to_numpy(), size=len(deltas), replace=True).mean())
        for _ in range(bootstrap_samples)
    ]
    return {
        "metric": "released_decision_path_overlap",
        "point": float(deltas.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "objects": int(len(paired)),
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
    }


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


def write_path_audit(
    captions: pd.DataFrame,
    labels: pd.DataFrame,
    output_dir: Path,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = path_audit_rows(captions, labels)
    rows.to_csv(output_dir / "judge_path_audit_rows.csv", index=False)
    (output_dir / "judge_path_audit_metrics.json").write_text(
        json.dumps(
            path_audit_metrics(rows, bootstrap_samples, seed),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
