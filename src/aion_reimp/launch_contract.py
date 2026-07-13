"""Evidence-bearing prerequisites for the Phase 2 smoke launch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

from .caption_audit import audit_metrics


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_launch_contract(
    reference_gate_path: Path,
    qwen_caption_audit_rows_path: Path,
    gpt_caption_audit_rows_path: Path,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    reference_gate_path = Path(reference_gate_path)
    qwen_caption_audit_rows_path = Path(qwen_caption_audit_rows_path)
    gpt_caption_audit_rows_path = Path(gpt_caption_audit_rows_path)
    reference = json.loads(reference_gate_path.read_text(encoding="utf-8"))
    required_reference = {"spiral", "merger", "lens", "lens_top10_confirmed"}
    if set(reference) != required_reference:
        raise ValueError("Phase 0 reference gate has unexpected fields")
    reference_passed = all(bool(reference[key]["passed"]) for key in required_reference)

    qwen_metrics = audit_metrics(
        pd.read_csv(qwen_caption_audit_rows_path),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    gpt_metrics = audit_metrics(
        pd.read_csv(gpt_caption_audit_rows_path),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    def diagnostic_record(metrics: Mapping[str, Any], source_path: Path) -> Dict[str, Any]:
        return {
            "accuracy": metrics["accuracy"],
            "accuracy_ci95": metrics["accuracy_ci95"],
            "abstention_rate": metrics["abstention_rate"],
            "abstention_ci95": metrics["abstention_ci95"],
            "objects": metrics["objects"],
            "answers": metrics["answers"],
            "source_path": str(source_path),
            "source_sha256": _file_sha256(source_path),
        }

    return {
        "phase0": {
            "passed": reference_passed,
            "spiral_ndcg_at_10": reference["spiral"],
            "merger_ndcg_at_10": reference["merger"],
            "lens_ndcg_at_10": reference["lens"],
            "lens_top10_confirmed": reference["lens_top10_confirmed"],
            "source_path": str(reference_gate_path),
            "source_sha256": _file_sha256(reference_gate_path),
        },
        "phase1": {
            "role": "diagnostic_only",
            "gpt41mini": diagnostic_record(gpt_metrics, gpt_caption_audit_rows_path),
            "qwen3vl_8b": diagnostic_record(qwen_metrics, qwen_caption_audit_rows_path),
            "accuracy_delta_gpt_minus_qwen": float(gpt_metrics["accuracy"])
            - float(qwen_metrics["accuracy"]),
        },
        "launch_allowed": reference_passed,
        "run_class": "engineering_smoke",
    }


def write_launch_contract(contract: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Launch contract exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(contract), indent=2, sort_keys=True), encoding="utf-8")


def require_launch_allowed(contract: Mapping[str, Any]) -> None:
    if not bool(contract.get("launch_allowed")):
        raise RuntimeError(
            "Phase 2 launch blocked because the Phase 0 reference gate did not pass: "
            f"phase0_passed={contract['phase0']['passed']}"
        )
