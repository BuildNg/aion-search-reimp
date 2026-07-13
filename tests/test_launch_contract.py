import json

import pandas as pd
import pytest

from aion_reimp.launch_contract import build_launch_contract, require_launch_allowed


def test_launch_contract_carries_phase1_diagnostics_without_blocking_smoke(tmp_path) -> None:
    reference = {
        name: {"passed": True, "observed": value, "observed_rounded": round(value, 3)}
        for name, value in {"spiral": 0.9413, "merger": 0.5543, "lens": 0.1731}.items()
    }
    reference["lens_top10_confirmed"] = {"passed": True, "observed": 2}
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    rows = pd.DataFrame(
        {
            "object_id": ["a", "b"],
            "question": ["smooth-or-featured", "smooth-or-featured"],
            "correct": [True, False],
            "abstained": [False, True],
        }
    )
    rows_path = tmp_path / "rows.csv"
    rows.to_csv(rows_path, index=False)
    gpt_rows_path = tmp_path / "gpt_rows.csv"
    rows.assign(correct=[True, True], abstained=[False, False]).to_csv(
        gpt_rows_path, index=False
    )
    contract = build_launch_contract(
        reference_path,
        rows_path,
        gpt_rows_path,
        bootstrap_samples=20,
        seed=1,
    )
    assert contract["phase0"]["passed"] is True
    assert contract["phase1"]["qwen3vl_8b"]["accuracy"] == 0.5
    assert contract["phase1"]["qwen3vl_8b"]["abstention_rate"] == 0.5
    assert contract["phase1"]["gpt41mini"]["accuracy"] == 1.0
    assert contract["phase1"]["accuracy_delta_gpt_minus_qwen"] == 0.5
    assert contract["phase1"]["role"] == "diagnostic_only"
    assert contract["launch_allowed"] is True
    require_launch_allowed(contract)


def test_launch_contract_still_blocks_failed_phase0(tmp_path) -> None:
    reference = {
        name: {"passed": name != "lens", "observed": value}
        for name, value in {"spiral": 0.941, "merger": 0.554, "lens": 0.1}.items()
    }
    reference["lens_top10_confirmed"] = {"passed": True, "observed": 2}
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    rows_path = tmp_path / "rows.csv"
    pd.DataFrame(
        {
            "object_id": ["a"],
            "question": ["smooth-or-featured"],
            "correct": [True],
            "abstained": [False],
        }
    ).to_csv(rows_path, index=False)
    contract = build_launch_contract(
        reference_path, rows_path, rows_path, bootstrap_samples=20, seed=1
    )
    assert contract["launch_allowed"] is False
    with pytest.raises(RuntimeError, match="Phase 0"):
        require_launch_allowed(contract)
