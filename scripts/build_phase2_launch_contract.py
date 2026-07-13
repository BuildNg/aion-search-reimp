"""Build the reviewable Phase 2 prerequisite contract from row-level evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aion_reimp.launch_contract import build_launch_contract, write_launch_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_smoke.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/gates/phase2_launch_contract_v1.json")
    )
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prerequisite = config["prerequisites"]
    contract = build_launch_contract(
        Path(prerequisite["phase0_reference_gate"]),
        Path(prerequisite["phase1_qwen_caption_audit_rows"]),
        Path(prerequisite["phase1_gpt_caption_audit_rows"]),
        bootstrap_samples=prerequisite["bootstrap_samples"],
        seed=config["run"]["seed"],
    )
    write_launch_contract(contract, args.output)
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
