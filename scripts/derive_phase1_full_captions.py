"""Restore immutable full-text Phase 1 caption artifacts from preserved responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aion_reimp.utils import read_jsonl, sha256_file


def _restore(source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite caption artifact: {output}")
    rows = read_jsonl(source)
    if len(rows) != 64 or len({str(row["object_id"]) for row in rows}) != 64:
        raise ValueError(f"Expected 64 unique caption rows in {source}")
    restored = []
    for row in rows:
        raw = row.get("raw_response")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"Missing raw_response for object {row.get('object_id')}")
        description = raw.strip()
        restored.append(
            {
                "object_id": str(row["object_id"]),
                "description": description,
                "raw_response": raw,
                "word_count": len(description.split()),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in sorted(restored, key=lambda item: item["object_id"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "rows": len(restored),
        "transform": "restore_description_from_preserved_raw_response_without_truncation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-source", type=Path, required=True)
    parser.add_argument("--gpt-source", type=Path, required=True)
    parser.add_argument("--qwen-output", type=Path, required=True)
    parser.add_argument("--gpt-output", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    args = parser.parse_args()

    if args.lineage.exists():
        raise FileExistsError(f"Refusing to overwrite lineage artifact: {args.lineage}")
    lineage = {
        "qwen": _restore(args.qwen_source, args.qwen_output),
        "gpt": _restore(args.gpt_source, args.gpt_output),
    }
    qwen_ids = {row["object_id"] for row in read_jsonl(args.qwen_output)}
    gpt_ids = {row["object_id"] for row in read_jsonl(args.gpt_output)}
    if qwen_ids != gpt_ids:
        raise ValueError("Restored Qwen and GPT artifacts do not share the same object set")
    args.lineage.parent.mkdir(parents=True, exist_ok=True)
    args.lineage.write_text(
        json.dumps(lineage, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
