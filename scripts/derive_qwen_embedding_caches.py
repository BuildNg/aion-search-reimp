"""Derive neutral fp32-normalized Qwen retrieval caches without mutating sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aion_reimp.cache import (
    NormalizationPolicy,
    cache_fingerprint,
    derive_fp32_normalized_cache,
    validate_embedding_cache,
)


CACHE_NAMES = (
    "qwen_query_embeddings.parquet",
    "released_summary_qwen_embeddings.parquet",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cache/qwen_embeddings/fp32_normalized_v1"),
    )
    args = parser.parse_args()
    policy = NormalizationPolicy(required=True, atol=1e-3)
    for name in CACHE_NAMES:
        source = args.source_dir / name
        if not source.exists():
            raise FileNotFoundError(f"Missing Qwen cache: {source}")
        output = args.output_dir / name
        if output.exists():
            derived = pd.read_parquet(output)
            validate_embedding_cache(derived, policy)
            meta = json.loads(
                output.with_suffix(output.suffix + ".meta.json").read_text(encoding="utf-8")
            )
            if meta["fingerprint"] != cache_fingerprint(derived):
                raise ValueError(f"Existing derived cache fingerprint mismatch: {output}")
            print(f"existing={output} fingerprint={meta['fingerprint']}")
            continue
        fingerprint = derive_fp32_normalized_cache(source, output, policy)
        print(f"derived={output} fingerprint={fingerprint}")


if __name__ == "__main__":
    main()
