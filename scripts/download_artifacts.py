"""Download the exact Phase 0/1 artifacts sequentially with resumable HTTP."""

from __future__ import annotations

from datetime import datetime, timezone

from huggingface_hub import snapshot_download


ARTIFACTS = (
    ("model", "astronolan/aion-search", "e6d56ee28b6768f4e3e4494b2c0b32a00abb2594"),
    (
        "dataset",
        "astronolan/galaxy-description-benchmark",
        "ebb13986d04b6b5e47529fb1fc68761839bffd75",
    ),
    (
        "dataset",
        "astronolan/gz-decals-embeddings",
        "c11f7a02aa1ed00b85f3dd43c222271046445a2e",
    ),
    (
        "dataset",
        "astronolan/lens-retrieval-ls-embeddings",
        "f5507c433552084e2b3d195a27dae5110037d64d",
    ),
    (
        "model",
        "Qwen/Qwen3-Embedding-0.6B",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    ),
    (
        "model",
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    ),
    (
        "dataset",
        "astronolan/galaxy-descriptions",
        "6890dc0c8fc793867c07a6ce4fce11e51c167d6e",
    ),
)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    for repo_type, repo_id, revision in ARTIFACTS:
        print(f"[{timestamp()}] START {repo_type} {repo_id}@{revision}", flush=True)
        path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            max_workers=1,
        )
        print(f"[{timestamp()}] DONE {repo_id} -> {path}", flush=True)


if __name__ == "__main__":
    main()
