"""Review-gated Phase 1 caption-screen and Qwen-embedding entrypoint."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pandas as pd
from aion_reimp.cache import write_embedding_cache
from aion_reimp.artifacts import initialize_run, tracked_run
from aion_reimp.caption_audit import write_audit
from aion_reimp.captioning import QwenCaptioner, append_caption_results
from aion_reimp.config import load_config
from aion_reimp.datasets import (
    load_pinned_dataset,
    load_query_rows,
    materialize_caption_screen,
)
from aion_reimp.text_embeddings import EmbeddingSpec, QwenEmbedder, embedding_frame


def main() -> None:
    config = load_config(Path("configs/phase1_open_text.yaml"))
    output_root = initialize_run(
        Path(config["run"]["output_root"]),
        config["run"]["id"],
        config,
        shlex.join(sys.argv),
    )
    audit_spec = config["caption_audit"]
    screen_dir = Path(audit_spec["output_dir"])
    labels_path = screen_dir / "caption_screen_labels.parquet"
    if not labels_path.exists():
        materialize_caption_screen(
            audit_spec["repo_id"], audit_spec["revision"], audit_spec["split"], screen_dir
        )
    labels = pd.read_parquet(labels_path)

    with tracked_run(output_root, {"phase": 1, "condition": "open_text_pipeline"}):
        caption_spec = config["captioning"]
        captioner = QwenCaptioner(
            caption_spec["model_id"],
            caption_spec["revision"],
            Path(caption_spec["prompt_file"]).read_text(encoding="utf-8"),
            dtype=caption_spec["dtype"],
            max_new_tokens=caption_spec["max_new_tokens"],
        )
        captions_path = output_root / "caption_screen_outputs.jsonl"
        append_caption_results(captioner, labels.to_dict(orient="records"), captions_path)
        captions = pd.read_json(captions_path, lines=True)
        write_audit(
            captions,
            labels,
            output_root / "caption_audit",
            bootstrap_samples=audit_spec["bootstrap_samples"],
            seed=config["run"]["seed"],
        )

        embedding_data = config["text_embedding"]
        embedding_spec = EmbeddingSpec.from_mapping(embedding_data)
        embedder = QwenEmbedder(embedding_spec)
        document_vectors = embedder.encode(captions["summary"].astype(str).tolist(), "document")
        document_frame = embedding_frame(
            captions["object_id"].astype(str).tolist(),
            captions["summary"].astype(str).tolist(),
            document_vectors,
            "document",
            embedding_spec,
        )
        write_embedding_cache(document_frame, output_root / "caption_screen_qwen_embeddings.parquet")

        query_rows = load_query_rows(Path(config["queries"]["file"]))
        query_texts = [row["text"] for row in query_rows]
        query_vectors = embedder.encode(query_texts, "query")
        query_frame = embedding_frame(
            [row["object_id"] for row in query_rows],
            query_texts,
            query_vectors,
            "query",
            embedding_spec,
        )
        for column in ("category", "variant"):
            query_frame[column] = [row[column] for row in query_rows]
        write_embedding_cache(query_frame, output_root / "qwen_query_embeddings.parquet")

        released_spec = config["released_text"]
        released = load_pinned_dataset(
            released_spec["repo_id"],
            released_spec["revision"],
            released_spec["split"],
        )
        required = {released_spec["object_id_column"], released_spec["text_column"]}
        if required - set(released.column_names):
            raise ValueError(
                f"Released text dataset missing columns: {sorted(required - set(released.column_names))}"
            )
        frames = []
        batch_size = released_spec["batch_size"]
        for start in range(0, len(released), batch_size):
            stop = min(start + batch_size, len(released))
            batch = released.select(range(start, stop))
            object_ids = [str(value) for value in batch[released_spec["object_id_column"]]]
            texts = [str(value) for value in batch[released_spec["text_column"]]]
            vectors = embedder.encode(texts, "document")
            frames.append(embedding_frame(object_ids, texts, vectors, "document", embedding_spec))
        write_embedding_cache(
            pd.concat(frames, ignore_index=True),
            output_root / "released_summary_qwen_embeddings.parquet",
        )



if __name__ == "__main__":
    main()
