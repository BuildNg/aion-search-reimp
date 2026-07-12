"""Thin command-line entrypoints for Phase 0/1 preparation and cluster work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .cache import ingest_released_embeddings, write_embedding_cache
from .caption_audit import write_audit
from .captioning import QwenCaptioner, append_caption_results
from .config import load_config
from .datasets import (
    load_query_rows,
    materialize_benchmark_coordinates,
    materialize_caption_screen,
)
from .evaluate import evaluate_released_benchmarks
from .manifest import build_manifest, coordinate_exclusion_table, write_manifest
from .reference import (
    assert_reference_equivalence,
    freeze_openai_queries,
    load_author_reference,
    load_reimplemented_reference,
    resolve_released_files,
)
from .text_embeddings import EmbeddingSpec, QwenEmbedder, embedding_frame


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _cmd_validate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print(json.dumps(config, indent=2, sort_keys=True))


def _cmd_manifest(args: argparse.Namespace) -> None:
    source = _read_table(args.input)
    exclusions: Dict[str, List[str]] = {}
    for spec in args.exclusion:
        name, path_text = spec.split("=", 1)
        frame = _read_table(Path(path_text))
        exclusions[name] = frame["object_id"].astype(str).tolist()
    frame = build_manifest(source, exclusions, seed=args.seed, train_ratio=args.train_ratio)
    fingerprint = write_manifest(frame, args.output, overwrite=args.overwrite)
    print(f"manifest_rows={len(frame)} fingerprint={fingerprint}")


def _cmd_coordinate_exclusions(args: argparse.Namespace) -> None:
    source = _read_table(args.source)
    benchmarks = {}
    for spec in args.benchmark:
        name, path_text = spec.split("=", 1)
        benchmarks[name] = _read_table(Path(path_text))
    matches = coordinate_exclusion_table(source, benchmarks, radius_arcsec=args.radius_arcsec)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Coordinate exclusion artifact exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output, index=False)
    print(f"coordinate_exclusions={len(matches)}")


def _cmd_ingest_released(args: argparse.Namespace) -> None:
    source = _read_table(args.input)
    frame = ingest_released_embeddings(source)
    fingerprint = write_embedding_cache(frame, args.output, overwrite=args.overwrite)
    print(f"cache_rows={len(frame)} fingerprint={fingerprint}")


def _cmd_freeze_queries(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    query_rows = load_query_rows(Path(config["queries"]["file"]))
    freeze_openai_queries(
        query_rows,
        Path(config["queries"]["openai_cache"]),
        model=config["queries"]["openai_model"],
    )


def _cmd_reference_equivalence(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    spec = config["reference_model"]
    config_path, weights_path = resolve_released_files(
        spec["repo_id"], spec["revision"], spec["config_file"], spec["weights_file"]
    )
    reimplemented = load_reimplemented_reference(config_path, weights_path)
    author = load_author_reference(Path(spec["orig_repo"]), config_path, weights_path)
    maxima = assert_reference_equivalence(reimplemented, author, seed=config["run"]["seed"])
    print(json.dumps(maxima, indent=2, sort_keys=True))


def _cmd_reference_evaluation(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    spec = config["reference_model"]
    config_path, weights_path = resolve_released_files(
        spec["repo_id"], spec["revision"], spec["config_file"], spec["weights_file"]
    )
    model = load_reimplemented_reference(config_path, weights_path)
    query_cache = pd.read_parquet(Path(config["queries"]["openai_cache"]))
    output_dir = Path(config["run"]["output_root"]) / config["run"]["id"] / "released_evaluation"
    evaluate_released_benchmarks(
        model,
        query_cache,
        config["benchmarks"],
        output_dir,
        reference_gate=config["reference_gate"],
    )


def _cmd_benchmark_coordinates(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    outputs = materialize_benchmark_coordinates(config["benchmarks"], args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


def _cmd_materialize_screen(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    spec = config["caption_audit"]
    frame = materialize_caption_screen(
        spec["repo_id"], spec["revision"], spec["split"], Path(spec["output_dir"])
    )
    print(f"caption_screen_rows={len(frame)}")


def _cmd_caption_screen(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    spec = config["captioning"]
    prompt = Path(spec["prompt_file"]).read_text(encoding="utf-8")
    captioner = QwenCaptioner(
        spec["model_id"],
        spec["revision"],
        prompt,
        dtype=spec["dtype"],
        max_new_tokens=spec["max_new_tokens"],
    )
    labels = pd.read_parquet(Path(config["caption_audit"]["output_dir"]) / "caption_screen_labels.parquet")
    append_caption_results(captioner, labels.to_dict(orient="records"), args.output)


def _cmd_audit(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    audit = config["caption_audit"]
    captions = pd.read_json(args.captions, lines=True)
    labels = pd.read_parquet(Path(audit["output_dir"]) / "caption_screen_labels.parquet")
    write_audit(
        captions,
        labels,
        Path(config["run"]["output_root"]) / config["run"]["id"] / "caption_audit",
        bootstrap_samples=audit["bootstrap_samples"],
        seed=config["run"]["seed"],
    )


def _cmd_embed(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    spec_data = config["text_embedding"]
    spec = EmbeddingSpec.from_mapping(spec_data)
    source = _read_table(args.input)
    vectors = QwenEmbedder(spec).encode(source[args.text_column].astype(str).tolist(), args.role)
    frame = embedding_frame(
        source["object_id"].astype(str).tolist(),
        source[args.text_column].astype(str).tolist(),
        vectors,
        args.role,
        spec,
    )
    write_embedding_cache(frame, args.output, overwrite=args.overwrite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aion-reimp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config", type=Path)
    validate.set_defaults(func=_cmd_validate)

    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("--input", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--exclusion", action="append", default=[], help="NAME=CSV_OR_PARQUET")
    manifest.add_argument("--seed", type=int, default=42)
    manifest.add_argument("--train-ratio", type=float, default=0.8)
    manifest.add_argument("--overwrite", action="store_true")
    manifest.set_defaults(func=_cmd_manifest)

    coordinate = subparsers.add_parser("build-coordinate-exclusions")
    coordinate.add_argument("--source", type=Path, required=True)
    coordinate.add_argument("--benchmark", action="append", required=True, help="NAME=CSV_OR_PARQUET")
    coordinate.add_argument("--radius-arcsec", type=float, default=1.0)
    coordinate.add_argument("--output", type=Path, required=True)
    coordinate.add_argument("--overwrite", action="store_true")
    coordinate.set_defaults(func=_cmd_coordinate_exclusions)

    ingest = subparsers.add_parser("ingest-released")
    ingest.add_argument("--input", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--overwrite", action="store_true")
    ingest.set_defaults(func=_cmd_ingest_released)

    freeze = subparsers.add_parser("freeze-openai-queries")
    freeze.add_argument("config", type=Path)
    freeze.set_defaults(func=_cmd_freeze_queries)

    equivalence = subparsers.add_parser("reference-equivalence")
    equivalence.add_argument("config", type=Path)
    equivalence.set_defaults(func=_cmd_reference_equivalence)

    reference_eval = subparsers.add_parser("reference-evaluation")
    reference_eval.add_argument("config", type=Path)
    reference_eval.set_defaults(func=_cmd_reference_evaluation)

    benchmark_coordinates = subparsers.add_parser("materialize-benchmark-coordinates")
    benchmark_coordinates.add_argument("config", type=Path)
    benchmark_coordinates.add_argument("--output-dir", type=Path, required=True)
    benchmark_coordinates.set_defaults(func=_cmd_benchmark_coordinates)

    materialize = subparsers.add_parser("materialize-caption-screen")
    materialize.add_argument("config", type=Path)
    materialize.set_defaults(func=_cmd_materialize_screen)

    caption = subparsers.add_parser("caption-screen")
    caption.add_argument("config", type=Path)
    caption.add_argument("--output", type=Path, required=True)
    caption.set_defaults(func=_cmd_caption_screen)

    audit = subparsers.add_parser("audit-captions")
    audit.add_argument("config", type=Path)
    audit.add_argument("--captions", type=Path, required=True)
    audit.set_defaults(func=_cmd_audit)

    embed = subparsers.add_parser("embed-text")
    embed.add_argument("config", type=Path)
    embed.add_argument("--input", type=Path, required=True)
    embed.add_argument("--text-column", required=True)
    embed.add_argument("--role", choices=["document", "query"], required=True)
    embed.add_argument("--output", type=Path, required=True)
    embed.add_argument("--overwrite", action="store_true")
    embed.set_defaults(func=_cmd_embed)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
