"""Small safe writers for reproducible run artifacts."""

from __future__ import annotations

import json
import platform
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import yaml


def create_run_dir(output_root: Path, run_id: str, overwrite: bool = False) -> Path:
    path = Path(output_root) / run_id
    if path.exists() and not overwrite:
        raise FileExistsError(f"Run directory exists: {path}")
    path.mkdir(parents=True, exist_ok=overwrite)
    return path


def write_json(path: Path, value: Mapping[str, Any], overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def initialize_run(
    output_root: Path,
    run_id: str,
    resolved_config: Mapping[str, Any],
    command: str,
) -> Path:
    run_dir = create_run_dir(output_root, run_id)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(dict(resolved_config), sort_keys=False), encoding="utf-8"
    )
    (run_dir / "command.txt").write_text(command.strip() + "\n", encoding="utf-8")
    return run_dir


@contextmanager
def tracked_run(run_dir: Path, metadata: Optional[Mapping[str, Any]] = None) -> Iterator[None]:
    started = datetime.now(timezone.utc).isoformat()
    base = {
        "status": "running",
        "started_at": started,
        "host": platform.node(),
        **dict(metadata or {}),
    }
    status_path = Path(run_dir) / "run_status.json"
    write_json(status_path, base)
    try:
        yield
    except Exception as error:
        write_json(
            status_path,
            {
                **base,
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
            overwrite=True,
        )
        raise
    else:
        write_json(
            status_path,
            {
                **base,
                "status": "complete",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
            overwrite=True,
        )
