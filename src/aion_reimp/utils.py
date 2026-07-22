"""Small serialization, filesystem, and hash helpers shared by scripts and modules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set


def read_jsonl(path: Path, missing_ok: bool = False) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping blank lines.

    A missing file raises ``FileNotFoundError`` unless ``missing_ok`` is set,
    in which case it returns an empty list (for resumable caption/artifact
    jobs that have not written a first row yet).
    """
    path = Path(path)
    if missing_ok and not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Optional[Path], record: Mapping[str, Any]) -> None:
    """Append one JSON record as a line, creating parent directories as needed.

    A ``None`` path is a no-op, matching optional error-log sinks that callers
    may choose not to wire up.
    """
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
        handle.flush()


def read_object_ids(path: Path) -> Set[str]:
    """Read the set of ``object_id`` values from a JSONL file, or empty if absent."""
    path = Path(path)
    if not path.exists():
        return set()
    return {
        str(json.loads(line)["object_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Hex digest of a file's bytes, streamed in 1 MiB blocks."""
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    return file_digest(path, "sha256")
