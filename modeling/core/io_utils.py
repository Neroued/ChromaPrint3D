"""
Shared I/O and label utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_json(path: "str | Path") -> Dict[str, object]:
    """Load a JSON file and return its content as a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_label(label: str) -> str:
    """Normalize a label to lowercase alphanumeric characters only."""
    return "".join(ch for ch in label.lower().strip() if ch.isalnum())


def parse_layer_order(value: object, default: str = "Top2Bottom") -> str:
    """Parse a layer_order value from JSON (string or int) to a canonical string."""
    if isinstance(value, str):
        if value in ("Top2Bottom", "Bottom2Top"):
            return value
    if isinstance(value, int):
        if value == 0:
            return "Top2Bottom"
        if value == 1:
            return "Bottom2Top"
    return default


def resolve_db_paths(db_path: Path) -> List[Path]:
    """Resolve a path to a list of colorDB JSON files (file or directory)."""
    if db_path.is_dir():
        candidates = sorted(
            [
                p
                for p in db_path.iterdir()
                if p.is_file() and p.suffix.lower() == ".json"
            ]
        )
        if not candidates:
            raise ValueError(f"No json files found in directory: {db_path}")
        return candidates
    if db_path.is_file():
        return [db_path]
    raise FileNotFoundError(f"DB path not found: {db_path}")
