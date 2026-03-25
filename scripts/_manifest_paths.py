#!/usr/bin/env python3
"""Helpers for resolving dataset-relative paths stored in CSV manifests."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value)


def infer_manifest_root(csv_path: Path) -> Path:
    """Infer the dataset/package root that relative CSV paths should resolve against."""
    csv_path = csv_path.resolve()
    candidates = [csv_path.parent, csv_path.parent.parent, csv_path.parent.parent.parent]
    for candidate in candidates:
        if candidate.exists() and (candidate / "data").exists():
            return candidate
    return csv_path.parent


def resolve_manifest_path(value, manifest_root: Optional[Path]) -> Optional[Path]:
    """Resolve a path field from a manifest into an absolute local path."""
    if _is_missing(value):
        return None
    raw = Path(str(value))
    if raw.is_absolute():
        return raw
    if manifest_root is None:
        return raw
    return (manifest_root / raw).resolve()


def derive_map_path_from_image(image_path: Path, frame_idx: int) -> Optional[Path]:
    """Infer the matching attention-map frame path from an image path."""
    suffix = image_path.suffix or ".jpg"
    if image_path.parent.name == "images":
        return image_path.parent.parent / "maps" / f"{frame_idx:04d}{suffix}"

    s = str(image_path)
    if "/images/" in s:
        return Path(s.replace("/images/", "/maps/"))
    if "\\images\\" in s:
        return Path(s.replace("\\images\\", "\\maps\\"))
    return None
