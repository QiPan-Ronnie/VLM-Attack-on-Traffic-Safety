#!/usr/bin/env python3
"""Build LOTVS-DADA train/val/test splits from repo JSON files.

Outputs symlinked or copied split folders plus clips.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List


JSON_BY_SPLIT = {
    "train": "train_file.json",
    "val": "val_file.json",
    "test": "test_file.json",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dada-root", required=True, help="Root like DADA_dataset/<category>/<clip_id>/images")
    p.add_argument("--repo-root", required=True, help="Unzipped LOTVS-DADA-master directory")
    p.add_argument("--out-root", required=True, help="Output split root")
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--fallback-copy", action="store_true", help="If symlink fails, fall back to copy")
    return p.parse_args()


def safe_link_or_copy(src: Path, dst: Path, mode: str, fallback_copy: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return "exists"
    if mode == "copy":
        shutil.copytree(src, dst)
        return "copy"
    try:
        os.symlink(src, dst, target_is_directory=True)
        return "symlink"
    except Exception:
        if fallback_copy:
            shutil.copytree(src, dst)
            return "copy_fallback"
        raise


def main() -> None:
    args = parse_args()
    dada_root = Path(args.dada_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    alias_seen = {s: set() for s in JSON_BY_SPLIT}

    for split, json_name in JSON_BY_SPLIT.items():
        json_path = repo_root / json_name
        if not json_path.exists():
            raise FileNotFoundError(f"Missing {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            # repo format: [[category, clip_id], alias]
            pair, alias = item
            category, clip_id = str(pair[0]), str(pair[1]).zfill(3)
            alias = str(alias).zfill(3)
            src_clip_dir = dada_root / category / clip_id
            if not src_clip_dir.exists():
                print(f"[WARN] Missing source clip: {src_clip_dir}")
                continue
            if alias in alias_seen[split]:
                raise RuntimeError(f"Duplicate alias in {split}: {alias}")
            alias_seen[split].add(alias)
            dst_clip_dir = split_dir / alias
            method = safe_link_or_copy(src_clip_dir, dst_clip_dir, args.mode, args.fallback_copy)
            rows.append({
                "split": split,
                "alias": alias,
                "category": category,
                "clip_id": clip_id,
                "src_clip_dir": str(src_clip_dir),
                "dst_clip_dir": str(dst_clip_dir),
                "method": method,
            })

    clips_csv = out_root / "clips.csv"
    with open(clips_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "alias", "category", "clip_id", "src_clip_dir", "dst_clip_dir", "method"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved splits to: {out_root}")
    print(f"Saved clips csv: {clips_csv}")


if __name__ == "__main__":
    main()
