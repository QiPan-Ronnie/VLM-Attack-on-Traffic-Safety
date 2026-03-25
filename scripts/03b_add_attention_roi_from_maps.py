#!/usr/bin/env python3
"""Add weak critical-region ROI boxes from DADA attention maps.

This is optional but recommended for safety-attack evaluation.
Given a frame manifest, the script locates the corresponding `maps/000X.jpg`
frame for each `images/000X.jpg` and extracts a salient ROI box from the
grayscale attention map.

Output columns added:
    map_path, roi_x1, roi_y1, roi_x2, roi_y2, roi_source, roi_area_ratio
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from _manifest_paths import derive_map_path_from_image, infer_manifest_root, resolve_manifest_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--data-root", default=None, help="Optional root used to resolve relative image/map paths")
    p.add_argument("--expand-ratio", type=float, default=0.18, help="Expand ROI bbox by this fraction")
    p.add_argument("--percentile", type=float, default=92.0, help="Threshold percentile on attention map")
    p.add_argument("--min-area-ratio", type=float, default=0.002, help="Fallback if thresholded region too small")
    return p.parse_args()


def resolve_map_path(row: dict, manifest_root: Path, frame_idx: int) -> Optional[Path]:
    explicit = resolve_manifest_path(row.get("map_path"), manifest_root)
    if explicit is not None:
        return explicit
    image_path = resolve_manifest_path(row.get("image_path"), manifest_root)
    if image_path is None:
        return None
    return derive_map_path_from_image(image_path, frame_idx)


def fallback_center_box(w: int, h: int) -> Tuple[int, int, int, int]:
    bw = int(w * 0.28)
    bh = int(h * 0.22)
    x1 = (w - bw) // 2
    y1 = int(h * 0.45) - bh // 2
    x1 = max(0, min(x1, w - bw))
    y1 = max(0, min(y1, h - bh))
    return x1, y1, x1 + bw, y1 + bh


def expand_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int, ratio: float) -> Tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    ex = int(round(bw * ratio))
    ey = int(round(bh * ratio))
    return (
        max(0, x1 - ex),
        max(0, y1 - ey),
        min(w, x2 + ex),
        min(h, y2 + ey),
    )


def extract_roi_from_map(map_path: Path, percentile: float, expand_ratio: float, min_area_ratio: float) -> Tuple[Tuple[int, int, int, int], float]:
    img = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read map image: {map_path}")
    h, w = img.shape[:2]
    blur = cv2.GaussianBlur(img, (9, 9), 0)
    thr = np.percentile(blur, percentile)
    mask = (blur >= thr).astype(np.uint8) * 255

    # if threshold is too sparse, relax it
    area_ratio = float(mask.sum() / 255.0) / float(w * h)
    if area_ratio < min_area_ratio:
        thr = np.percentile(blur, max(70.0, percentile - 15.0))
        mask = (blur >= thr).astype(np.uint8) * 255
        area_ratio = float(mask.sum() / 255.0) / float(w * h)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    best_score = -1.0
    for lab in range(1, num_labels):
        x, y, bw, bh, area = stats[lab]
        if area <= 0:
            continue
        region = blur[labels == lab]
        score = float(area) * float(region.mean() if region.size else 0.0)
        if score > best_score:
            best_score = score
            best = (int(x), int(y), int(x + bw), int(y + bh))
    if best is None:
        best = fallback_center_box(w, h)

    x1, y1, x2, y2 = expand_box(*best, w, h, ratio=expand_ratio)
    roi_area_ratio = float((x2 - x1) * (y2 - y1)) / float(w * h)
    return (x1, y1, x2, y2), roi_area_ratio


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    manifest_root = Path(args.data_root).resolve() if args.data_root else infer_manifest_root(input_csv)
    df = pd.read_csv(input_csv)
    rows = []
    for row in tqdm(df.to_dict("records"), total=len(df), desc="Add attention ROI"):
        frame_idx = int(row["frame_idx"])
        mp = resolve_map_path(row, manifest_root, frame_idx)
        row["map_path"] = str(mp) if mp is not None else None
        if mp is None or not mp.exists():
            row["roi_x1"] = np.nan
            row["roi_y1"] = np.nan
            row["roi_x2"] = np.nan
            row["roi_y2"] = np.nan
            row["roi_source"] = "missing"
            row["roi_area_ratio"] = np.nan
            rows.append(row)
            continue
        try:
            (x1, y1, x2, y2), area_ratio = extract_roi_from_map(
                mp,
                percentile=float(args.percentile),
                expand_ratio=float(args.expand_ratio),
                min_area_ratio=float(args.min_area_ratio),
            )
            row["roi_x1"] = x1
            row["roi_y1"] = y1
            row["roi_x2"] = x2
            row["roi_y2"] = y2
            row["roi_source"] = "attention_map"
            row["roi_area_ratio"] = area_ratio
        except Exception:
            row["roi_x1"] = np.nan
            row["roi_y1"] = np.nan
            row["roi_x2"] = np.nan
            row["roi_y2"] = np.nan
            row["roi_source"] = "error"
            row["roi_area_ratio"] = np.nan
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved ROI manifest: {out_path}")


if __name__ == "__main__":
    main()
