#!/usr/bin/env python3
"""Extract DADA raw video streams to LOTVS-style frame folders.

Expected raw file names:
    images_<category>_<clipid>.mp4
    maps_<category>_<clipid>.mp4

Outputs:
    <out_root>/<category>/<clipid>/images/0001.jpg
    <out_root>/<category>/<clipid>/maps/0001.jpg
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4"}

PATTERN = re.compile(r"^(images|maps)_(\d+)_([A-Za-z0-9]+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True, help="Directory containing raw videos")
    p.add_argument("--out-root", required=True, help="Output DADA_dataset root")
    p.add_argument("--recursive", action="store_true", help="Search raw-dir recursively")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing frame folders")
    p.add_argument("--grayscale-maps", action="store_true", default=True, help="Save maps as grayscale")
    p.add_argument("--report-json", default=None, help="Optional report path")
    return p.parse_args()


def find_videos(raw_dir: Path, recursive: bool) -> List[Path]:
    it = raw_dir.rglob("*") if recursive else raw_dir.glob("*")
    out = [p for p in it if p.is_file() and p.suffix in VIDEO_EXTS]
    out.sort()
    return out


def parse_video_name(path: Path) -> Optional[Tuple[str, str, str]]:
    m = PATTERN.match(path.stem)
    if not m:
        return None
    stream, category, clip_id = m.group(1), m.group(2), m.group(3)
    return stream, str(int(category)), str(clip_id).zfill(3)


def extract_video(video_path: Path, out_dir: Path, grayscale: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out_path = out_dir / f"{count:04d}.jpg"
        cv2.imwrite(str(out_path), frame)
    cap.release()
    return count


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    videos = find_videos(raw_dir, args.recursive)
    if not videos:
        raise RuntimeError(f"No videos found under {raw_dir}")

    rows: List[Dict[str, object]] = []
    for vp in tqdm(videos, desc="Extract DADA streams"):
        parsed = parse_video_name(vp)
        if parsed is None:
            print(f"[WARN] Skip unrecognized file name: {vp.name}")
            continue
        stream, category, clip_id = parsed
        out_dir = out_root / category / clip_id / stream
        if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
            num_existing = len([p for p in out_dir.iterdir() if p.is_file()])
            rows.append({
                "video_path": str(vp),
                "stream": stream,
                "category": category,
                "clip_id": clip_id,
                "num_frames": num_existing,
                "status": "skipped_existing",
                "out_dir": str(out_dir),
            })
            continue
        if args.overwrite and out_dir.exists():
            for p in out_dir.glob("*"):
                if p.is_file():
                    p.unlink()

        grayscale = bool(args.grayscale_maps and stream == "maps")
        num_frames = extract_video(vp, out_dir, grayscale=grayscale)
        rows.append({
            "video_path": str(vp),
            "stream": stream,
            "category": category,
            "clip_id": clip_id,
            "num_frames": num_frames,
            "status": "ok",
            "out_dir": str(out_dir),
        })

    report = {
        "raw_dir": str(raw_dir),
        "out_root": str(out_root),
        "num_files_seen": len(videos),
        "num_ok": sum(1 for r in rows if r["status"] == "ok"),
        "num_skipped_existing": sum(1 for r in rows if r["status"] == "skipped_existing"),
        "rows": rows[:20],
    }
    report_path = Path(args.report_json) if args.report_json else out_root / "extract_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
