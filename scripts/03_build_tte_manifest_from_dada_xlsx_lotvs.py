
#!/usr/bin/env python3
"""Build an onset-based DADA TTE frame manifest from the built-in DADA xlsx + LOTVS splits.

This script is tailored to the *actual* DADA2000 annotation workbook the user uploaded:
  - Sheet name: "Sheet1"
  - Relevant columns:
      video
      type
      whether an accident occurred (1/0)
      abnormal start frame
      accident frame
      abnormal end frame
      total frames

Important notes
---------------
1) The workbook's timing columns behave like 0-based interval boundaries:
      [0,tai] + [tai,tae] + [tae,end] == total frames
      t_co - t_ai == [tai,tco]
      t_ae - t_ai == [tai,tae]
   Therefore, when matching to extracted image files named 0001.jpg, 0002.jpg, ...
   the aligned 1-based frame indices are:
      first_abnormal_frame = t_ai + 1
      first_collision_frame = t_co + 1   (if accident happened)
      last_abnormal_frame_inclusive = t_ae
   We do NOT simply add +1 to t_ae, because t_ae is the exclusive end boundary
   in the original 0-based interval notation.

2) The LOTVS-DADA split JSON files use 54 *experiment categories*, while the DADA xlsx
   uses the original DADA paper type numbering. This script remaps xlsx type -> LOTVS
   experiment category automatically using the mapping in
   LOTVS-DADA/DADA_accident_categories/readme.md.

3) By default, this script builds the main benchmark only from clips where
      "whether an accident occurred (1/0)" == 1
   because onset-based TTE for "imminent accident" is not well-defined for no-accident clips.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# LOTVS 54-category experiment id -> original DADA paper type id
EXP_TO_PAPER_TYPE: Dict[int, int] = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11,
    12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19,
    20: 21, 21: 22, 22: 24, 23: 28, 24: 30, 25: 33, 26: 35, 27: 36, 28: 37,
    29: 38, 30: 39, 31: 40, 32: 41, 33: 42, 34: 43, 35: 44, 36: 45, 37: 46,
    38: 48, 39: 49, 40: 50, 41: 51, 42: 52, 43: 53, 44: 54, 45: 55, 46: 56,
    47: 57, 48: 58, 49: 59, 50: 60, 51: 61, 52: 62, 53: 63,
    # 54 == others (no stable original type id to map back to)
}
PAPER_TO_EXP_TYPE: Dict[int, int] = {paper: exp for exp, paper in EXP_TO_PAPER_TYPE.items()}


@dataclass
class ClipTiming:
    exp_category: str
    clip_id: str
    accident_occurred: int
    orig_type: int
    first_abnormal_frame: Optional[int]   # 1-based inclusive
    first_collision_frame: Optional[int]  # 1-based inclusive
    last_abnormal_frame: Optional[int]    # 1-based inclusive
    total_frames: Optional[int]
    fps: float
    source_row_index: int


@dataclass
class SampledFrame:
    frame_idx: int
    sample_tag: str
    target_tte_sec: Optional[float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split-root", required=True, help="Root created by 02_build_dada_splits.py")
    p.add_argument("--clips-csv", required=True, help="clips.csv produced by 02_build_dada_splits.py")
    p.add_argument("--annotation-xlsx", required=True, help="DADA2000 built-in xlsx")
    p.add_argument("--sheet-name", default="Sheet1", help="Use Sheet1 for DADA timing metadata")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--stream-name", default="images")
    p.add_argument("--event-frame", choices=["ai", "co"], default="ai",
                   help="ai = accident window start (recommended); co = collision start")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--sample-mode", choices=["all", "every_n", "tte_targets"], default="tte_targets")
    p.add_argument("--every-n", type=int, default=10)
    p.add_argument("--sample-tte-secs", nargs="*", type=float, default=[2.0, 1.0, 0.5, 0.2],
                   help="Representative target TTEs to sample when sample-mode=tte_targets")
    p.add_argument("--include-event-frame", action="store_true")
    p.add_argument("--positive-horizon-sec", type=float, default=1.0,
                   help="Binary positive if 0 < tte_sec <= this")
    p.add_argument("--negative-min-sec", type=float, default=2.0,
                   help="Binary negative if tte_sec >= this")
    p.add_argument("--include-aw-as-positive", action="store_true",
                   help="Optionally also label frames inside the accident window as positive")
    p.add_argument("--skip-no-accident", action="store_true", default=True,
                   help="Skip rows where whether an accident occurred == 0 (recommended)")
    p.add_argument("--keep-unlabeled", action="store_true",
                   help="Keep rows with NaN label_risk for future MCQ / ranking tasks")
    p.add_argument("--strict-metadata", action="store_true",
                   help="Fail if a split clip has no matching timing row")
    p.add_argument("--write-report-json", default=None)
    p.add_argument("--write-canonical-csv", default=None,
                   help="Optional canonical timing CSV for auditing")
    p.add_argument("--write-missing-csv", default=None,
                   help="Optional CSV of split clips that did not match any xlsx row")
    return p.parse_args()


def _int_or_none(x) -> Optional[int]:
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return None
    try:
        return int(round(float(x)))
    except Exception:
        return None


def _safe_str_int(x, pad: Optional[int] = None) -> Optional[str]:
    val = _int_or_none(x)
    if val is None:
        return None
    if pad is None:
        return str(val)
    return str(val).zfill(pad)


def load_clips_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "split": str,
            "alias": str,
            "category": str,
            "clip_id": str,
            "src_clip_dir": str,
            "dst_clip_dir": str,
            "method": str,
        },
    )


def load_dada_xlsx(path: Path, sheet_name: str, fps: float, skip_no_accident: bool) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    required = [
        "video",
        "type",
        "whether an accident occurred (1/0)",
        "abnormal start frame",
        "accident frame",
        "abnormal end frame",
        "total frames",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in xlsx sheet {sheet_name!r}: {missing}")

    out = pd.DataFrame({
        "video_raw": df["video"],
        "orig_type": df["type"],
        "accident_occurred": df["whether an accident occurred (1/0)"],
        "t_ai_raw": df["abnormal start frame"],
        "t_co_raw": df["accident frame"],
        "t_ae_raw": df["abnormal end frame"],
        "total_frames": df["total frames"],
    }).copy()

    # original DADA paper type -> LOTVS experiment category
    out["exp_category_num"] = out["orig_type"].map(PAPER_TO_EXP_TYPE)

    # clip id inside category
    out["clip_id"] = out["video_raw"].apply(lambda x: _safe_str_int(x, pad=3))

    # interpret the workbook as 0-based interval boundaries:
    # [0, t_ai), [t_ai, t_co), [t_co, t_ae), [t_ae, end)
    # convert to extracted-image indexing (0001.jpg = original frame 0):
    #   first abnormal frame      = t_ai + 1
    #   first collision frame     = t_co + 1
    #   last abnormal frame incl. = t_ae
    out["t_ai_raw"] = out["t_ai_raw"].apply(_int_or_none)
    out["t_co_raw"] = out["t_co_raw"].apply(_int_or_none)
    out["t_ae_raw"] = out["t_ae_raw"].apply(_int_or_none)
    out["total_frames"] = out["total_frames"].apply(_int_or_none)
    out["accident_occurred"] = out["accident_occurred"].apply(_int_or_none)
    out["orig_type"] = out["orig_type"].apply(_int_or_none)

    out["first_abnormal_frame"] = out["t_ai_raw"].apply(lambda x: None if x is None else x + 1)
    out["first_collision_frame"] = out.apply(
        lambda r: None if (r["t_co_raw"] is None or r["t_co_raw"] < 0 or r["accident_occurred"] == 0) else int(r["t_co_raw"] + 1),
        axis=1,
    )
    out["last_abnormal_frame"] = out["t_ae_raw"].apply(_int_or_none)
    out["fps"] = float(fps)

    # coverage / audit helpers
    out["exp_category"] = out["exp_category_num"].apply(lambda x: None if pd.isna(x) else str(int(x)))
    out["clip_key"] = out.apply(
        lambda r: None if (r["exp_category"] is None or r["clip_id"] is None) else f'{r["exp_category"]}_{r["clip_id"]}',
        axis=1,
    )

    if skip_no_accident:
        out = out[out["accident_occurred"] == 1].copy()

    return out.reset_index(drop=False).rename(columns={"index": "source_row_index"})


def make_timing_index(df: pd.DataFrame) -> Tuple[Dict[str, ClipTiming], List[int]]:
    timings: Dict[str, ClipTiming] = {}
    unmapped_types: List[int] = []

    for _, row in df.iterrows():
        if row["exp_category"] is None or row["clip_key"] is None:
            if row["orig_type"] is not None:
                unmapped_types.append(int(row["orig_type"]))
            continue
        key = str(row["clip_key"])
        if key in timings:
            # keep first occurrence
            continue
        timings[key] = ClipTiming(
            exp_category=str(row["exp_category"]),
            clip_id=str(row["clip_id"]),
            accident_occurred=int(row["accident_occurred"]),
            orig_type=int(row["orig_type"]),
            first_abnormal_frame=_int_or_none(row["first_abnormal_frame"]),
            first_collision_frame=_int_or_none(row["first_collision_frame"]),
            last_abnormal_frame=_int_or_none(row["last_abnormal_frame"]),
            total_frames=_int_or_none(row["total_frames"]),
            fps=float(row["fps"]),
            source_row_index=int(row["source_row_index"]),
        )
    return timings, sorted(set(unmapped_types))


def list_frames(folder: Path) -> List[Path]:
    frames = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    frames.sort()
    return frames


def resolve_frame_dir(split_root: Path, rec: Dict[str, str], stream_name: str) -> Optional[Path]:
    # Prefer the split alias directory, fallback to source clip dir
    split_name = str(rec["split"])
    alias = str(rec["alias"])
    p1 = split_root / split_name / alias / stream_name
    if p1.exists():
        return p1
    p2 = Path(str(rec.get("src_clip_dir", ""))) / stream_name
    if p2.exists():
        return p2
    return None


def choose_event_frame(timing: ClipTiming, event_frame: str) -> Optional[int]:
    if event_frame == "ai":
        return timing.first_abnormal_frame
    if event_frame == "co":
        return timing.first_collision_frame
    raise ValueError(event_frame)


def pick_indices(num_frames: int, mode: str, every_n: int, target_tte_secs: List[float],
                 event_frame_idx: Optional[int], fps: float, include_event_frame: bool) -> List[SampledFrame]:
    if num_frames <= 0:
        return []
    if mode == "all":
        return [SampledFrame(frame_idx=i, sample_tag="all", target_tte_sec=None) for i in range(1, num_frames + 1)]
    if mode == "every_n":
        step = max(1, int(every_n))
        return [SampledFrame(frame_idx=i, sample_tag=f"every_{step}", target_tte_sec=None) for i in range(1, num_frames + 1, step)]
    if event_frame_idx is None or fps <= 0:
        return []
    picked: Dict[int, SampledFrame] = {}
    for tte_sec in target_tte_secs:
        if tte_sec < 0:
            continue
        idx = int(round(event_frame_idx - tte_sec * fps))
        if idx < 1 or idx > num_frames:
            continue
        picked[idx] = SampledFrame(frame_idx=idx, sample_tag="tte_target", target_tte_sec=float(tte_sec))
    if include_event_frame and event_frame_idx is not None and 1 <= event_frame_idx <= num_frames:
        picked[event_frame_idx] = SampledFrame(frame_idx=event_frame_idx, sample_tag="event_frame", target_tte_sec=0.0)
    return [picked[k] for k in sorted(picked)]


def phase_name(frame_idx: int, timing: ClipTiming) -> str:
    if timing.first_abnormal_frame is None:
        return "unknown"
    if frame_idx < timing.first_abnormal_frame:
        return "before_aw"
    if timing.last_abnormal_frame is not None and frame_idx <= timing.last_abnormal_frame:
        return "accident_window"
    return "after_aw"


def tte_sec(frame_idx: int, event_frame_idx: Optional[int], fps: float) -> Optional[float]:
    if event_frame_idx is None or fps <= 0:
        return None
    return max(float(event_frame_idx - frame_idx) / float(fps), 0.0)


def ttc_sec_to_co(frame_idx: int, timing: ClipTiming) -> Optional[float]:
    if timing.first_collision_frame is None or timing.fps <= 0:
        return None
    return max(float(timing.first_collision_frame - frame_idx) / float(timing.fps), 0.0)


def tte_bin(frame_idx: int, event_frame_idx: Optional[int], timing: ClipTiming, fps: float) -> Optional[str]:
    if event_frame_idx is None or fps <= 0:
        return None
    if frame_idx < event_frame_idx:
        sec = (event_frame_idx - frame_idx) / fps
        if sec <= 1.0:
            return "pre_0_1s"
        if sec <= 2.0:
            return "pre_1_2s"
        if sec <= 3.0:
            return "pre_2_3s"
        return "pre_over_3s"
    if timing.last_abnormal_frame is not None and frame_idx <= timing.last_abnormal_frame:
        return "accident_window"
    return "post_event"


def binary_label_and_role(frame_idx: int, event_frame_idx: Optional[int], timing: ClipTiming,
                          positive_horizon_sec: float, negative_min_sec: float,
                          include_aw_as_positive: bool) -> Tuple[float, str]:
    if event_frame_idx is None or timing.fps <= 0:
        return np.nan, "missing_event"
    if frame_idx < event_frame_idx:
        sec = (event_frame_idx - frame_idx) / timing.fps
        if 0.0 < sec <= positive_horizon_sec:
            return 1.0, "positive_imminent"
        if sec >= negative_min_sec:
            return 0.0, "negative_far"
        return np.nan, "gap_ignore"
    if include_aw_as_positive and timing.last_abnormal_frame is not None and frame_idx <= timing.last_abnormal_frame:
        return 1.0, "positive_in_aw"
    return np.nan, "post_event_ignore"


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root).resolve()
    clips_csv = Path(args.clips_csv).resolve()
    xlsx_path = Path(args.annotation_xlsx).resolve()
    output_csv = Path(args.output_csv).resolve()
    report_json = Path(args.write_report_json) if args.write_report_json else output_csv.with_name(output_csv.stem + "_report.json")
    canonical_csv = Path(args.write_canonical_csv) if args.write_canonical_csv else output_csv.with_name(output_csv.stem + "_canonical_timing.csv")
    missing_csv = Path(args.write_missing_csv) if args.write_missing_csv else output_csv.with_name(output_csv.stem + "_missing_clips.csv")

    clips_df = load_clips_csv(clips_csv)
    ann_df = load_dada_xlsx(xlsx_path, args.sheet_name, args.fps, bool(args.skip_no_accident))
    timings, unmapped_types = make_timing_index(ann_df)

    if args.write_canonical_csv:
        canonical_csv.parent.mkdir(parents=True, exist_ok=True)
        ann_df.to_csv(canonical_csv, index=False)

    rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []
    count_mismatches: List[Dict[str, object]] = []

    for rec in tqdm(clips_df.to_dict("records"), total=len(clips_df), desc="Build TTE manifest from xlsx"):
        category = str(rec["category"])
        clip_id = str(rec["clip_id"])
        key = f"{category}_{clip_id}"
        timing = timings.get(key)
        if timing is None:
            missing_rows.append({
                "split": rec["split"],
                "alias": rec["alias"],
                "category": category,
                "clip_id": clip_id,
                "clip_key": key,
                "reason": "no_matching_timing_row_after_type_remap_or_xlsx_missing",
            })
            if args.strict_metadata:
                raise RuntimeError(f"Missing timing row for split clip: {key}")
            continue

        frame_dir = resolve_frame_dir(split_root, rec, args.stream_name)
        if frame_dir is None or not frame_dir.exists():
            missing_rows.append({
                "split": rec["split"],
                "alias": rec["alias"],
                "category": category,
                "clip_id": clip_id,
                "clip_key": key,
                "reason": f"missing_frame_dir:{frame_dir}",
            })
            if args.strict_metadata:
                raise RuntimeError(f"Missing frame directory for split clip: {key}")
            continue

        frames = list_frames(frame_dir)
        num_frames = len(frames)
        if num_frames == 0:
            missing_rows.append({
                "split": rec["split"],
                "alias": rec["alias"],
                "category": category,
                "clip_id": clip_id,
                "clip_key": key,
                "reason": "empty_frame_dir",
            })
            if args.strict_metadata:
                raise RuntimeError(f"Empty frame directory for split clip: {key}")
            continue

        if timing.total_frames is not None and num_frames != timing.total_frames:
            count_mismatches.append({
                "clip_key": key,
                "split": rec["split"],
                "alias": rec["alias"],
                "num_frames_found": num_frames,
                "total_frames_xlsx": timing.total_frames,
                "delta": int(num_frames - timing.total_frames),
            })

        event_frame_idx = choose_event_frame(timing, args.event_frame)
        picked = pick_indices(num_frames, args.sample_mode, args.every_n, list(args.sample_tte_secs), event_frame_idx, timing.fps, bool(args.include_event_frame))
        for s in picked:
            frame_idx = int(s.frame_idx)
            if frame_idx < 1 or frame_idx > num_frames:
                continue
            label_risk, eval_role = binary_label_and_role(
                frame_idx=frame_idx,
                event_frame_idx=event_frame_idx,
                timing=timing,
                positive_horizon_sec=float(args.positive_horizon_sec),
                negative_min_sec=float(args.negative_min_sec),
                include_aw_as_positive=bool(args.include_aw_as_positive),
            )
            if (not args.keep_unlabeled) and pd.isna(label_risk):
                continue

            img_path = frames[frame_idx - 1]
            rows.append({
                "split": rec["split"],
                "alias": rec["alias"],
                "category": category,
                "clip_id": clip_id,
                "clip_key": key,
                "image_path": str(img_path),
                "frame_dir": str(frame_dir),
                "frame_idx": frame_idx,           # 1-based aligned to extracted frames
                "num_frames_found": num_frames,
                "orig_type_from_xlsx": timing.orig_type,
                "accident_occurred": timing.accident_occurred,
                "fps": timing.fps,
                "t_ai_raw_boundary_0based": timing.first_abnormal_frame - 1 if timing.first_abnormal_frame is not None else np.nan,
                "t_co_raw_boundary_0based": timing.first_collision_frame - 1 if timing.first_collision_frame is not None else np.nan,
                "t_ae_raw_boundary_0based": timing.last_abnormal_frame if timing.last_abnormal_frame is not None else np.nan,
                "t_ai_frame_1based": timing.first_abnormal_frame if timing.first_abnormal_frame is not None else np.nan,
                "t_co_frame_1based": timing.first_collision_frame if timing.first_collision_frame is not None else np.nan,
                "t_ae_last_frame_1based": timing.last_abnormal_frame if timing.last_abnormal_frame is not None else np.nan,
                "event_frame_type": args.event_frame,
                "event_frame_idx": event_frame_idx if event_frame_idx is not None else np.nan,
                "phase": phase_name(frame_idx, timing),
                "tte_sec": tte_sec(frame_idx, event_frame_idx, timing.fps),
                "ttc_sec": tte_sec(frame_idx, event_frame_idx, timing.fps),   # backward compatibility
                "ttc_sec_to_co": ttc_sec_to_co(frame_idx, timing),
                "tte_bin": tte_bin(frame_idx, event_frame_idx, timing, timing.fps),
                "label_risk": label_risk,
                "eval_role": eval_role,
                "sample_tag": s.sample_tag,
                "target_tte_sec": s.target_tte_sec if s.target_tte_sec is not None else np.nan,
                "source_row_index_xlsx": timing.source_row_index,
                "is_pre_event": 1 if (event_frame_idx is not None and frame_idx < event_frame_idx) else 0,
                "attack_eval_candidate_gt": 1 if (not pd.isna(label_risk) and float(label_risk) == 1.0) else 0,
                "safe_neg_candidate_gt": 1 if (not pd.isna(label_risk) and float(label_risk) == 0.0) else 0,
            })

    if not rows:
        raise RuntimeError("No rows produced. Check split-root, clips.csv, xlsx sheet name, and whether the frame folders exist.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)

    missing_df = pd.DataFrame(missing_rows)
    if len(missing_df) > 0:
        missing_df.to_csv(missing_csv, index=False)

    report = {
        "annotation_xlsx": str(xlsx_path),
        "sheet_name": args.sheet_name,
        "clips_csv": str(clips_csv),
        "split_root": str(split_root),
        "output_csv": str(output_csv),
        "event_frame": args.event_frame,
        "fps": args.fps,
        "sample_mode": args.sample_mode,
        "sample_tte_secs": list(args.sample_tte_secs),
        "positive_horizon_sec": args.positive_horizon_sec,
        "negative_min_sec": args.negative_min_sec,
        "skip_no_accident": bool(args.skip_no_accident),
        "total_split_clips": int(len(clips_df)),
        "matched_timing_clips": int(len(set(out_df["clip_key"].tolist()))),
        "produced_rows": int(len(out_df)),
        "missing_split_clips": int(len(missing_rows)),
        "missing_reason_examples": missing_rows[:10],
        "frame_count_mismatch_count": int(len(count_mismatches)),
        "frame_count_mismatch_examples": count_mismatches[:10],
        "unmapped_xlsx_types": unmapped_types,
        "label_distribution": {str(k): int(v) for k, v in out_df["label_risk"].value_counts(dropna=False).to_dict().items()},
        "tte_bin_distribution": {str(k): int(v) for k, v in out_df["tte_bin"].value_counts(dropna=False).to_dict().items()},
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved manifest: {output_csv}")
    print(f"Saved report:   {report_json}")
    if len(missing_rows) > 0:
        print(f"Saved missing:  {missing_csv}")


if __name__ == "__main__":
    main()
