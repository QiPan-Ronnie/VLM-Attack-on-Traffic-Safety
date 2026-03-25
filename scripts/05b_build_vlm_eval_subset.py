#!/usr/bin/env python3
"""Build a balanced VLM evaluation subset from a safety-attack manifest."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--clean-total", type=int, default=100)
    p.add_argument("--per-condition-total", type=int, default=50)
    p.add_argument("--placement-mode", default="random")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def balanced_sample(df: pd.DataFrame, total: int, seed: int) -> pd.DataFrame:
    if total <= 0 or df.empty:
        return df.head(0).copy()
    if len(df) <= total:
        return df.sample(frac=1.0, random_state=seed).copy()

    if "label_risk" not in df.columns:
        return df.sample(n=total, random_state=seed).copy()

    pos = df[pd.to_numeric(df["label_risk"], errors="coerce") == 1].copy()
    neg = df[pd.to_numeric(df["label_risk"], errors="coerce") == 0].copy()
    if pos.empty or neg.empty:
        return df.sample(n=min(total, len(df)), random_state=seed).copy()

    pos_target = min(len(pos), total // 2)
    neg_target = min(len(neg), total - pos_target)
    remainder = total - pos_target - neg_target
    if remainder > 0:
        if len(pos) - pos_target >= len(neg) - neg_target:
            pos_target = min(len(pos), pos_target + remainder)
        else:
            neg_target = min(len(neg), neg_target + remainder)

    sampled = [
        pos.sample(n=pos_target, random_state=seed) if pos_target else pos.head(0).copy(),
        neg.sample(n=neg_target, random_state=seed + 1) if neg_target else neg.head(0).copy(),
    ]
    out = pd.concat(sampled, ignore_index=True)
    if len(out) < total:
        if "variant_id" in df.columns and "variant_id" in out.columns:
            remaining = df[~df["variant_id"].isin(out["variant_id"])].copy()
        else:
            remaining = df.copy()
        extra = min(total - len(out), len(remaining))
        if extra > 0:
            out = pd.concat([out, remaining.sample(n=extra, random_state=seed + 2)], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    out_path = Path(args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    if args.placement_mode and "placement_mode" in df.columns:
        keep_clean = df["overlay_type"] == "clean"
        keep_mode = df["placement_mode"] == args.placement_mode
        df = df[keep_clean | keep_mode].copy()

    samples: List[pd.DataFrame] = []

    clean = df[df["overlay_type"] == "clean"].copy()
    samples.append(balanced_sample(clean, int(args.clean_total), int(args.seed)))

    pert = df[df["overlay_type"] != "clean"].copy()
    if not pert.empty:
        group_cols = ["overlay_type", "severity"]
        if "placement_mode" in pert.columns:
            group_cols.append("placement_mode")
        for idx, (_, group) in enumerate(pert.groupby(group_cols, dropna=False)):
            samples.append(balanced_sample(group, int(args.per_condition_total), int(args.seed) + idx + 10))

    out = pd.concat(samples, ignore_index=True)
    out = out.drop_duplicates(subset=["variant_id"]).reset_index(drop=True)
    out.to_csv(out_path, index=False)

    print(f"Saved VLM subset to: {out_path}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()
