# VLM Attack on Traffic Safety

This repository contains the code, prompts, lightweight results, and handoff docs for a traffic-safety VLM robustness benchmark built on DADA-2000.

## Goal

We study whether visual overlays can break imminent-accident detection on traffic frames:

- clean frame: the model should predict that an accident is likely to start within about 1 second
- attacked frame: the model may flip from `yes` to `no`, or warn later than under clean input

The project focuses on:

- `DADA-2000`
- overlay attacks on accident-imminent frames
- clean baseline evaluation first
- frame-level evasion ASR and clip-level warning-delay ASR

## Current Formal Benchmark Target

- Clean frames: `6935`
- Overlay types:
  - `text_watermark`
  - `timestamp_box`
  - `logo_patch`
  - `semi_transparent_bar`
- Placement modes:
  - `random`
  - `critical`
  - `background`
- Severities:
  - `2`
  - `4`
- Seed: `42`
- Alert threshold: `risk_score >= 3` or `risk = "yes"`
- Delay threshold: `0.5s`

Expected benchmark size:

```text
6935 clean
+ 6935 x 4 overlays x 3 placements x 2 severities
= 173375 rows
```

## Repository Layout

```text
configs/
docs/
prompts/
results/local_summary/
scripts/
tools/
README.md
README_CN.md
requirements.txt
```

## What Is Included

- runnable scripts
- current prompt variants
- helper PowerShell tools
- Chinese handoff and experiment summary
- lightweight local result summaries

## What Is Not Included

Large local-only artifacts are intentionally excluded from Git:

- HF cache
- full prediction jsonl files
- generated images
- heavy logs
- local absolute-path manifests

## Start Here

- Chinese full project handoff: `docs/PROJECT_SUMMARY_CN.md`
- Current formal benchmark config: `configs/benchmark_dada2000_formal.yaml`
- Main prompt: `prompts/binary_imminent_risk_prompt.txt`
