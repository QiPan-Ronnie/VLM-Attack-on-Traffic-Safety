# CSCI699 Baseline Status

Updated: 2026-03-23

## Current Stage

The pilot baseline stage is functionally complete.

Completed:
- Environment `CSCI699` is working with CUDA PyTorch on the laptop GPU.
- Pilot attack benchmark was generated from the clean master manifest.
- CLIP full-pilot baseline was run and evaluated.
- LLaVA 0.5B subset baseline was run and evaluated.
- LLaVA 1.5 7B subset baseline was run and evaluated.
- LLaVA-OneVision Qwen2 7B was made stable on the laptop via 4-bit loading and local HF cache.
- A balanced `clean100` validation run was completed for OneVision 7B.
- A first attacked subset run (`attack_subset_80`) was completed for OneVision 7B.
- Long-running CLIP and LLaVA scripts now support tail-able progress logs.
- OneVision 7B full-clean `6935` run has been launched with resumable background execution and unique per-run logs.

Still blocked:
- Formal `test-only` baseline cannot be rebuilt yet because LOTVS split JSON files are still missing locally.

## Key Outputs

Pilot benchmark:
- `projects/work/baseline_llava/pilot_attack/manifest_attack.csv`
- `projects/work/baseline_llava/pilot_attack/summary.json`

CLIP:
- `projects/work/baseline_llava/clip/clip_attack_preds.jsonl`
- `projects/work/baseline_llava/clip/eval/clean_metrics.json`

LLaVA 0.5B:
- `projects/work/baseline_llava/llava_0p5b/vlm_eval_subset.csv`
- `projects/work/baseline_llava/llava_0p5b/llava_0p5b_preds.jsonl`
- `projects/work/baseline_llava/llava_0p5b/eval/clean_metrics.json`

LLaVA 7B:
- `projects/work/baseline_llava/llava_7b/vlm_eval_subset.csv`
- `projects/work/baseline_llava/llava_7b/llava_7b_preds.jsonl`
- `projects/work/baseline_llava/llava_7b/eval/clean_metrics.json`

LLaVA-OneVision 7B:
- `projects/work/baseline_llava/llava_onevision_7b/clean_only_100.csv`
- `projects/work/baseline_llava/llava_onevision_7b/clean_only_100_preds.jsonl`
- `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935.csv`
- `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_summary.json`
- `projects/work/baseline_llava/llava_onevision_7b/attack_subset_80.csv`
- `projects/work/baseline_llava/llava_onevision_7b/attack_subset_80_preds.jsonl`
- `projects/work/baseline_llava/llava_onevision_7b/attack_subset_80_metrics.json`
- `projects/work/baseline_llava/llava_onevision_7b/attack_subset_80_group_metrics.csv`
- `projects/work/baseline_llava/llava_onevision_7b/attack_severity4_all_160.csv`
- `projects/work/baseline_llava/llava_onevision_7b/attack_severity4_all_160_preds.jsonl`
- `projects/work/baseline_llava/llava_onevision_7b/attack_severity4_all_160_metrics.json`
- `projects/work/baseline_llava/llava_onevision_7b/attack_severity4_all_160_group_metrics.csv`

Logs:
- `projects/codes/safety_attack_dada_bundle/logs/clip_smoke.log`
- `projects/codes/safety_attack_dada_bundle/logs/llava_0p5b_subset.log`
- `projects/codes/safety_attack_dada_bundle/logs/llava_7b_pilot.log`
- `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_clean100_4bit.log`
- `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_attack80_4bit.log`
- `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_attack_sev4_160_4bit.log`
- `projects/work/baseline_llava/llava_onevision_7b/clean_only_6935_latest_run.json`

## Headline Metrics

CLIP clean metrics on full pilot clean frames (`n=6935`):
- `parse_success_rate = 1.0`
- `risk_accuracy = 0.7384`
- `risk_f1 = 0.8452`
- `false_negative_rate = 0.1072`
- `false_positive_rate = 0.8795`

LLaVA 0.5B clean metrics on subset clean frames (`n=100`):
- `parse_success_rate = 1.0`
- `risk_accuracy = 0.5`
- `risk_f1 = 0.0`
- `false_negative_rate = 1.0`
- `false_positive_rate = 0.0`

LLaVA 7B clean metrics on subset clean frames (`n=100`):
- `parse_success_rate = 1.0`
- `risk_accuracy = 0.5`
- `risk_f1 = 0.0`
- `false_negative_rate = 1.0`
- `false_positive_rate = 0.0`

LLaVA-OneVision 7B clean metrics on balanced clean100 (`n=100`):
- `parse_success_rate = 1.0`
- `risk_accuracy = 0.51`
- `risk_precision = 0.5063`
- `risk_recall = 0.8`
- `risk_f1 = 0.6202`
- `false_negative_rate = 0.2`
- `false_positive_rate = 0.78`

LLaVA-OneVision 7B attacked subset metrics (`n=80` attacked rows from clean-positive candidates):
- `parse_success_rate = 1.0`
- `frame_asr_fn_strict = 0.1`
- severity `2` overlays: `ASR = 0.0` across all four attack types
- severity `4` overlays: `ASR` ranged from `0.1` to `0.3`
- strongest current group: `text_watermark`, severity `4` with `ASR = 0.3`

LLaVA-OneVision 7B severity-4 all-candidate metrics (`n=160`, `40` clean-positive candidate frames):
- `parse_success_rate = 1.0`
- `frame_asr_fn_strict = 0.0625`
- `logo_patch = 0.075`
- `semi_transparent_bar = 0.025`
- `text_watermark = 0.075`
- `timestamp_box = 0.075`

## Notes

- The 7B raw JSONL often contains truncated JSON-like strings such as `risk` and `risk_score` without a closed JSON object. Evaluation now recovers these predictions from `raw_output`, so the current eval artifacts are valid even though many raw rows still show `parsed_ok=false`.
- The 7B run succeeded on this machine by using GPU plus CPU offload. Early layers stayed on `cuda:0` and later layers were offloaded to CPU.
- After a reboot, CUDA PyTorch import was blocked by `torch\\lib\\nvrtc64_120_0.alt.dll`. The workaround that restored GPU imports was to rename that file so Windows policy no longer blocked `import torch`.
- OneVision 7B required `4-bit NF4` loading plus local cache resolution; fp16/bf16 inference was not stable on the laptop GPU.
- OneVision 7B is the first LLaVA-family checkpoint in this project that shows usable clean recall on imminent-accident frames.
- The current random-placement pilot attacks do affect OneVision 7B at severity `4`, but the observed attack success rate is still modest. This means the victim model is now more credible, while the attack side likely still needs stronger placement or larger coverage.
- The interrupted `severity4_all_160` run resumed cleanly from `128/160` using `--skip-existing`, so the long-run workflow is now robust to accidental shutdowns.

## Recommended Next Phase

Priority order:
1. Scale OneVision 7B from `attack_subset_80` to a larger attacked candidate set, prioritizing severity `4`.
2. Add or regenerate more targeted placement modes so attacks are not limited to `random` placement.
3. Compare CLIP vs older LLaVA checkpoints vs OneVision 7B in the project write-up, explicitly separating clean victim quality from attack strength.
4. Once LOTVS split files are available, rebuild the formal `test-only` manifest and rerun the same pipeline on the true test split.
