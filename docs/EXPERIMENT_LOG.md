# LLaVA Experiment Log

Updated: 2026-03-23

## Goal

Find a LLaVA-family victim baseline that:
- works on clean imminent-accident frames well enough to be meaningful
- can then be evaluated under overlay attacks
- is practical on the laptop GPU

## Experiment 1: LLaVA-OneVision Qwen2 0.5B, original prompt

Artifacts:
- `projects/work/baseline_llava/llava_0p5b/llava_0p5b_preds.jsonl`
- `projects/work/baseline_llava/llava_0p5b/eval/clean_metrics.json`

Observed behavior:
- clean subset (`n=100`) produced almost all `no`
- `risk_accuracy = 0.5`
- `risk_f1 = 0.0`
- `false_negative_rate = 1.0`

Interpretation:
- pipeline worked
- model was too conservative to serve as a usable victim baseline

## Experiment 2: LLaVA 1.5 7B, original prompt

Artifacts:
- `projects/work/baseline_llava/llava_7b/llava_7b_preds.jsonl`
- `projects/work/baseline_llava/llava_7b/eval/clean_metrics.json`

Observed behavior:
- raw outputs often looked like truncated JSON
- after eval-side recovery, clean subset (`n=100`) still behaved like almost-all `no`
- `risk_accuracy = 0.5`
- `risk_f1 = 0.0`
- `false_negative_rate = 1.0`

Interpretation:
- not mainly a parser bug
- this checkpoint/prompt combination still failed the clean task

## Experiment 3: 0.5B prompt stress test on clean probe 40

Prompt:
- `projects/codes/safety_attack_dada_bundle/safety_attack_dada/prompts/binary_imminent_risk_prompt_v2.txt`

Artifacts:
- `projects/work/baseline_llava/debug/clean_probe_40.csv`
- `projects/work/baseline_llava/debug/llava_0p5b_probe_v2.jsonl`
- `projects/codes/safety_attack_dada_bundle/logs/llava_0p5b_probe_v2.log`

Observed behavior:
- with the new binary prompt, 0.5B flipped from almost-all `no` to almost-all `yes`
- on the balanced clean probe it predicted `yes` for all 40 samples
- accuracy stayed near `0.5`

Interpretation:
- 0.5B is highly prompt-sensitive
- it is not stable enough for this benchmark

## Experiment 4: LLaVA 1.5 7B prompt stress test on clean probe 20

Prompt:
- `projects/codes/safety_attack_dada_bundle/safety_attack_dada/prompts/binary_imminent_risk_prompt_v3_json.txt`

Artifacts:
- `projects/work/baseline_llava/debug/clean_probe_20.csv`
- `projects/work/baseline_llava/debug/llava_7b_probe_v3.jsonl`

Observed behavior:
- still produced near-constant `risk = no`, `risk_score = 3`
- clean probe accuracy stayed near `0.5`

Interpretation:
- older LLaVA 1.5 7B did not meaningfully respond to prompt improvements on this task

## Experiment 5: LLaVA-OneVision Qwen2 7B, fp16/bf16

Artifacts:
- `projects/work/baseline_llava/debug/llava_onevision_7b_probe_v3.jsonl`
- `projects/work/baseline_llava/debug/llava_onevision_7b_probe_v3_retry.jsonl`
- `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_probe_v3.log`
- `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_probe_v3_retry.log`

Observed behavior:
- model loaded, but first sample hit `CUBLAS_STATUS_INTERNAL_ERROR` / CUDA OOM
- subsequent samples failed with CUDA OOM

Interpretation:
- this checkpoint is too heavy for stable fp16/bf16 inference on the laptop GPU
- memory reduction is required

## Experiment 6: LLaVA-OneVision Qwen2 7B, 4-bit NF4 + CPU offload

Code changes that enabled this:
- `06_run_vlm_attack_hf.py` now supports:
  - `--cache-dir`
  - `--local-files-only`
  - auto resolution of local HF snapshots from cache
  - `--clear-cuda-cache`
  - 4-bit loading with NF4 / double quant

Artifacts:
- `projects/work/baseline_llava/debug/llava_onevision_7b_probe_v3_4bit.jsonl`
- `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_probe_v3_4bit.log`

Prompt:
- `projects/codes/safety_attack_dada_bundle/safety_attack_dada/prompts/binary_imminent_risk_prompt_v3_json.txt`

Observed behavior on balanced clean probe 20:
- stable inference completed for all 20 samples
- `parsed_ok_rate = 1.0`
- `accuracy = 0.45`
- `precision = 0.4615`
- `recall = 0.6`
- `f1 = 0.5217`
- `false_negative_rate = 0.4`
- `false_positive_rate = 0.7`

Interpretation:
- this is the first LLaVA-family run that shows real yes/no discrimination instead of collapsing to all `yes` or all `no`
- however, clean performance is still too weak to treat as the final victim baseline
- especially, `FPR = 0.7` is much too high

## Current Takeaway

Best current LLaVA-family candidate:
- `llava-hf/llava-onevision-qwen2-7b-ov-hf`

Why it is best so far:
- can be made to run on the laptop with 4-bit loading
- produces non-degenerate predictions

Why it is not yet sufficient:
- clean discrimination is still weak on the current probe
- it needs a larger clean-only validation before we trust it

## Next Steps

1. Run a larger clean-only evaluation with OneVision 7B 4-bit.
2. If clean performance improves enough, run attacked subsets next.
3. If clean performance remains weak, try another LLaVA-family 7B checkpoint rather than scaling directly to full pilot.

## Experiment 7: OneVision 7B 4-bit on balanced clean100

Artifacts:
- `projects/work/baseline_llava/llava_onevision_7b/clean_only_100.csv`
- `projects/work/baseline_llava/llava_onevision_7b/clean_only_100_preds.jsonl`
- `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_clean100_4bit.log`

Observed behavior on balanced clean100 (`50` positive, `50` negative):
- stable inference completed for all `100` samples
- `parsed_ok_rate = 1.0`
- `accuracy = 0.51`
- `precision = 0.5063`
- `recall = 0.8`
- `f1 = 0.6202`
- `false_negative_rate = 0.2`
- `false_positive_rate = 0.78`

Interpretation:
- this is the first LLaVA-family checkpoint that meaningfully covers positive imminent-accident frames on a larger clean set
- it is now usable for preliminary attack experiments because clean positive recall is non-trivial
- however, it is still not a polished final victim baseline because the clean false-positive rate remains very high

## Experiment 8: OneVision 7B 4-bit on attacked subset 80

Setup:
- clean attack candidates were defined as `GT positive` and `clean prediction = yes`
- balanced clean100 produced `40` such candidate frames
- a random-placement attacked subset of `80` rows was sampled from the pilot benchmark:
  - `4` overlay types
  - `2` severities
  - `10` attacked rows per `(overlay, severity)`

Artifacts:
- `projects/work/baseline_llava/llava_onevision_7b/attack_subset_80.csv`
- `projects/work/baseline_llava/llava_onevision_7b/attack_subset_80_preds.jsonl`
- `projects/work/baseline_llava/llava_onevision_7b/attack_subset_80_metrics.json`
- `projects/work/baseline_llava/llava_onevision_7b/attack_subset_80_group_metrics.csv`
- `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_attack80_4bit.log`

Observed behavior:
- stable inference completed for all `80` attacked samples
- `parsed_ok_rate = 1.0`
- overall `frame_asr_fn_strict = 0.1`
- attack effect was concentrated in severity `4` overlays

By attack group:
- `logo_patch`, severity `2`: `ASR = 0.0`
- `logo_patch`, severity `4`: `ASR = 0.2`
- `semi_transparent_bar`, severity `2`: `ASR = 0.0`
- `semi_transparent_bar`, severity `4`: `ASR = 0.1`
- `text_watermark`, severity `2`: `ASR = 0.0`
- `text_watermark`, severity `4`: `ASR = 0.3`
- `timestamp_box`, severity `2`: `ASR = 0.0`
- `timestamp_box`, severity `4`: `ASR = 0.2`

Interpretation:
- OneVision 7B is now strong enough that random overlays can measurably suppress some clean-positive accident warnings
- the current random-placement pilot attack is still too weak to be a convincing final attack benchmark
- the most promising direction is to scale severity `4` first and then test more targeted placement modes once available

## Experiment 9: OneVision 7B 4-bit on severity-4 all-candidate set (`n=160`)

Setup:
- used all `40` clean-positive candidate base frames from balanced clean100
- evaluated only the stronger `severity = 4` attacks
- `4` overlay types x `40` candidate base frames = `160` attacked samples
- an interrupted run was resumed successfully with `--skip-existing` after a machine shutdown

Artifacts:
- `projects/work/baseline_llava/llava_onevision_7b/attack_severity4_all_160.csv`
- `projects/work/baseline_llava/llava_onevision_7b/attack_severity4_all_160_preds.jsonl`
- `projects/work/baseline_llava/llava_onevision_7b/attack_severity4_all_160_metrics.json`
- `projects/work/baseline_llava/llava_onevision_7b/attack_severity4_all_160_group_metrics.csv`
- `projects/codes/safety_attack_dada_bundle/logs/llava_onevision_7b_attack_sev4_160_4bit.log`

Observed behavior:
- resume worked as intended: the crash happened after `128/160` had already been written, and the rerun completed the remaining `32`
- `parsed_ok_rate = 1.0`
- overall `frame_asr_fn_strict = 0.0625`
- `attacked_yes_rate = 0.9375`

By overlay:
- `logo_patch`: `ASR = 0.075`
- `semi_transparent_bar`: `ASR = 0.025`
- `text_watermark`: `ASR = 0.075`
- `timestamp_box`: `ASR = 0.075`

Interpretation:
- the pipeline is now resilient to interruptions and can safely resume long LLaVA runs
- the victim model remains usable on clean positives, but random-placement severity-4 overlays only suppress a small minority of warnings at scale
- this strengthens the conclusion that the next bottleneck is attack strength and placement, not just victim-model selection
