#!/usr/bin/env python3
"""Run a Hugging Face VLM on the safety-attack manifest."""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BitsAndBytesConfig, pipeline

from _manifest_paths import infer_manifest_root, resolve_manifest_path

DEFAULT_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "binary_imminent_risk_prompt.txt"
VALID_RISK = {"yes", "no"}
YES_NO_ONLY_RE = re.compile(r"^\s*(yes|no)\s*[\.\!\?]*\s*$", flags=re.IGNORECASE)
PARTIAL_RISK_RE = re.compile(r'"?risk"?\s*:\s*"?(yes|no)"?', flags=re.IGNORECASE)
PARTIAL_RISK_SCORE_RE = re.compile(r'"?risk_score"?\s*:\s*([1-5])', flags=re.IGNORECASE)
PARTIAL_REASON_RE = re.compile(r'"?reason"?\s*:\s*"([^"\n\r}]*)', flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--data-root", default=None, help="Optional root used to resolve relative generated_image_path values")
    p.add_argument("--prompt-file", default=str(DEFAULT_PROMPT))
    p.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--offload-dir", default=None)
    p.add_argument("--attn-implementation", default="auto", choices=["auto", "eager", "sdpa", "flash_attention_2"])
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--local-files-only", action="store_true", help="Load model/tokenizer from local HF cache only")
    p.add_argument("--clear-cuda-cache", action="store_true", help="Call gc.collect() and torch.cuda.empty_cache() after each sample")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--log-file", default=None, help="Optional progress log written line-by-line for tailing")
    return p.parse_args()


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def dtype_from_str(name: str):
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def extract_prediction(text: str) -> Optional[Dict[str, Any]]:
    normalized_text = text.replace("\\_", "_")
    obj = extract_json(text)
    if obj is not None:
        return obj
    m = YES_NO_ONLY_RE.match(normalized_text.strip())
    if m:
        return {"risk": m.group(1).lower()}
    partial_risk = PARTIAL_RISK_RE.search(normalized_text)
    partial_score = PARTIAL_RISK_SCORE_RE.search(normalized_text)
    partial_reason = PARTIAL_REASON_RE.search(normalized_text)
    if partial_risk or partial_score or partial_reason:
        out: Dict[str, Any] = {}
        if partial_risk:
            out["risk"] = partial_risk.group(1).lower()
        if partial_score:
            out["risk_score"] = int(partial_score.group(1))
        if partial_reason:
            out["reason"] = partial_reason.group(1).strip()
        return out
    tokens = re.findall(r"[A-Za-z]+", normalized_text.strip().lower())
    if 0 < len(tokens) <= 3 and tokens[0] in VALID_RISK:
        return {"risk": tokens[0]}
    return None


def normalize(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if obj is None:
        return {
            "parsed_ok": False,
            "risk": None,
            "risk_score": None,
            "reason": None,
            "pred_label": None,
        }
    risk = str(obj.get("risk", "")).strip().lower()
    if risk not in VALID_RISK:
        risk = None
    risk_score = obj.get("risk_score", None)
    try:
        risk_score = int(risk_score) if risk_score is not None else None
    except Exception:
        risk_score = None
    if risk_score is not None:
        risk_score = max(1, min(5, risk_score))
    reason = obj.get("reason", None)
    if reason is not None:
        reason = str(reason).strip()
    pred_label = None
    if risk is not None:
        pred_label = 1 if risk == "yes" else 0
    elif risk_score is not None:
        pred_label = 1 if risk_score >= 3 else 0
    parsed_ok = (risk is not None) or (risk_score is not None)
    return {
        "parsed_ok": parsed_ok,
        "risk": risk,
        "risk_score": risk_score,
        "reason": reason,
        "pred_label": pred_label,
    }


def build_pipeline(args: argparse.Namespace):
    model_kwargs: Dict[str, Any] = {}
    pipeline_kwargs: Dict[str, Any] = {}
    dtype = dtype_from_str(args.dtype)
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit:
        bnb_kwargs: Dict[str, Any] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
        if dtype != "auto":
            bnb_kwargs["bnb_4bit_compute_dtype"] = dtype
        model_kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)
    if args.offload_dir:
        offload_dir = Path(args.offload_dir).resolve()
        offload_dir.mkdir(parents=True, exist_ok=True)
        model_kwargs["offload_folder"] = str(offload_dir)
    if args.cache_dir:
        cache_dir = Path(args.cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
        pipeline_kwargs["cache_dir"] = str(cache_dir)

    model_name = resolve_model_name_or_path(args.model_name, args.cache_dir, bool(args.local_files_only))

    return pipeline(
        task="image-text-to-text",
        model=model_name,
        device_map=args.device_map,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
        model_kwargs=model_kwargs,
        **pipeline_kwargs,
    )


def resolve_model_name_or_path(model_name: str, cache_dir: Optional[str], local_files_only: bool) -> str:
    candidate = Path(str(model_name)).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    if (not local_files_only) or (not cache_dir) or ("/" not in str(model_name)):
        return model_name

    model_dir = Path(cache_dir).resolve() / f"models--{str(model_name).replace('/', '--')}"
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return model_name

    ref_main = model_dir / "refs" / "main"
    if ref_main.exists():
        snapshot = snapshots_dir / ref_main.read_text(encoding="utf-8").strip()
        if snapshot.exists():
            return str(snapshot.resolve())

    snapshots = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if snapshots:
        return str(snapshots[-1].resolve())
    return model_name


def describe_pipeline_device(pipe) -> str:
    model = getattr(pipe, "model", None)
    if model is None:
        return "unknown"
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        return str(hf_device_map)
    device = getattr(model, "device", None)
    if device is not None:
        return str(device)
    return "unknown"


def make_logger(log_file: Optional[str]):
    handle: Optional[TextIO] = None
    if log_file:
        path = Path(log_file).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a", encoding="utf-8")

    def _emit(message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        if handle is not None:
            handle.write(line + "\n")
            handle.flush()

    return _emit, handle


def maybe_clear_cuda_cache(enabled: bool) -> None:
    if not enabled:
        return
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def is_fatal_cuda_error_message(message: str) -> bool:
    message = str(message).lower()
    return ("cuda" in message) or ("out of memory" in message)


def main() -> None:
    args = parse_args()
    log, log_handle = make_logger(args.log_file)
    prompt = load_prompt(args.prompt_file)
    manifest_path = Path(args.manifest).resolve()
    manifest_root = Path(args.data_root).resolve() if args.data_root else infer_manifest_root(manifest_path)
    df = pd.read_csv(manifest_path)
    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    if args.skip_existing and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    seen.add(json.loads(line)["variant_id"])
                except Exception:
                    pass

    rows = [row for row in df.to_dict("records") if str(row["variant_id"]) not in seen]
    log(
        f"Run config manifest={manifest_path} total_manifest={len(df)} "
        f"resume_seen={len(seen)} remaining={len(rows)} output_jsonl={out_path}"
    )
    if torch.cuda.is_available():
        log(f"CUDA available: True ({torch.cuda.get_device_name(0)})")
    else:
        log("CUDA available: False")
    log(f"Requested device_map: {args.device_map}")
    log(f"Model source: {resolve_model_name_or_path(args.model_name, args.cache_dir, bool(args.local_files_only))}")
    pipe = build_pipeline(args)
    log(f"Resolved model placement: {describe_pipeline_device(pipe)}")

    ok_count = 0
    err_count = 0
    run_start = time.time()
    try:
        with open(out_path, "a", encoding="utf-8") as f:
            for idx, row in enumerate(tqdm(rows, total=len(rows), desc=f"Infer {args.model_name}"), start=1):
                variant_id = str(row["variant_id"])
                resolved_image = resolve_manifest_path(row.get("generated_image_path"), manifest_root)
                image_path = str(resolved_image) if resolved_image is not None else str(row.get("generated_image_path"))
                rec: Dict[str, Any] = {
                    "model_name": args.model_name,
                    "variant_id": variant_id,
                    "base_id": row.get("base_id"),
                    "image_path": image_path,
                    "overlay_type": row.get("overlay_type"),
                    "severity": int(row.get("severity", 0)),
                    "placement_mode": row.get("placement_mode"),
                }
                item_start = time.time()
                log(f"START {idx}/{len(rows)} variant_id={variant_id}")
                try:
                    with Image.open(image_path) as image:
                        image = image.convert("RGB")
                        messages = [{
                            "role": "user",
                            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
                        }]
                        outputs = pipe(
                            text=messages,
                            max_new_tokens=args.max_new_tokens,
                            return_full_text=False,
                            do_sample=False,
                        )
                    generated_text = outputs[0].get("generated_text", "") if isinstance(outputs, list) else str(outputs)
                    if isinstance(generated_text, list):
                        generated_text = json.dumps(generated_text, ensure_ascii=False)
                    parsed = extract_prediction(str(generated_text))
                    rec.update({"status": "ok", "raw_output": generated_text, **normalize(parsed)})
                    ok_count += 1
                except Exception as e:
                    rec.update({
                        "status": "error",
                        "raw_output": None,
                        "parsed_ok": False,
                        "risk": None,
                        "risk_score": None,
                        "reason": None,
                        "pred_label": None,
                        "error": str(e),
                    })
                    err_count += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

                elapsed_item = time.time() - item_start
                elapsed_total = time.time() - run_start
                avg_sec = elapsed_total / idx if idx else 0.0
                eta_sec = avg_sec * (len(rows) - idx)
                log(
                    f"DONE {idx}/{len(rows)} variant_id={variant_id} status={rec['status']} "
                    f"parsed_ok={rec.get('parsed_ok')} risk={rec.get('risk')} pred_label={rec.get('pred_label')} "
                    f"item_sec={elapsed_item:.1f} eta_sec={eta_sec:.1f} ok={ok_count} err={err_count}"
                )
                maybe_clear_cuda_cache(args.clear_cuda_cache)
                if rec["status"] == "error" and is_fatal_cuda_error_message(str(rec.get("error", ""))):
                    log(
                        f"FATAL CUDA ERROR at {idx}/{len(rows)} variant_id={variant_id}; "
                        "stopping run early so it can be resumed safely with --skip-existing"
                    )
                    break
    finally:
        log(f"Saved predictions to: {out_path}")
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    main()
