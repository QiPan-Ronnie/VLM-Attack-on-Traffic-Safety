#!/usr/bin/env python3
"""Run an open_clip zero-shot baseline for imminent-accident safety attack evaluation."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import open_clip
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "clip_attack_prompts.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--clip-model", required=True)
    p.add_argument("--clip-pretrained", default=None)
    p.add_argument("--custom-checkpoint", default=None)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=None, help="Inference batch size. Defaults to 64 on CUDA and 1 on CPU.")
    p.add_argument("--log-file", default=None, help="Optional batch progress log written for tailing")
    return p.parse_args()


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prompts = cfg.get("clip_prompts", {}).get("risk", {})
    normalized = {}
    for key, value in prompts.items():
        if key is True:
            normalized["yes"] = value
        elif key is False:
            normalized["no"] = value
        else:
            normalized[str(key)] = value
    cfg["clip_prompts"]["risk"] = normalized
    return cfg


def load_model(args: argparse.Namespace, device: str):
    if args.custom_checkpoint:
        model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=None)
        state = torch.load(args.custom_checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    else:
        if args.clip_pretrained is None:
            raise ValueError("Either --clip-pretrained or --custom-checkpoint must be provided")
        model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrained)
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    model.to(device).eval()
    return model, preprocess, tokenizer


def resolve_device(device: str) -> str:
    device = str(device).strip()
    if device == "cuda":
        device = "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def infer_batch_size(device: str, requested: int | None) -> int:
    if requested is not None:
        return max(1, int(requested))
    return 64 if device.startswith("cuda") else 1


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


@torch.no_grad()
def build_text_embeddings(model, tokenizer, prompts_by_label: Dict[str, List[str]], device: str):
    labels = list(prompts_by_label.keys())
    vectors = []
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.startswith("cuda") else nullcontext()
    for label in labels:
        texts = prompts_by_label[label]
        tokens = tokenizer(texts).to(device)
        with autocast_ctx:
            feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vec = feats.mean(dim=0, keepdim=True)
        vec = vec / vec.norm(dim=-1, keepdim=True)
        vectors.append(vec)
    return labels, torch.cat(vectors, dim=0)


@torch.no_grad()
def predict_batch(model, preprocess, image_paths: List[str], risk_labels, risk_matrix, device: str):
    tensors = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            tensors.append(preprocess(image.convert("RGB")))
    x = torch.stack(tensors, dim=0).to(device, non_blocking=device.startswith("cuda"))
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device.startswith("cuda") else nullcontext()
    with autocast_ctx:
        image_feat = model.encode_image(x)
    image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

    risk_logits = 100.0 * image_feat @ risk_matrix.T
    risk_probs = torch.softmax(risk_logits, dim=-1)
    yes_index = risk_labels.index("yes")
    outputs = []
    for row_probs in risk_probs:
        risk_idx = int(torch.argmax(row_probs))
        risk_label = risk_labels[risk_idx]
        yes_prob = float(row_probs[yes_index].item())
        risk_score = int(round(1 + 4 * yes_prob))
        risk_score = max(1, min(5, risk_score))
        outputs.append({
            "risk": risk_label,
            "risk_score": risk_score,
            "reason": None,
            "pred_label": 1 if risk_label == "yes" else 0,
            "risk_yes_prob": yes_prob,
            "risk_probs": {label: float(prob.item()) for label, prob in zip(risk_labels, row_probs)},
        })
    return outputs


def main() -> None:
    args = parse_args()
    log, log_handle = make_logger(args.log_file)
    cfg = load_cfg(args.config)
    df = pd.read_csv(args.manifest)
    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()

    device = resolve_device(args.device)
    batch_size = infer_batch_size(device, args.batch_size)
    log(f"Using device: {device}")
    if device.startswith("cuda"):
        log(f"CUDA device: {torch.cuda.get_device_name(torch.device(device))}")
    log(f"Batch size: {batch_size}")

    model, preprocess, tokenizer = load_model(args, device)
    risk_labels, risk_matrix = build_text_embeddings(model, tokenizer, cfg["clip_prompts"]["risk"], device)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_id = args.custom_checkpoint if args.custom_checkpoint else f"{args.clip_model}:{args.clip_pretrained}"

    rows = df.to_dict("records")
    total_batches = (len(rows) + batch_size - 1) // batch_size if rows else 0
    log(f"Run config manifest={args.manifest} total_rows={len(rows)} total_batches={total_batches} output_jsonl={out_path}")

    ok_count = 0
    err_count = 0
    run_start = time.time()
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for batch_idx, start in enumerate(
                tqdm(range(0, len(rows), batch_size), total=total_batches, desc="CLIP safety-attack baseline"),
                start=1,
            ):
                batch_rows = rows[start:start + batch_size]
                image_paths = [str(row["generated_image_path"]) for row in batch_rows]
                batch_start = time.time()
                log(f"START batch={batch_idx}/{total_batches} rows={len(batch_rows)} first_variant={batch_rows[0]['variant_id']}")
                try:
                    preds = predict_batch(model, preprocess, image_paths, risk_labels, risk_matrix, device)
                except Exception as batch_error:
                    preds = [batch_error] * len(batch_rows)

                for row, pred_or_error, image_path in zip(batch_rows, preds, image_paths):
                    rec: Dict[str, Any] = {
                        "model_name": model_id,
                        "variant_id": row["variant_id"],
                        "base_id": row.get("base_id"),
                        "image_path": image_path,
                        "overlay_type": row.get("overlay_type"),
                        "severity": int(row.get("severity", 0)),
                        "placement_mode": row.get("placement_mode"),
                    }
                    if isinstance(pred_or_error, Exception):
                        rec.update({
                            "status": "error",
                            "parsed_ok": False,
                            "risk": None,
                            "risk_score": None,
                            "reason": None,
                            "pred_label": None,
                            "error": str(pred_or_error),
                        })
                        err_count += 1
                    else:
                        rec.update({"status": "ok", "parsed_ok": True, **pred_or_error})
                        ok_count += 1
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

                elapsed_total = time.time() - run_start
                avg_batch_sec = elapsed_total / batch_idx if batch_idx else 0.0
                eta_sec = avg_batch_sec * (total_batches - batch_idx)
                log(
                    f"DONE batch={batch_idx}/{total_batches} item_sec={time.time() - batch_start:.1f} "
                    f"eta_sec={eta_sec:.1f} ok={ok_count} err={err_count}"
                )
    finally:
        log(f"Saved predictions to: {out_path}")
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    main()
