#!/usr/bin/env python3
"""Build a safety-attack benchmark from a DADA TTE frame manifest.

This script focuses on *overlay-based evasion attacks*:
clean positive frame  -> model predicts imminent accident
overlay applied       -> model predicts no imminent accident

Recommended input:
    frame_manifest_tte.csv from 03_build_tte_manifest_from_dada_xlsx_lotvs.py
Optional:
    ROI columns roi_x1..roi_y2 from 03b_add_attention_roi_from_maps.py

Output:
    manifest_attack.csv with clean + perturbed variants
    generated images under output_dir/images/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps
from tqdm import tqdm

from _manifest_paths import infer_manifest_root, resolve_manifest_path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ATTACK_OVERLAYS = [
    "text_watermark",
    "timestamp_box",
    "logo_patch",
    "privacy_mosaic",
    "privacy_blur",
    "semi_transparent_bar",
]
STRESSOR_OVERLAYS = [
    "windshield_reflection",
    "dirt_splash",
    "raindrops",
]
OVERLAY_GROUPS = {
    "attack": ATTACK_OVERLAYS,
    "stressor": STRESSOR_OVERLAYS,
    "all": ATTACK_OVERLAYS + STRESSOR_OVERLAYS,
}
PLACEMENT_CHOICES = ["random", "critical", "background"]


@dataclass
class Record:
    image_path: str
    metadata: Dict[str, object]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-root", default=None, help="Optional root used to resolve relative image/map paths")
    p.add_argument("--overlay-group", choices=["attack", "stressor", "all"], default="attack")
    p.add_argument("--overlay-types", nargs="*", default=None, help="Explicit list; overrides --overlay-group")
    p.add_argument("--placement-modes", nargs="+", choices=PLACEMENT_CHOICES, default=["random", "critical", "background"])
    p.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--variants-per-setting", type=int, default=1)
    p.add_argument("--include-clean", action="store_true")
    p.add_argument("--labeled-only", action="store_true")
    p.add_argument("--positive-only", action="store_true")
    p.add_argument("--save-ext", default=".png", choices=[".png", ".jpg", ".jpeg"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def pil_to_rgba(img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(img).convert("RGBA")


def save_rgb(img: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = img.convert("RGB")
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        rgb.save(out_path, quality=95)
    else:
        rgb.save(out_path)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def severity_scale(severity: int) -> float:
    mapping = {1: 0.18, 2: 0.35, 3: 0.52, 4: 0.72, 5: 0.95}
    severity = int(severity)
    if severity not in mapping:
        raise ValueError(f"severity must be 1..5, got {severity}")
    return mapping[severity]


def rgba(r: int, g: int, b: int, a: int) -> Tuple[int, int, int, int]:
    return int(r), int(g), int(b), int(a)


def overlay_image(base: Image.Image, top: Image.Image) -> Image.Image:
    if base.mode != "RGBA":
        base = base.convert("RGBA")
    if top.mode != "RGBA":
        top = top.convert("RGBA")
    return Image.alpha_composite(base, top)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            try:
                return ImageFont.truetype(cand, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def rect_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(area_a + area_b - inter)


def box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def get_roi_bbox(row: Dict[str, object], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    keys = ["roi_x1", "roi_y1", "roi_x2", "roi_y2"]
    if not all(k in row for k in keys):
        return None
    try:
        x1 = int(float(row["roi_x1"]))
        y1 = int(float(row["roi_y1"]))
        x2 = int(float(row["roi_x2"]))
        y2 = int(float(row["roi_y2"]))
    except Exception:
        return None
    x1 = int(clamp(x1, 0, w - 1))
    y1 = int(clamp(y1, 0, h - 1))
    x2 = int(clamp(x2, x1 + 1, w))
    y2 = int(clamp(y2, y1 + 1, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def random_bbox(
    w: int,
    h: int,
    rng: random.Random,
    area_ratio_range: Tuple[float, float] = (0.03, 0.12),
    aspect_range: Tuple[float, float] = (0.7, 2.0),
) -> Tuple[int, int, int, int]:
    area_ratio = rng.uniform(*area_ratio_range)
    aspect = rng.uniform(*aspect_range)
    area = area_ratio * w * h
    bw = int(max(8, min(w - 1, round(math.sqrt(area * aspect)))))
    bh = int(max(8, min(h - 1, round(math.sqrt(area / aspect)))))
    x1 = rng.randint(0, max(0, w - bw))
    y1 = rng.randint(0, max(0, h - bh))
    return x1, y1, x1 + bw, y1 + bh


def expand_bbox(
    box: Tuple[int, int, int, int], w: int, h: int, expand_ratio: float
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    ex, ey = int(bw * expand_ratio), int(bh * expand_ratio)
    return max(0, x1 - ex), max(0, y1 - ey), min(w, x2 + ex), min(h, y2 + ey)


def choose_local_box(
    row: Dict[str, object],
    w: int,
    h: int,
    rng: random.Random,
    scale: float,
    placement_mode: str,
) -> Tuple[Tuple[int, int, int, int], Dict[str, object]]:
    roi = get_roi_bbox(row, w, h)
    desired = (0.03 + 0.10 * scale, 0.07 + 0.18 * scale)
    if placement_mode == "critical":
        if roi is not None:
            box = expand_bbox(roi, w, h, 0.15 + 0.35 * scale)
            return box, {"roi_available": True, "target": "roi"}
        # fallback: center box
        bw = int(w * (0.16 + 0.14 * scale))
        bh = int(h * (0.12 + 0.12 * scale))
        x1 = (w - bw) // 2
        y1 = (h - bh) // 2
        return (x1, y1, x1 + bw, y1 + bh), {"roi_available": False, "target": "center_fallback"}

    if placement_mode == "background":
        if roi is None:
            return random_bbox(w, h, rng, desired), {"roi_available": False, "target": "random_no_roi"}
        best = None
        best_score = -1e9
        roi_cx, roi_cy = box_center(roi)
        for _ in range(80):
            cand = random_bbox(w, h, rng, desired)
            iou = rect_iou(cand, roi)
            cx, cy = box_center(cand)
            dist = math.hypot(cx - roi_cx, cy - roi_cy)
            score = dist - 4000.0 * iou
            if score > best_score:
                best_score = score
                best = cand
        assert best is not None
        return best, {"roi_available": True, "target": "background_away_from_roi", "roi_iou": rect_iou(best, roi)}

    # random
    return random_bbox(w, h, rng, desired), {"roi_available": roi is not None, "target": "random"}


def choose_patch_xy(
    patch_w: int,
    patch_h: int,
    w: int,
    h: int,
    rng: random.Random,
    placement_mode: str,
    roi: Optional[Tuple[int, int, int, int]],
) -> Tuple[int, int, Dict[str, object]]:
    def clamp_xy(x: float, y: float) -> Tuple[int, int]:
        xx = int(clamp(round(x), 0, max(0, w - patch_w)))
        yy = int(clamp(round(y), 0, max(0, h - patch_h)))
        return xx, yy

    if placement_mode == "critical":
        if roi is not None:
            cx, cy = box_center(roi)
            return *clamp_xy(cx - patch_w / 2.0, cy - patch_h / 2.0), {"roi_available": True, "target": "roi_center"}
        return *clamp_xy((w - patch_w) / 2.0, (h - patch_h) / 2.0), {"roi_available": False, "target": "center_fallback"}

    if placement_mode == "background":
        corners = [
            (0.02 * w, 0.02 * h, "tl"),
            (w - patch_w - 0.02 * w, 0.02 * h, "tr"),
            (0.02 * w, h - patch_h - 0.02 * h, "bl"),
            (w - patch_w - 0.02 * w, h - patch_h - 0.02 * h, "br"),
        ]
        if roi is None:
            x, y, name = random.choice(corners)
            xx, yy = clamp_xy(x, y)
            return xx, yy, {"roi_available": False, "target": f"corner_{name}"}
        roi_cx, roi_cy = box_center(roi)
        best = None
        best_dist = -1e9
        for x, y, name in corners:
            xx, yy = clamp_xy(x, y)
            patch_box = (xx, yy, xx + patch_w, yy + patch_h)
            cx, cy = box_center(patch_box)
            dist = math.hypot(cx - roi_cx, cy - roi_cy) - 4000.0 * rect_iou(patch_box, roi)
            if dist > best_dist:
                best_dist = dist
                best = (xx, yy, name)
        assert best is not None
        xx, yy, name = best
        return xx, yy, {"roi_available": True, "target": f"corner_{name}"}

    # random
    xx = rng.randint(0, max(0, w - patch_w))
    yy = rng.randint(0, max(0, h - patch_h))
    return xx, yy, {"roi_available": roi is not None, "target": "random"}


def make_text_patch(text: str, font_size: int, alpha: int, rng: random.Random) -> Image.Image:
    font = load_font(font_size)
    bbox = ImageDraw.Draw(Image.new("RGBA", (10, 10))).textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(6, int(font_size * 0.35))
    patch = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(patch)
    shadow = rgba(0, 0, 0, int(alpha * 0.35))
    fg = rgba(255, 255, 255, alpha)
    d.text((pad + 2, pad + 2), text, font=font, fill=shadow)
    d.text((pad, pad), text, font=font, fill=fg)
    angle = rng.uniform(-28, 28)
    patch = patch.rotate(angle, expand=True, resample=Image.BICUBIC)
    patch = patch.filter(ImageFilter.GaussianBlur(radius=max(0.0, font_size * 0.01)))
    return patch


def add_text_watermark(base: Image.Image, severity: int, rng: random.Random, placement_mode: str, row: Dict[str, object]):
    img = base.copy()
    w, h = img.size
    scale = severity_scale(severity)
    roi = get_roi_bbox(row, w, h)
    texts = ["DASHCAM", "REC", "SAMPLE", "TRAFFIC MONITOR", "ROAD VIEW"]
    text = rng.choice(texts)
    font_size = int(min(w, h) * (0.045 + 0.08 * scale))
    alpha = int(45 + 125 * scale)
    patch = make_text_patch(text, font_size, alpha, rng)
    x, y, meta = choose_patch_xy(patch.width, patch.height, w, h, rng, placement_mode, roi)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    overlay.alpha_composite(patch, dest=(x, y))
    img = overlay_image(img, overlay)
    return img, {"text": text, "alpha": alpha, "bbox": [x, y, x + patch.width, y + patch.height], **meta}


def add_timestamp_box(base: Image.Image, severity: int, rng: random.Random, placement_mode: str, row: Dict[str, object]):
    img = base.copy()
    w, h = img.size
    scale = severity_scale(severity)
    roi = get_roi_bbox(row, w, h)
    box_h = int(h * (0.05 + 0.04 * scale))
    box_w = int(w * (0.22 + 0.18 * scale))
    patch = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(patch)
    bg = rgba(20, 20, 20, int(110 + 95 * scale))
    d.rounded_rectangle((0, 0, box_w - 1, box_h - 1), radius=max(4, box_h // 6), fill=bg)
    font = load_font(max(12, int(box_h * 0.40)))
    year = rng.randint(2023, 2026)
    txt = f"{year:04d}/09/08  17:{rng.randint(10,59):02d}:{rng.randint(10,59):02d}  {rng.randint(0,120)}km/h"
    d.text((int(box_h * 0.18) + 1, int(box_h * 0.18) + 1), txt, font=font, fill=rgba(0,0,0,180))
    d.text((int(box_h * 0.18), int(box_h * 0.18)), txt, font=font, fill=rgba(255,255,255,235))
    x, y, meta = choose_patch_xy(box_w, box_h, w, h, rng, placement_mode, roi)
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    overlay.alpha_composite(patch, dest=(x, y))
    img = overlay_image(img, overlay)
    return img, {"text": txt, "bbox": [x, y, x + box_w, y + box_h], "alpha": bg[3], **meta}


def add_logo_patch(base: Image.Image, severity: int, rng: random.Random, placement_mode: str, row: Dict[str, object]):
    img = base.copy()
    w, h = img.size
    scale = severity_scale(severity)
    roi = get_roi_bbox(row, w, h)
    patch_w = int(w * (0.07 + 0.08 * scale))
    patch_h = int(patch_w * 0.58)
    patch = Image.new("RGBA", (patch_w, patch_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(patch)
    bg = random.choice([
        rgba(255, 50, 50, int(85 + 120 * scale)),
        rgba(30, 30, 30, int(90 + 140 * scale)),
        rgba(250, 250, 250, int(75 + 110 * scale)),
    ])
    d.rounded_rectangle((0, 0, patch_w - 1, patch_h - 1), radius=max(4, patch_h // 6), fill=bg)
    font = load_font(max(11, int(patch_h * 0.48)))
    label = rng.choice(["REC", "HD", "CAM", "AI"])
    d.ellipse((int(patch_h * 0.15), int(patch_h * 0.16), int(patch_h * 0.43), int(patch_h * 0.44)), fill=rgba(255,255,255,220))
    d.text((int(patch_h * 0.50), int(patch_h * 0.12)), label, font=font, fill=rgba(255,255,255,235))
    x, y, meta = choose_patch_xy(patch_w, patch_h, w, h, rng, placement_mode, roi)
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    overlay.alpha_composite(patch, dest=(x, y))
    img = overlay_image(img, overlay)
    return img, {"label": label, "bbox": [x, y, x + patch_w, y + patch_h], "alpha": bg[3], **meta}


def mosaic_region(region: Image.Image, block: int) -> Image.Image:
    w, h = region.size
    ds_w = max(1, w // max(1, block))
    ds_h = max(1, h // max(1, block))
    return region.resize((ds_w, ds_h), resample=Image.BILINEAR).resize((w, h), resample=Image.NEAREST)


def add_privacy_mosaic(base: Image.Image, severity: int, rng: random.Random, placement_mode: str, row: Dict[str, object]):
    img = base.copy()
    w, h = img.size
    scale = severity_scale(severity)
    box, meta = choose_local_box(row, w, h, rng, scale, placement_mode)
    x1, y1, x2, y2 = box
    block = int(6 + 28 * scale)
    region = img.crop(box).convert("RGB")
    mos = mosaic_region(region, block)
    img.paste(mos, (x1, y1))
    return img, {"bbox": [x1, y1, x2, y2], "block": block, **meta}


def add_privacy_blur(base: Image.Image, severity: int, rng: random.Random, placement_mode: str, row: Dict[str, object]):
    img = base.copy()
    w, h = img.size
    scale = severity_scale(severity)
    box, meta = choose_local_box(row, w, h, rng, scale, placement_mode)
    x1, y1, x2, y2 = box
    radius = 2.5 + 10.0 * scale
    region = img.crop(box).filter(ImageFilter.GaussianBlur(radius=radius))
    img.paste(region, (x1, y1))
    return img, {"bbox": [x1, y1, x2, y2], "radius": round(radius, 2), **meta}


def add_semi_transparent_bar(base: Image.Image, severity: int, rng: random.Random, placement_mode: str, row: Dict[str, object]):
    img = base.copy()
    w, h = img.size
    scale = severity_scale(severity)
    roi = get_roi_bbox(row, w, h)
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))

    bar_len = int(max(w, h) * rng.uniform(0.75, 1.20))
    bar_thick = int(min(w, h) * rng.uniform(0.03, 0.07 + 0.05 * scale))
    alpha = int(35 + 110 * scale)
    color = rng.choice([rgba(230,230,230,alpha), rgba(40,40,40,alpha), rgba(90,90,90,alpha)])
    patch = Image.new("RGBA", (bar_len, bar_thick), (0,0,0,0))
    d = ImageDraw.Draw(patch)
    d.rounded_rectangle((0,0,bar_len-1,bar_thick-1), radius=max(2, bar_thick//3), fill=color)
    angle = rng.uniform(-35, 35)
    patch = patch.rotate(angle, expand=True, resample=Image.BICUBIC).filter(ImageFilter.GaussianBlur(radius=1.0 + 2.0 * scale))

    x, y, meta = choose_patch_xy(patch.width, patch.height, w, h, rng, placement_mode, roi)
    overlay.alpha_composite(patch, dest=(x, y))
    img = overlay_image(img, overlay)
    return img, {"bbox": [x, y, x + patch.width, y + patch.height], "angle": round(angle, 2), "alpha": alpha, **meta}


def make_soft_ellipse_canvas(w: int, h: int, cx: float, cy: float, rx: float, ry: float, color: Tuple[int,int,int], alpha: int) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w]
    d = ((xx - cx) / max(1e-6, rx)) ** 2 + ((yy - cy) / max(1e-6, ry)) ** 2
    mask = np.exp(-d * 2.2)
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., 0] = color[0]
    arr[..., 1] = color[1]
    arr[..., 2] = color[2]
    arr[..., 3] = np.clip(mask * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def add_windshield_reflection(base: Image.Image, severity: int, rng: random.Random, placement_mode: str, row: Dict[str, object]):
    img = base.copy()
    w, h = img.size
    scale = severity_scale(severity)
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    params = []
    for _ in range(2 if severity >= 3 else 1):
        cx = rng.uniform(0.25 * w, 0.75 * w)
        cy = rng.uniform(0.55 * h, 0.95 * h)
        rx = rng.uniform(0.18 * w, 0.40 * w)
        ry = rng.uniform(0.08 * h, 0.22 * h)
        alpha = int(25 + 90 * scale)
        color = rng.choice([(230,230,230), (255,245,210), (200,210,230)])
        ell = make_soft_ellipse_canvas(w, h, cx, cy, rx, ry, color=color, alpha=alpha)
        overlay = overlay_image(overlay, ell)
        params.append({"cx": round(cx,1), "cy": round(cy,1), "rx": round(rx,1), "ry": round(ry,1), "alpha": alpha})
    return overlay_image(img, overlay), {"ellipses": params, "placement_mode_effective": "random"}


def add_raindrops(base: Image.Image, severity: int, rng: random.Random, placement_mode: str, row: Dict[str, object]):
    img = base.copy()
    w, h = img.size
    scale = severity_scale(severity)
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    n = int(8 + 35 * scale)
    params = []
    for _ in range(n):
        rx = rng.uniform(4, 12 + 18 * scale)
        ry = rx * rng.uniform(1.1, 1.8)
        x = rng.uniform(0, w)
        y = rng.uniform(0, h)
        alpha = int(25 + 70 * scale)
        d.ellipse((x-rx, y-ry, x+rx, y+ry), outline=rgba(220,220,235,alpha), width=max(1, int(rx*0.18)))
        d.ellipse((x-rx*0.25, y-ry*0.35, x+rx*0.12, y-ry*0.08), fill=rgba(255,255,255,int(alpha*0.55)))
        params.append({"x": round(x,1), "y": round(y,1), "rx": round(rx,1)})
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.3 + 1.4 * scale))
    return overlay_image(img, overlay), {"drops": params[:10], "num_drops": n, "placement_mode_effective": "random"}


def random_blob_points(cx: float, cy: float, radius: float, rng: random.Random, count: int = 16):
    pts = []
    for i in range(count):
        ang = 2.0 * math.pi * i / count + rng.uniform(-0.15, 0.15)
        rr = radius * rng.uniform(0.55, 1.25)
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    return pts


def add_dirt_splash(base: Image.Image, severity: int, rng: random.Random, placement_mode: str, row: Dict[str, object]):
    img = base.copy()
    w, h = img.size
    scale = severity_scale(severity)
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    n = int(8 + 22 * scale)
    for _ in range(n):
        cx = rng.uniform(0, w)
        cy = rng.uniform(0.15 * h, h)
        radius = rng.uniform(5, 12 + 24 * scale)
        color = rng.choice([rgba(100,76,48,int(55 + 120 * scale)), rgba(78,58,35,int(50 + 110 * scale))])
        d.polygon(random_blob_points(cx, cy, radius, rng), fill=color)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.2 + 1.2 * scale))
    return overlay_image(img, overlay), {"num_blobs": n, "placement_mode_effective": "random"}


OVERLAY_FUNCS = {
    "text_watermark": add_text_watermark,
    "timestamp_box": add_timestamp_box,
    "logo_patch": add_logo_patch,
    "privacy_mosaic": add_privacy_mosaic,
    "privacy_blur": add_privacy_blur,
    "semi_transparent_bar": add_semi_transparent_bar,
    "windshield_reflection": add_windshield_reflection,
    "dirt_splash": add_dirt_splash,
    "raindrops": add_raindrops,
}


def load_records(input_csv: Path, data_root: Optional[Path], labeled_only: bool, positive_only: bool, limit: Optional[int]) -> List[Record]:
    df = pd.read_csv(input_csv)
    if labeled_only and "label_risk" in df.columns:
        df = df[~df["label_risk"].isna()].copy()
    if positive_only and "label_risk" in df.columns:
        df = df[pd.to_numeric(df["label_risk"], errors="coerce") == 1].copy()
    if limit is not None:
        df = df.head(limit).copy()
    if "image_path" not in df.columns:
        raise ValueError("input csv must contain image_path column")
    rows = []
    for r in df.to_dict(orient="records"):
        image_path = resolve_manifest_path(r["image_path"], data_root)
        if image_path is None:
            continue
        rows.append(Record(image_path=str(image_path), metadata=dict(r)))
    return rows


def resolve_overlay_list(args: argparse.Namespace) -> List[str]:
    overlays = args.overlay_types if args.overlay_types else OVERLAY_GROUPS[args.overlay_group]
    unknown = [o for o in overlays if o not in OVERLAY_FUNCS]
    if unknown:
        raise ValueError(f"Unknown overlay types: {unknown}")
    return list(overlays)


def base_id_for_row(row: Dict[str, object]) -> str:
    split = str(row.get("split", "na"))
    category = str(row.get("category", "na"))
    clip_id = str(row.get("clip_id", "na"))
    frame_idx = int(float(row.get("frame_idx", 0)))
    return f"{split}_{category}_{clip_id}_{frame_idx:04d}"


def clip_key_for_row(row: Dict[str, object]) -> Optional[str]:
    if row.get("clip_key") not in (None, ""):
        return str(row.get("clip_key"))
    category = row.get("category")
    clip_id = row.get("clip_id")
    if category is None or clip_id is None:
        return None
    try:
        return f"{int(float(category))}_{int(float(clip_id)):03d}"
    except Exception:
        return f"{category}_{clip_id}"


def output_subdir(output_dir: Path, overlay_type: str, severity: int, placement_mode: str, base_id: str) -> Path:
    return output_dir / "images" / overlay_type / f"s{severity}" / placement_mode / base_id


def make_variant_id(base_id: str, overlay_type: str, severity: int, placement_mode: str, variant_idx: int) -> str:
    return f"{base_id}__{overlay_type}__s{severity}__p{placement_mode}__v{variant_idx}"


def main() -> None:
    args = parse_args()
    overlays = resolve_overlay_list(args)
    input_csv = Path(args.input_csv).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else infer_manifest_root(input_csv)
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    records = load_records(input_csv, data_root=data_root, labeled_only=args.labeled_only, positive_only=args.positive_only, limit=args.limit)
    manifest_rows: List[Dict[str, object]] = []

    for idx, rec in enumerate(tqdm(records, desc="Build safety attack benchmark")):
        src = Path(rec.image_path)
        if not src.exists():
            print(f"[WARN] Missing image: {src}")
            continue
        base_img = pil_to_rgba(Image.open(src))
        base_row = dict(rec.metadata)
        base_id = base_id_for_row(base_row)
        base_row["base_id"] = base_id
        base_row["clip_key"] = clip_key_for_row(base_row)
        base_row["source_image_path"] = str(src.resolve())

        if args.include_clean:
            clean_out_dir = out_root / "images" / "clean" / "s0" / "clean" / base_id
            clean_out_dir.mkdir(parents=True, exist_ok=True)
            clean_path = clean_out_dir / f"{base_id}__clean{args.save_ext}"
            save_rgb(base_img, clean_path)
            row = dict(base_row)
            row.update({
                "generated_image_path": str(clean_path),
                "variant_id": f"{base_id}__clean",
                "overlay_type": "clean",
                "overlay_group": "clean",
                "placement_mode": "clean",
                "severity": 0,
                "variant_idx": 0,
                "overlay_params": json.dumps({}, ensure_ascii=False),
            })
            manifest_rows.append(row)

        for overlay_type in overlays:
            overlay_group = "attack" if overlay_type in ATTACK_OVERLAYS else "stressor"
            placement_modes = list(args.placement_modes)
            # stressors ignore placement; still keep a single random mode to simplify analysis
            if overlay_group == "stressor":
                placement_modes = ["random"]
            for severity in args.severities:
                for placement_mode in placement_modes:
                    for variant_idx in range(1, args.variants_per_setting + 1):
                        seed = (args.seed * 1000003) ^ (idx * 10007) ^ (int(severity) * 101) ^ (variant_idx * 271) ^ sum(ord(c) for c in overlay_type + placement_mode)
                        rng = random.Random(seed)
                        try:
                            aug, params = OVERLAY_FUNCS[overlay_type](base_img, int(severity), rng, placement_mode, base_row)
                        except Exception as e:
                            print(f"[WARN] Failed on {src.name} {overlay_type} s{severity} {placement_mode}: {e}")
                            continue
                        vid = make_variant_id(base_id, overlay_type, int(severity), placement_mode, variant_idx)
                        out_dir = output_subdir(out_root, overlay_type, int(severity), placement_mode, base_id)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"{vid}{args.save_ext}"
                        save_rgb(aug, out_path)

                        row = dict(base_row)
                        row.update({
                            "generated_image_path": str(out_path),
                            "variant_id": vid,
                            "overlay_type": overlay_type,
                            "overlay_group": overlay_group,
                            "placement_mode": placement_mode,
                            "severity": int(severity),
                            "variant_idx": variant_idx,
                            "overlay_params": json.dumps(params, ensure_ascii=False),
                        })
                        manifest_rows.append(row)

    out_df = pd.DataFrame(manifest_rows)
    manifest_path = out_root / "manifest_attack.csv"
    out_df.to_csv(manifest_path, index=False)
    summary = {
        "input_csv": str(input_csv),
        "data_root": str(data_root),
        "output_dir": str(out_root),
        "num_input_frames": len(records),
        "num_output_rows": int(len(out_df)),
        "overlay_types": overlays,
        "placement_modes": list(args.placement_modes),
        "severities": [int(s) for s in args.severities],
        "variants_per_setting": int(args.variants_per_setting),
        "include_clean": bool(args.include_clean),
        "labeled_only": bool(args.labeled_only),
        "positive_only": bool(args.positive_only),
        "manifest_csv": str(manifest_path),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
