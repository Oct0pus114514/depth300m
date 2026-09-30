"""Inference: student/teacher depth at short-side process_res, native aspect.

Usage:
  python infer_model.py --ckpt runs/s1/last.pt --ema --images <dir|file> --out <dir>
  python infer_model.py --baseline da3base|da3mono --images ... --out ...
Saves: <stem>.npy (fp16 linear depth) + <stem>_logdepth.png (8-bit viz).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

import student as S


def load_model(args):
    if args.baseline:
        if args.baseline == "da3base":
            return S._load("da3-base", "model.safetensors").cuda().eval()
        return S._load("da3mono-large", "DA3MONO-LARGE.safetensors").cuda().eval()
    m = S.load_student()
    sd = S.load_sd(args.ckpt, args.ema)
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"[infer] loaded {args.ckpt} ({'ema' if args.ema else 'model'}), miss={len(miss)} unexp={len(unexp)}")
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--baseline", default=None, choices=[None, "da3base", "da3mono"])
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=1024)
    args = ap.parse_args()

    model = load_model(args)
    src = Path(args.images)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted([src] if src.is_file() else list(src.rglob("*")))
    files = [f for f in files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    print(f"[infer] {len(files)} images -> {out}")
    for i, f in enumerate(files):
        rgb = Image.open(f).convert("RGB")
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            d = model.inference([rgb], process_res=args.res).depth[0]
        d = np.asarray(d, np.float32)
        np.save(out / f"{f.stem}.npy", d.astype(np.float16))
        ld = np.log(np.clip(d, 1e-4, None))
        lo, hi = np.percentile(ld, 2), np.percentile(ld, 98)
        vis = np.clip((ld - lo) / max(hi - lo, 1e-6), 0, 1)
        Image.fromarray((vis * 255).astype(np.uint8)).save(out / f"{f.stem}_logdepth.png")
        if (i + 1) % 25 == 0:
            print(f"[infer] {i+1}/{len(files)}", flush=True)
    print("[infer] DONE")


if __name__ == "__main__":
    main()
