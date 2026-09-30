"""HW portrait eval: Pearson corr vs SD3 reference (log-depth) + Laplacian sharpness ratio.

Usage: python eval_hw.py [--ckpt ... --ema | --baseline da3base|da3mono] [--res 1024]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import laplace

HW_ROOT = Path("/root/autodl-tmp/datasets/eval/HW")


def load_model(args):
    import student as S
    if args.baseline:
        ck = "model.safetensors" if args.baseline == "da3base" else "DA3MONO-LARGE.safetensors"
        name = "da3-base" if args.baseline == "da3base" else "da3mono-large"
        return S._load(name, ck).cuda().eval()
    m = S.load_student()
    sd = S.load_sd(args.ckpt, args.ema)
    m.load_state_dict(sd, strict=False)
    return m.eval()


def logdepth_norm(a):
    ld = np.log(np.clip(np.asarray(a, np.float64), 1e-6, None))
    lo, hi = np.percentile(ld, 2), np.percentile(ld, 98)
    return (ld - lo) / max(hi - lo, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--save_vis", default="/root/autodl-tmp/depth300m_v2/eval_out/hw_vis")
    args = ap.parse_args()
    model = load_model(args)
    imgs = sorted((HW_ROOT / "data").glob("*.jpg"))
    rows = []
    vis_dir = Path(args.save_vis); vis_dir.mkdir(parents=True, exist_ok=True)
    for f in imgs:
        # CMU files carry a double extension in the image name; refs may or may not keep the inner .jpg
        stems = [f.stem, f.stem.replace(".jpg_edof", "_edof"), f.stem.replace(".jpg_edof", ""), f.stem.replace(".jpg", "")]
        ref_hits = []
        for st in stems:
            ref_hits = list((HW_ROOT / "SD3").glob(f"{st}*"))
            if ref_hits:
                break
        if not ref_hits:
            continue
        ref = np.array(Image.open(ref_hits[0]).convert("L"), np.float64)
        rgb = Image.open(f).convert("RGB")
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            pred = model.inference([rgb], process_res=args.res).depth[0]
        pred = np.asarray(pred, np.float32)
        ref_r = np.array(Image.fromarray(ref).resize((pred.shape[1], pred.shape[0]), Image.BILINEAR), np.float64)
        pl = logdepth_norm(pred).ravel()
        # SD3 refs are near-bright disparity-style visualizations -> flip to log-depth orientation
        rl = (1.0 - logdepth_norm(ref_r)).ravel()
        corr = float(np.corrcoef(pl, rl)[0, 1])
        lap_s = float(laplace(logdepth_norm(pred), mode="reflect").var())
        lap_r = float(laplace(logdepth_norm(ref_r), mode="reflect").var())
        rows.append(dict(name=f.stem, corr=corr, lap_ratio=lap_s / max(lap_r, 1e-12),
                         lap_s=lap_s, lap_r=lap_r))
        H0 = 384
        rgb_sm = np.asarray(rgb.resize((int(H0 * rgb.width / rgb.height), H0)), np.float32)
        pd_sm = np.array(Image.fromarray(pred).resize((int(H0 * pred.shape[1] / pred.shape[0]), H0)), np.float32)
        ld_sm = logdepth_norm(pd_sm)
        depth_sm = (np.stack([ld_sm] * 3, -1)[..., ::-1] * 255).astype(np.uint8)
        trio = np.concatenate([np.asarray(rgb_sm, np.uint8), depth_sm], axis=1)
        Image.fromarray(trio).save(vis_dir / f"{f.stem}_cmp.jpg")
    corrs = [r["corr"] for r in rows]
    ratios = [r["lap_ratio"] for r in rows]
    summary = dict(n=len(rows),
                   corr_median=float(np.median(corrs)), corr_mean=float(np.mean(corrs)),
                   corr_min=float(np.min(corrs)),
                   lap_ratio_median=float(np.median(ratios)),
                   lap_ratio_mean=float(np.mean(ratios)))
    print("[hw] SUMMARY", json.dumps(summary, indent=1))
    tag = args.baseline or (args.ckpt or "x").replace("/", "_")
    out = Path("/root/autodl-tmp/depth300m_v2/eval_out")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"hw_{tag}.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
