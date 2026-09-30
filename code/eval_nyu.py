"""Full NYUv2 official-654 eval (eigen crop, disparity-lstsq alignment).

Usage: python eval_nyu.py [--ckpt runs/s1/last.pt --ema | --baseline da3base|da3mono] [--res 768]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

NYU_ROOT = Path("/root/autodl-tmp/datasets/eval/nyu_labeled_extracted")


def load_model(args):
    import torch, student as S
    if args.baseline:
        ck = "model.safetensors" if args.baseline == "da3base" else "DA3MONO-LARGE.safetensors"
        name = "da3-base" if args.baseline == "da3base" else "da3mono-large"
        return S._load(name, ck).cuda().eval()
    m = S.load_student()
    ck = torch.load(args.ckpt, map_location="cuda")
    sd = ck["ema"] if args.ema else ck["model"]
    m.load_state_dict(sd, strict=False)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    model = load_model(args)
    names = (NYU_ROOT / "filename_list_test.txt").read_text().splitlines()
    names = [ln.split()[0] for ln in names if ln.strip()]  # col0 = test/<scene>/rgb_xxxx.png
    if args.limit:
        names = names[: args.limit]
    rows = []
    for n, name in enumerate(names):
        rgb_p = NYU_ROOT / name
        gt_p = rgb_p.parent / rgb_p.name.replace("rgb_", "depth_")
        if not rgb_p.exists() or not gt_p.exists():
            continue
        rgb = Image.open(rgb_p).convert("RGB")
        pred = model.inference([rgb], process_res=args.res).depth[0]
        gt = np.array(Image.open(gt_p), dtype=np.float64)
        if gt.max() > 1000:
            gt = gt / 1000.0
        H0, W0 = gt.shape
        pred = np.array(Image.fromarray(pred).resize((W0, H0), Image.BILINEAR), dtype=np.float64)
        # eigen crop applied IDENTICALLY to pred and gt (never resize into the crop window)
        pred = pred[45:471, 41:601]
        gt = gt[45:471, 41:601]
        valid = (gt > 1e-3) & (gt < 10)
        # robust: clip near-zero outliers (mono heads can emit ~0 pixels) before disparity
        pred = np.clip(pred, max(np.quantile(pred, 0.01), 1e-6), None)
        p, g = pred[valid], gt[valid]
        # primary: log-affine alignment (DA3 heads compress dynamic range; log space absorbs it)
        lp, lg = np.log(p), np.log(g)
        A2 = np.stack([lp, np.ones_like(lp)], 1)
        c2, *_ = np.linalg.lstsq(A2, lg, rcond=None)
        d_hat = np.exp(np.clip(c2[0] * lp + c2[1], None, 10))
        # secondary: disparity-lstsq
        pd_, gd_ = 1 / p, 1 / g
        A = np.stack([pd_, np.ones_like(pd_)], 1)
        c, *_ = np.linalg.lstsq(A, gd_, rcond=None)
        d_hat2 = 1 / np.clip(c[0] * pd_ + c[1], 1e-9, None)
        g = gt[valid]
        rows.append(dict(
            name=name,
            absrel=float(np.mean(np.abs(d_hat - g) / g)),
            absrel_disparity=float(np.mean(np.abs(d_hat2 - g) / g)),
            d1=float(np.mean(np.maximum(d_hat / g, g / d_hat) < 1.25)),
            rmse=float(np.sqrt(((d_hat - g) ** 2).mean())),
        ))
        if (n + 1) % 100 == 0:
            print(f"[nyu] {n+1}/{len(names)} running AbsRel={np.mean([r['absrel'] for r in rows]):.4f}", flush=True)
    ar = [r["absrel"] for r in rows]
    d1 = [r["d1"] for r in rows]
    rmse = [r["rmse"] for r in rows]
    summary = dict(n=len(rows), absrel_mean=float(np.mean(ar)), absrel_median=float(np.median(ar)),
                   d1=float(np.mean(d1)), rmse=float(np.mean(rmse)))
    print("[nyu] SUMMARY", json.dumps(summary))
    tag = args.baseline or (args.ckpt or "x").replace("/", "_")
    Path("/root/autodl-tmp/depth300m_v2/eval_out").mkdir(parents=True, exist_ok=True)
    with open(f"/root/autodl-tmp/depth300m_v2/eval_out/nyu_{tag}.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
