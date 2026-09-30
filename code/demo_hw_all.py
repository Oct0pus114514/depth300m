"""HW full-set demos: RGB | Ours (EMA) | SD3 ref.

All depth panels use turbo_r (reversed turbo): near = red, far = blue,
per-image 2-98% quantile log-depth normalization (same protocol as eval_hw).
"""
from __future__ import annotations
import sys, torch, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
from matplotlib import font_manager

sys.path.insert(0, "/root/autodl-tmp/depth300m_v2")
import student as S
from eval_hw import logdepth_norm, HW_ROOT

CKPT = "/root/autodl-tmp/depth300m_v2/runs/s1/last.pt"
OUT = Path("/root/autodl-tmp/demo/hw_all")

try:
    CMAP = cm.get_cmap("turbo_r")
except Exception:
    from matplotlib import colormaps
    CMAP = colormaps["turbo_r"]

FONT_BOLD = ImageFont.truetype(font_manager.findfont("DejaVu Sans Bold"), 26)
FONT_SMALL = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 15)


def find_ref(stem):
    for st in (stem, stem.replace(".jpg_edof", "_edof"),
               stem.replace(".jpg_edof", ""), stem.replace(".jpg", "")):
        hits = list((HW_ROOT / "SD3").glob(f"{st}*"))
        if hits:
            return hits[0]
    return None


def colorize(a01):
    return (CMAP(np.clip(a01, 0, 1))[..., :3] * 255).astype(np.uint8)


def to_h(arr, h=384):
    im = Image.fromarray(arr)
    return np.asarray(im.resize((int(h * im.width / im.height), h), Image.BILINEAR), np.uint8)


def panel_rgb(img, h=384):
    w = int(h * img.width / img.height)
    return np.asarray(img.resize((w, h), Image.BILINEAR), np.uint8)


def titled(img, label, sub=None):
    bar_im = Image.fromarray(np.full((58, img.shape[1], 3), 255, np.uint8))
    d = ImageDraw.Draw(bar_im)
    d.text((10, 4), label, fill=(0, 0, 0), font=FONT_BOLD)
    if sub:
        d.text((12, 36), sub, fill=(60, 60, 60), font=FONT_SMALL)
    return np.concatenate([np.asarray(bar_im), img], 0)


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ours = S.load_student()
    ck = torch.load(CKPT, map_location="cpu")
    ours.load_state_dict(ck["ema"], strict=False)
    ours.eval()
    iterno = ck["iter"]
    print(f"[demo] EMA weights @ iter {iterno}", flush=True)
    imgs = sorted((HW_ROOT / "data").glob("*.jpg"))
    rows = []
    for i, f in enumerate(imgs, 1):
        rgb = Image.open(f).convert("RGB")
        with torch.autocast("cuda", torch.bfloat16):
            pred = ours.inference([rgb], process_res=1024).depth[0]
        pred = np.asarray(pred, np.float32)
        o01 = logdepth_norm(pred)
        ref_p = find_ref(f.stem)
        if ref_p is None:
            print(f"[skip] no SD3 ref for {f.name}", flush=True)
            continue
        ref = np.array(Image.open(ref_p).convert("L"), np.float64)
        ref_r = np.array(Image.fromarray(ref).resize((pred.shape[1], pred.shape[0]),
                                                     Image.BILINEAR), np.float64)
        rl = 1.0 - logdepth_norm(ref_r)      # disparity-style ref -> near-red orientation
        r = float(np.corrcoef(o01.ravel(), rl.ravel())[0, 1])
        rows.append((f.name, r))
        H = 384
        panels = [
            titled(panel_rgb(rgb, H), "RGB"),
            titled(to_h(colorize(o01), H), "Ours", f"EMA iter {iterno}, r={r:.3f}"),
            titled(to_h(colorize(rl), H), "SD3 ref", "near red / far blue"),
        ]
        gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
        comp = panels[0]
        for p in panels[1:]:
            comp = np.concatenate([comp, gap, p], 1)
        Image.fromarray(comp).save(OUT / f"demo_{i:02d}.jpg", quality=90)
        print(f"demo_{i:02d}.jpg r={r:.4f} <- {f.name}", flush=True)
    rs = [r for _, r in rows]
    summary = f"n={len(rows)} corr_median={np.median(rs):.4f} corr_mean={np.mean(rs):.4f}"
    print("[demo] SUMMARY", summary)
    with open(OUT / "demo_index.txt", "w", encoding="utf-8") as fh:
        fh.write(f"ours = EMA weights @ iter {iterno} of runs/s1\ncolormap: turbo_r (near=red, far=blue)\n{summary}\n")
        for i, (name, r) in enumerate(rows, 1):
            fh.write(f"demo_{i:02d}.jpg <- {name} | r={r:.4f}\n")


if __name__ == "__main__":
    main()
