"""Final HW demos, 4 panels: RGB | Marigold V2 | DA3-BASE | Ours (EMA).

All depth panels turbo_r (near=red, far=blue), per-image 2-98% log-depth norm.
Index file records corr vs SD3 ref for each depth panel.
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
T1_DIR = HW_ROOT / "pseudo_t1"
OUT = Path("/root/autodl-tmp/demo/hw_final")

try:
    CMAP = cm.get_cmap("turbo_r")
except Exception:
    from matplotlib import colormaps
    CMAP = colormaps["turbo_r"]

FONT_BOLD = ImageFont.truetype(font_manager.findfont("DejaVu Sans Bold"), 26)
FONT_SMALL = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 15)


def ref_for(stem):
    for st in (stem, stem.replace(".jpg_edof", "_edof"),
               stem.replace(".jpg_edof", ""), stem.replace(".jpg", "")):
        hits = list((HW_ROOT / "SD3").glob(f"{st}*"))
        if hits:
            return hits[0]
    return None


def t1_for(stem):
    hits = [p for p in T1_DIR.rglob(f"{stem}.npy")]
    return hits[0] if hits else None


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
    base = S._load("da3-base", "model.safetensors").cuda().eval()
    iterno = ck["iter"]
    print(f"[demo] ours = EMA @ iter {iterno}", flush=True)
    imgs = sorted((HW_ROOT / "data").glob("*.jpg"))
    rows = []
    for i, f in enumerate(imgs, 1):
        rgb = Image.open(f).convert("RGB")
        t1p = t1_for(f.stem)
        assert t1p is not None, f"no Marigold label for {f.name}"
        t1 = np.asarray(np.load(t1p, mmap_mode="r"), np.float32)
        with torch.autocast("cuda", torch.bfloat16):
            p_ours = np.asarray(ours.inference([rgb], process_res=1024).depth[0], np.float32)
            p_base = np.asarray(base.inference([rgb], process_res=1024).depth[0], np.float32)
        m01 = (np.asarray(t1, np.float32) + 1.0) / 2.0          # [-1,1] label -> [0,1]
        b01 = logdepth_norm(p_base)
        o01 = logdepth_norm(p_ours)
        # corr vs SD3 ref on the ours-pred grid
        rp = ref_for(f.stem)
        r_m = r_b = r_o = float("nan")
        if rp is not None:
            ref = np.array(Image.open(rp).convert("L"), np.float64)
            ref_r = np.array(Image.fromarray(ref).resize((p_ours.shape[1], p_ours.shape[0]),
                                                         Image.BILINEAR), np.float64)
            rl = 1.0 - logdepth_norm(ref_r)
            m_r = np.array(Image.fromarray((np.clip(m01, 0, 1) * 255).astype(np.uint8)).resize(
                (rl.shape[1], rl.shape[0]), Image.BILINEAR), np.float32) / 255.0
            r_o = float(np.corrcoef(o01.ravel(), rl.ravel())[0, 1])
            r_b = float(np.corrcoef(b01.ravel(), rl.ravel())[0, 1])
            r_m = float(np.corrcoef(m_r.ravel(), rl.ravel())[0, 1])
        rows.append((f.name, r_m, r_b, r_o))
        H = 384
        panels = [
            titled(panel_rgb(rgb, H), "RGB"),
            titled(to_h(colorize(m01), H), "Marigold V2", f"teacher T1, r={r_m:.3f}"),
            titled(to_h(colorize(b01), H), "DA3-BASE", f"pretrained, r={r_b:.3f}"),
            titled(to_h(colorize(o01), H), "Ours", f"EMA iter {iterno}, r={r_o:.3f}"),
        ]
        gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
        comp = panels[0]
        for p in panels[1:]:
            comp = np.concatenate([comp, gap, p], 1)
        Image.fromarray(comp).save(OUT / f"demo_{i:02d}.jpg", quality=90)
        print(f"demo_{i:02d}.jpg r: t1={r_m:.4f} base={r_b:.4f} ours={r_o:.4f} <- {f.name}", flush=True)
    rs = np.array([(m, b, o) for _, m, b, o in rows])
    med = np.nanmedian(rs, 0)
    print(f"[demo] SUMMARY corr_median: marigold={med[0]:.4f} base={med[1]:.4f} ours={med[2]:.4f}")
    with open(OUT / "demo_index.txt", "w", encoding="utf-8") as fh:
        fh.write(f"ours = EMA weights @ iter {iterno}; colormap turbo_r near=red far=blue\n")
        fh.write(f"corr_median vs SD3 ref: marigold={med[0]:.4f} da3base={med[1]:.4f} ours={med[2]:.4f}\n")
        for i, (name, m, b, o) in enumerate(rows, 1):
            fh.write(f"demo_{i:02d}.jpg <- {name} | r_t1={m:.4f} r_base={b:.4f} r_ours={o:.4f}\n")


if __name__ == "__main__":
    main()
