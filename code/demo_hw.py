"""HW demo: 4-panel composites (RGB | Ours-EMA | DA3-BASE | SD3 ref).

Depth panels shown in log-depth orientation (near=dark, far=bright), turbo colormap,
per-image 2-98% quantile normalization (same as eval_hw.py protocol).
"""
from __future__ import annotations
import sys, torch, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
from matplotlib import font_manager

FONT_BOLD = ImageFont.truetype(font_manager.findfont("DejaVu Sans Bold"), 26)
FONT_SMALL = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 15)

sys.path.insert(0, "/root/autodl-tmp/depth300m_v2")
import student as S
from eval_hw import logdepth_norm, HW_ROOT

CKPT = "/root/autodl-tmp/depth300m_v2/runs/s1/last.pt"
OUT = Path("/root/autodl-tmp/demo")
PICKS = [
    "027_zicai_004_SA+人像_IMG_20260318_234518_edof.jpg",
    "076_zicai_051_SCA_IMG_1778726593_055_edof.jpg",
    "10014_CMU_测试5346_IMG_19700103_122329.jpg_edof.jpg",
]


def find_ref(stem):
    # same candidate logic as eval_hw.py (CMU double extensions)
    for st in (stem, stem.replace(".jpg_edof", "_edof"),
               stem.replace(".jpg_edof", ""), stem.replace(".jpg", "")):
        hits = list((HW_ROOT / "SD3").glob(f"{st}*"))
        if hits:
            return hits[0]
    return None


def colorize(a01):
    rgb = cm.get_cmap("turbo")(np.clip(a01, 0, 1))[..., :3]
    return (rgb * 255).astype(np.uint8)


def panel(img_rgb, h=384):
    w = int(h * img_rgb.width / img_rgb.height)
    return np.asarray(img_rgb.resize((w, h), Image.BILINEAR), np.uint8)


def to_h(arr, h=384):
    im = Image.fromarray(arr)
    return np.asarray(im.resize((int(h * im.width / im.height), h), Image.BILINEAR), np.uint8)


def titled(img, label, sub=None):
    # draw on the PIL image itself; fromarray COPIES the buffer, so drawing on
    # a fromarray(bar) copy and concatenating the original bar renders nothing
    bar_im = Image.fromarray(np.full((58, img.shape[1], 3), 255, np.uint8))
    d = ImageDraw.Draw(bar_im)
    d.text((10, 4), label, fill=(0, 0, 0), font=FONT_BOLD)
    if sub:
        d.text((12, 36), sub, fill=(60, 60, 60), font=FONT_SMALL)
    return np.concatenate([np.asarray(bar_im), img], 0)


@torch.no_grad()
def infer(model, rgb, res=1024):
    with torch.autocast("cuda", torch.bfloat16):
        pred = model.inference([rgb], process_res=res).depth[0]
    return np.asarray(pred, np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ours = S.load_student()
    ck = torch.load(CKPT, map_location="cpu")
    ours.load_state_dict(ck["ema"], strict=False)
    ours.eval()
    iterno = ck["iter"]
    base = S._load("da3-base", "model.safetensors").cuda().eval()
    index_lines = [f"ours = EMA weights @ iter {iterno} of runs/s1"]
    for i, name in enumerate(PICKS, 1):
        f = HW_ROOT / "data" / name
        rgb = Image.open(f).convert("RGB")
        p_ours = infer(ours, rgb)
        p_base = infer(base, rgb)
        ref_p = find_ref(f.stem)
        assert ref_p is not None, f"no SD3 ref for {name}"
        ref = np.array(Image.open(ref_p).convert("L"), np.float64)
        ref_r = np.array(Image.fromarray(ref).resize((p_ours.shape[1], p_ours.shape[0]),
                                                     Image.BILINEAR), np.float64)
        rl = (1.0 - logdepth_norm(ref_r))          # flip to near-dark/far-bright
        o01 = logdepth_norm(p_ours)
        b01 = logdepth_norm(p_base)
        r_ours = float(np.corrcoef(o01.ravel(), rl.ravel())[0, 1])
        r_base = float(np.corrcoef(b01.ravel(), rl.ravel())[0, 1])
        H = 384
        panels = [
            titled(panel(rgb, H), "RGB"),
            titled(to_h(colorize(o01), H), "Ours", f"EMA iter {iterno}, r={r_ours:.3f}"),
            titled(to_h(colorize(b01), H), "DA3-BASE", f"pretrained, r={r_base:.3f}"),
            titled(to_h(colorize(rl), H), "SD3 ref"),
        ]
        hmax = max(p.shape[0] for p in panels)
        pads = [np.vstack([p, np.zeros((hmax - p.shape[0], p.shape[1], 3), np.uint8)]) for p in panels]
        gap = np.full((hmax, 6, 3), 255, np.uint8)
        comp = pads[0]
        for p in pads[1:]:
            comp = np.concatenate([comp, gap, p], 1)
        out_p = OUT / f"demo_{i}.jpg"
        Image.fromarray(comp).save(out_p, quality=92)
        line = f"demo_{i}.jpg <- {name} | ref={ref_p.name} | r_ours={r_ours:.4f} r_base={r_base:.4f}"
        index_lines.append(line)
        print(line, flush=True)
    (OUT / "demo_index.txt").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print("[demo] saved to", OUT)


if __name__ == "__main__":
    main()
