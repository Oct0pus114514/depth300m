"""Dual-teacher dataset: COCO + PPR10K with T1/T2 labels, batch-level crop config.

Label space (both teachers): per-image 2-98% quantile normalized log-depth in [-1,1].
"""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

COCO_IMG = Path("/root/autodl-tmp/datasets/train/coco2017/train2017")
COCO_T1 = Path("/root/autodl-tmp/datasets/train/coco2017/pseudo_depth_marigoldv2/images/predictions_npy")
COCO_T2 = Path("/root/autodl-tmp/datasets/train/coco2017/pseudo_depth_da3mono")
PPR_IMG = Path("/root/autodl-tmp/datasets/train/PPR10K/ppr10k_1536")
PPR_T2 = Path("/root/autodl-tmp/datasets/train/PPR10K/pseudo_depth_da3mono")
_PPR_T1_CANDIDATES = [
    Path("/root/autodl-tmp/datasets/train/PPR10K/pseudo_depth_marigoldv2/predictions_npy"),
    Path("/root/autodl-tmp/datasets/train/PPR10K/pseudo_depth_marigoldv2/images/predictions_npy"),
    Path("/root/autodl-tmp/datasets/train/PPR10K/ppr10k_1536/predictions_npy"),
]


def find_ppr_t1_dir():
    for d in _PPR_T1_CANDIDATES:
        if d.is_dir() and any(d.glob("*.npy")):
            return d
    root = Path("/root/autodl-tmp/datasets/train/PPR10K/pseudo_depth_marigoldv2")
    if root.is_dir():
        hits = [p for p in root.rglob("predictions_npy") if any(p.glob("*.npy"))]
        if hits:
            return hits[0]
    return None


def build_entries():
    """List of (rgb, t1, t2, source); requires both labels to exist."""
    entries = []
    t1_dir = find_ppr_t1_dir()
    for img in sorted(COCO_IMG.glob("*.jpg")):
        t1 = COCO_T1 / f"{img.stem}.npy"
        t2 = COCO_T2 / f"{img.stem}.npy"
        if t1.exists() and t2.exists():
            entries.append((img, t1, t2, "coco"))
    if t1_dir is not None:
        for img in sorted(PPR_IMG.glob("*.jpg")):
            t1 = t1_dir / f"{img.stem}.npy"
            t2 = PPR_T2 / f"{img.stem}.npy"
            if t1.exists() and t2.exists():
                entries.append((img, t1, t2, "ppr"))
    return entries


class CropConfig:
    """All dims are multiples of 14 (patch size)."""
    COCO = [(448, 616)]                                   # 32x14, 44x14
    PPR = [(756, 756), (770, 1022), (1008, 1008)]         # square / ~3:4 portrait / hi-res
    # per-spec batch sizes ~equalize pixels per forward (RTX PRO 6000 96GB).
    # OOM field test: 1008^2 x10 -> 90.7 GiB real (probe said 84: EMA+grads+fragments
    # unaccounted). 8x1008^2 ran 4000 iters OOM-free -> proven worst envelope.
    SPEC_BATCH = {(448, 616): 24, (756, 756): 12, (770, 1022): 10, (1008, 1008): 8}

    @classmethod
    def draw(cls, source, rng):
        if source == "coco":
            return cls.COCO[0]
        r = rng.random()
        if r < 0.34:
            return cls.PPR[2]
        return cls.PPR[0] if rng.random() < 0.5 else cls.PPR[1]


class DualTeacherBatchDataset(Dataset):
    """__getitem__ takes (idx, (h, w)); crop geometry is decided by the batch sampler."""

    def __init__(self, entries, holdout_ppr=300, seed=42):
        self.entries = entries
        rng = random.Random(seed)
        ppr_all = [i for i, e in enumerate(entries) if e[3] == "ppr"]
        rng.shuffle(ppr_all)
        self.holdout = set(ppr_all[:holdout_ppr])
        self.train_idx = [i for i in range(len(entries)) if i not in self.holdout]
        self.coco_idx = [i for i in self.train_idx if entries[i][3] == "coco"]
        self.ppr_idx = [i for i in self.train_idx if entries[i][3] == "ppr"]

    def __len__(self):
        return len(self.train_idx)

    def __getitem__(self, args):
        idx, (th, tw) = args
        img_p, t1_p, t2_p, _ = self.entries[idx]
        rgb = Image.open(img_p).convert("RGB")
        W, H = rgb.size
        ar_t = th / tw
        if W / H > ar_t:
            ch, cw = H, int(round(H * ar_t))
        else:
            cw, ch = W, int(round(W / ar_t))
        # crop scale: keep within image; slight jitter
        can_full = (min(W, H) >= max(th, tw)) if min(cw, ch) >= min(th, tw) else False
        s = random.uniform(0.7, 1.0)
        cw2 = max(8, min(W, int(cw * s)))
        ch2 = max(8, min(H, int(ch * s)))
        x0 = random.randint(0, W - cw2)
        y0 = random.randint(0, H - ch2)
        rgb = rgb.crop((x0, y0, x0 + cw2, y0 + ch2)).resize((tw, th), Image.BILINEAR)
        rgb_t = torch.from_numpy(np.asarray(rgb, np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        rgb_t = (rgb_t - mean) / std

        t1 = np.asarray(np.load(t1_p, mmap_mode="r"), np.float32)
        t2 = np.asarray(np.load(t2_p, mmap_mode="r"), np.float32)

        def crop_resize(a):
            im = Image.fromarray(np.uint8(np.clip(a * 0.5 + 0.5, 0, 1) * 255.0))
            # map image-space crop box to label-space (labels share image geometry)
            bx0 = int(round(x0 / W * im.width)); by0 = int(round(y0 / H * im.height))
            bx1 = int(round((x0 + cw2) / W * im.width)); by1 = int(round((y0 + ch2) / H * im.height))
            bx1 = max(bx0 + 1, bx1); by1 = max(by0 + 1, by1)
            im = im.crop((bx0, by0, bx1, by1)).resize((tw, th), Image.BILINEAR)
            return torch.from_numpy(np.asarray(im, np.float32) / 255.0 * 2.0 - 1.0)

        return rgb_t, crop_resize(t1), crop_resize(t2)


class BatchSpecSampler:
    """Yields lists of (idx, (h,w)); one crop spec per collated batch."""

    def __init__(self, ds, ppr_prob=0.25, batch=8, seed=0):
        self.ds, self.ppr_prob, self.batch = ds, ppr_prob, batch
        self.rng = random.Random(seed)

    def __iter__(self):
        while True:
            src = "ppr" if (self.ds.ppr_idx and self.rng.random() < self.ppr_prob) else "coco"
            pool = self.ds.ppr_idx if src == "ppr" else self.ds.coco_idx
            if not pool:
                pool = self.ds.coco_idx
            hw = CropConfig.draw(src, self.rng)
            bs = CropConfig.SPEC_BATCH.get(hw, self.batch)
            yield [(self.rng.choice(pool), hw) for _ in range(bs)]
