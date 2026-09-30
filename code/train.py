"""Single-stage dual-teacher fine-tune of DA3-BASE (plan v3.1).

L = 1.0*L_ssi + 0.7*L_grad + 0.5*L_edge(ramp) + 0.5*L_feat + L_conf
Guardrail fast-eval every eval_every iters: NYU-100 AbsRel + PPR-holdout.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import student as S
import losses as L
from dataset import build_entries, DualTeacherBatchDataset, BatchSpecSampler

NYU_ROOT = Path("/root/autodl-tmp/datasets/eval/nyu_labeled_extracted")
PPR_T2 = Path("/root/autodl-tmp/datasets/train/PPR10K/pseudo_depth_da3mono")
from dataset import find_ppr_t1_dir

STU_LAYERS = [2, 5, 8, 11]
TEA_LAYERS = [5, 11, 17, 23]   # 2i+1 mapping, ViT-B(12) <-> ViT-L(24)


def nyu_eval_list(n=100):
    lines = (NYU_ROOT / "filename_list_test.txt").read_text().splitlines()
    return [ln.split()[0] for ln in lines if ln.strip()][:n]  # rgb rel paths


@torch.no_grad()
def fast_eval_nyu(model, entries, process_res=768):
    model.eval()
    absrels, d1s = [], []
    from PIL import Image
    for name in entries:
        rgb_p = NYU_ROOT / name
        gt_p = rgb_p.parent / rgb_p.name.replace("rgb_", "depth_")
        if not rgb_p.exists() or not gt_p.exists():
            continue
        rgb = Image.open(rgb_p).convert("RGB")
        pred = model.inference([rgb], process_res=process_res).depth[0]
        if not np.isfinite(pred).all():   # poisoned weights must not kill the run
            continue
        gt = np.array(Image.open(gt_p), dtype=np.float64)
        if gt.max() > 1000:
            gt = gt / 1000.0
        H0, W0 = gt.shape
        pred = np.array(Image.fromarray(pred).resize((W0, H0), Image.BILINEAR), dtype=np.float64)
        pred = pred[45:471, 41:601]
        gt = gt[45:471, 41:601]
        valid = (gt > 1e-3) & (gt < 10)
        if valid.sum() < 100:
            continue
        pred = np.clip(pred, max(np.quantile(pred, 0.01), 1e-6), None)
        p, g = pred[valid], gt[valid]
        lp, lg = np.log(p), np.log(g)
        A = np.stack([lp, np.ones_like(lp)], 1)
        try:
            coef, *_ = np.linalg.lstsq(A, lg, rcond=None)
        except np.linalg.LinAlgError:
            continue
        d_hat = np.exp(np.clip(coef[0] * lp + coef[1], None, 10))
        g = gt[valid]
        absrels.append(np.mean(np.abs(d_hat - g) / g))
        d1s.append(np.mean(np.maximum(d_hat / g, g / d_hat) < 1.25))
    model.train()
    return float(np.median(absrels)) if absrels else float("nan"), float(np.mean(d1s)) if d1s else float("nan")


@torch.no_grad()
def fast_eval_ppr(model, ds, entries, n=50):
    model.eval()
    corrs, wmeans = [], []
    idxs = list(ds.holdout)[:n]
    t1_dir = find_ppr_t1_dir()
    from PIL import Image
    for i in idxs:
        img_p, t1_p, t2_p, _ = entries[i]
        rgb = Image.open(img_p).convert("RGB")
        pred = model.inference([rgb], process_res=768).depth[0]
        t1 = np.asarray(np.load(t1_p), np.float32)
        pred = np.array(Image.fromarray(pred).resize((t1.shape[1], t1.shape[0]), Image.BILINEAR), np.float32)
        pl = np.log(np.clip(pred, 1e-4, None)).ravel()
        tl = t1.ravel()
        if pl.size < 100:
            continue
        c = np.corrcoef(pl, tl)[0, 1]
        if np.isfinite(c):
            corrs.append(c)
        t2 = np.asarray(np.load(t2_p), np.float32)
        wmeans.append(float(np.exp(-np.abs(t1 - t2).mean() / L.TAU)))
    model.train()
    return float(np.median(corrs)) if corrs else float("nan"), float(np.mean(wmeans)) if wmeans else float("nan")


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()


class FiniteSampler(BatchSpecSampler):
    def __init__(self, ds, batches_per_epoch, **kw):
        super().__init__(ds, **kw)
        self.batches_per_epoch = batches_per_epoch

    def __iter__(self):
        it = super().__iter__()
        for _ in range(self.batches_per_epoch):
            yield next(it)

    def __len__(self):
        return self.batches_per_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", default="/root/autodl-tmp/depth300m_v2/runs/s1")
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--edge_warmup", type=int, default=500)
    ap.add_argument("--tau", type=float, default=0.2, help="dual-teacher consistency tau (R3: raise to 0.3 if mean w < 0.4)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--probe_only", action="store_true")
    args = ap.parse_args()
    L.TAU = args.tau

    # stdout goes to runs_s1.log (line-buffered so tail -f is live);
    # stderr carries the tqdm bar and stays on the tmux pane.
    sys.stdout.reconfigure(line_buffering=True)
    import logging
    logging.getLogger().setLevel(logging.WARNING)   # silence per-iter "Selecting reference view"
    try:
        from loguru import logger as _loguru
        _loguru.remove()
    except Exception:
        pass

    def _excepthook(tp, val, tb):
        import traceback
        txt = "".join(traceback.format_exception(tp, val, tb))
        sys.stdout.write(txt); sys.stdout.flush()   # also into the log file
        sys.stderr.write(txt)
    sys.excepthook = _excepthook

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model = S.load_student()
    # --- probe: figure out output keys / aux structure once ---
    out_probe = S.probe(model, STU_LAYERS)
    depth_key = "depth"
    conf_key = None
    for k in ("conf", "depth_conf", "confidence"):
        if k in out_probe:
            conf_key = k
    aux = getattr(out_probe, "aux", None) or {}
    print("[train] conf_key =", conf_key, "| aux keys:", list(aux.keys()) if isinstance(aux, dict) else type(aux))
    if args.probe_only:
        return

    teacher = S.load_teacher()
    tea_probe = S.probe(teacher, TEA_LAYERS)

    # L_feat wiring: student feat_layer_{i} <-> teacher feat_layer_{2i+1}
    pairs = [(f"feat_layer_{i}", f"feat_layer_{j}") for i, j in zip(STU_LAYERS, TEA_LAYERS)]
    s_dim = out_probe.aux[pairs[0][0]].shape[-1] if getattr(out_probe, "aux", None) else 768
    t_dim = tea_probe.aux[pairs[0][1]].shape[-1] if getattr(tea_probe, "aux", None) else 1024
    projectors = nn.ModuleDict({sk: nn.Linear(s_dim, t_dim) for sk, _ in pairs}).cuda()
    for mproj in projectors.values():
        nn.init.normal_(mproj.weight, std=0.02); nn.init.zeros_(mproj.bias)
    print(f"[train] L_feat pairs={pairs} dims {s_dim}->{t_dim}")

    entries = build_entries()
    ds = DualTeacherBatchDataset(entries)
    print(f"[train] entries: {len(entries)} (coco={len(ds.coco_idx)} ppr={len(ds.ppr_idx)} holdout={len(ds.holdout)})")
    sampler = FiniteSampler(ds, batches_per_epoch=1000, ppr_prob=0.25, batch=args.batch, seed=0)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=16, pin_memory=True, persistent_workers=True)

    params = list(model.parameters()) + list(projectors.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    warmup = 500
    def lr_at(it):
        if it < warmup:
            return args.lr * it / warmup
        t = (it - warmup) / max(1, args.iters - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))
    ema = EMA(model)
    start_it = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location="cuda")
        model.load_state_dict(ck["model"], strict=False)
        projectors.load_state_dict(ck["proj"])
        ema.shadow = {k: v.clone() for k, v in ck["ema"].items()}
        start_it = ck.get("iter", 0)
        print(f"[train] resumed from {args.resume} @ iter {start_it}")
    nyu_list = nyu_eval_list(100)

    it, t0 = start_it, time.time()
    nskip = 0
    pbar = tqdm(total=args.iters, initial=it, dynamic_ncols=True, file=sys.stderr,
                unit="it", smoothing=0.03, mininterval=2.0, leave=True)
    logf = open(out / "log.jsonl", "a")
    while it < args.iters:
        for batch in loader:
            if it >= args.iters:
                break
            rgb, t1, t2 = [x.cuda(non_blocking=True).float() for x in batch]
            for g in opt.param_groups:
                g["lr"] = lr_at(it)
            with torch.autocast("cuda", torch.bfloat16):
                s_out = S.forward_train(model, rgb, STU_LAYERS)
                with torch.no_grad():
                    t_out = S.forward_train(teacher, rgb, TEA_LAYERS)
            pred = s_out[depth_key].float()
            if pred.dim() == 4 and pred.shape[0] == 1:
                pred = pred[0]
            conf = s_out[conf_key].float() if conf_key else None
            if conf is not None and conf.dim() == 4 and conf.shape[0] == 1:
                conf = conf[0]
            # DPT head reconstructs H/W off-by-one vs input -> align to label grid
            if pred.shape[-2:] != t1.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=t1.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
                if conf is not None and conf.shape[-2:] != t1.shape[-2:]:
                    conf = torch.nn.functional.interpolate(
                        conf.unsqueeze(1), size=t1.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            w = L.dual_teacher_weight(t1, t2)
            pred_norm, _ = L.align_logdepth(pred, t1, w)
            edge_w = 0.5 * min(1.0, it / max(1, args.edge_warmup))
            loss = (1.0 * L.l_ssi(pred_norm, t1, w)
                    + 0.7 * L.l_grad(pred_norm, t1, w)
                    + edge_w * L.l_edge(pred_norm, t1, w)
                    + 0.5 * L.l_feat(s_out.aux, t_out.aux, projectors, pairs)
                    + L.l_conf(conf, (pred_norm - t1).abs().detach()))
            if not torch.isfinite(loss):
                # a single Inf/NaN batch used to poison all 269 trainable
                # tensors via clip_grad_norm; skip the step, keep going
                nskip += 1
                if nskip > 200:
                    raise RuntimeError("loss non-finite 200 times - model is poisoned, check weights")
                tqdm.write(f"[warn] non-finite loss @iter {it}, batch skipped ({nskip} total)", file=sys.stderr)
                pbar.update(1)
                it += 1
                continue
            (loss / args.accum).backward()
            if (it + 1) % args.accum == 0:
                # finite loss does NOT imply finite grads (sqrt(0)-type Inf
                # backwards); clip turns Inf into NaN across all params
                bad = any(p.grad is not None and not torch.isfinite(p.grad).all()
                          for p in params)
                if bad:
                    nskip += 1
                    if nskip > 200:
                        raise RuntimeError("non-finite grads 200 times - model poisoned")
                    tqdm.write(f"[warn] non-finite grads @iter {it}, step skipped ({nskip} total)",
                               file=sys.stderr)
                    opt.zero_grad(set_to_none=True)
                else:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step(); opt.zero_grad(set_to_none=True)
                    ema.update(model)
            if it % 100 == 0:
                ms = (time.time() - t0) * 1000 / max(1, it - start_it + 1)
                eta_m = (args.iters - it) * ms / 1000 / 60
                msg = {"iter": it, "pct": round(100 * it / args.iters, 1), "loss": round(loss.item(), 4),
                       "lr": lr_at(it), "ms": round(ms, 1), "eta_min": round(eta_m, 1)}
                print(f"[train] {it}/{args.iters} {msg['pct']}% loss {msg['loss']} "
                      f"lr {msg['lr']:.2e} {msg['ms']}ms/it ETA {msg['eta_min']}m", flush=True)
                pbar.set_postfix(loss=msg["loss"], lr=f"{msg['lr']:.1e}")
                logf.write(json.dumps(msg) + "\n"); logf.flush()
            if (it + 1) % args.eval_every == 0:
                tqdm.write("[eval] running sentinel (NYU-100 + PPR-50)...", file=sys.stderr)
                ar, d1 = fast_eval_nyu(model, nyu_list)
                corr, wmean = fast_eval_ppr(model, ds, entries)
                msg = {"iter": it + 1, "nyu_absrel_median": ar, "nyu_d1": d1,
                       "ppr_corr": corr, "ppr_w": wmean, "guard_nyu": ar}
                line = "[eval] " + json.dumps(msg)
                print(line, flush=True)
                tqdm.write(line, file=sys.stderr)
                pbar.set_postfix(loss=round(loss.item(), 4), nyu=round(ar, 4))
                logf.write(json.dumps(msg) + "\n"); logf.flush()
            if (it + 1) % args.save_every == 0:
                torch.save({"model": model.state_dict(), "ema": ema.shadow,
                            "proj": projectors.state_dict(), "iter": it + 1},
                           out / "last.pt")
            pbar.update(1)
            it += 1
    pbar.close()
    torch.save({"model": model.state_dict(), "ema": ema.shadow,
                "proj": projectors.state_dict(), "iter": it}, out / "last.pt")
    print("[train] DONE", it, "iters in", (time.time() - t0) / 3600, "h", flush=True)


if __name__ == "__main__":
    main()
