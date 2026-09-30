"""Five-term loss with dual-teacher consistency weighting (plan v3.1)."""
from __future__ import annotations
import torch
import torch.nn.functional as F

TAU = 0.2      # ~10% of the [-1,1] label range
WK = 7         # odd kernel so pooling preserves H/W (even kernels give +1)


def sobel(x):
    """Accepts (B,H,W) or (B,1,H,W); returns gradient maps with batch dims preserved."""
    squeeze = x.dim() == 3
    if squeeze:
        x = x.unsqueeze(1)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)[None, None]
    ky = kx.transpose(2, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    if squeeze:
        gx, gy = gx.squeeze(1), gy.squeeze(1)
    return gx, gy


def dual_teacher_weight(t1, t2, tau=TAU, k=WK):
    d = (t1 - t2).abs().unsqueeze(1)
    d = F.avg_pool2d(d, k, stride=1, padding=k // 2).squeeze(1)
    return torch.exp(-d / tau)


def align_logdepth(pred_depth, target, w):
    """Per-image weighted affine alignment of log(pred) to the label space."""
    # clamp BOTH sides: bf16 forward can emit +Inf; log(Inf) once poisoned all
    # 269 trainable tensors in a single step (iter-2800 NaN crash)
    lp = torch.log(torch.clamp(pred_depth, 1e-2, 1e4))
    B = lp.shape[0]
    flat_lp, flat_t, flat_w = lp.flatten(1), target.flatten(1), w.flatten(1)
    sw = flat_w.sum(1, keepdim=True) + 1e-6
    mw = flat_w / sw
    mlp = (mw * flat_lp).sum(1, keepdim=True)
    mt = (mw * flat_t).sum(1, keepdim=True)
    cov = (mw * (flat_lp - mlp) * (flat_t - mt)).sum(1, keepdim=True)
    var = (mw * (flat_lp - mlp) ** 2).sum(1, keepdim=True) + 1e-6
    a = cov / var
    b = mt - a * mlp
    a, b = a.view(-1, 1, 1), b.view(-1, 1, 1)  # (B,1,1) against (B,H,W)
    return a * lp + b, lp


def l_ssi(pred_norm, t1, w):
    return ((pred_norm - t1).abs() * w).mean()


def l_grad(pred_norm, t1, w, scales=(1.0, 0.5, 0.25)):
    total = 0.0
    for s in scales:
        if s == 1.0:
            p, t, ww = pred_norm, t1, w
        else:
            p = F.avg_pool2d(pred_norm.unsqueeze(1), 2, 2).squeeze(1)
            t = F.avg_pool2d(t1.unsqueeze(1), 2, 2).squeeze(1)
            ww = F.avg_pool2d(w.unsqueeze(1), 2, 2).squeeze(1)
        pgx, pgy = sobel(p)
        tgx, tgy = sobel(t)
        # +1e-8 inside sqrt: sqrt(0) has Inf gradient, and clip_grad_norm turns
        # a single Inf grad into NaN across ALL params (Inf * coef0 = NaN)
        pm = (pgx ** 2 + pgy ** 2 + 1e-8).sqrt()
        tm = (tgx ** 2 + tgy ** 2 + 1e-8).sqrt()
        mag = (pm - tm).abs()
        dot = (pgx * tgx + pgy * tgy) / (pm * tm + 1e-6)
        total = total + (mag * ww).mean() + (1.0 - dot.clamp(-1, 1)).mean()
    return total / len(scales)


def l_edge(pred_norm, t1, w, npairs=256, offset=3):
    """Boundary contrastive: match signed depth drops across T1 edges."""
    B, H, Wd = t1.shape
    loss = pred_norm.new_zeros(())
    gx, gy = sobel(t1)
    mag = (gx ** 2 + gy ** 2).sqrt()
    q = torch.quantile(mag.flatten(1), 0.9, dim=1) + 1e-6
    edges = mag > q[:, None, None]
    for b in range(B):
        ys, xs = torch.nonzero(edges[b], as_tuple=True)
        if ys.numel() < 8:
            continue
        k = min(npairs, ys.numel())
        sel = torch.randint(0, ys.numel(), (k,), device=t1.device)
        ys, xs = ys[sel], xs[sel]
        nvec = torch.stack([gx[b, ys, xs], gy[b, ys, xs]], 1)
        nvec = nvec / (nvec.norm(dim=1, keepdim=True) + 1e-6)
        o = torch.round(nvec * offset).long()
        yp, xp = (ys + o[:, 0]).clamp(0, H - 1), (xs + o[:, 1]).clamp(0, Wd - 1)
        ym, xm = (ys - o[:, 0]).clamp(0, H - 1), (xs - o[:, 1]).clamp(0, Wd - 1)
        ok = (w[b, yp, xp] > 0.5) & (w[b, ym, xm] > 0.5)
        if ok.sum() < 8:
            continue
        td = (t1[b, yp, xp] - t1[b, ym, xm])[ok]
        pd = (pred_norm[b, yp, xp] - pred_norm[b, ym, xm])[ok]
        loss = loss + (pd - td).abs().mean()
    return loss / max(B, 1)


def l_feat(s_aux, t_aux, projectors, pairs):
    """LAPTOP-style normalized L1 between student/teacher spatial features.
    pairs: list of (student_key, teacher_key); projectors keyed by student_key."""
    total, n = 0.0, 0
    for sk, tk in pairs:
        if sk not in s_aux or tk not in t_aux or sk not in projectors:
            continue
        sf = projectors[sk](s_aux[sk].float())
        tf = t_aux[tk].float()
        if sf.dim() == 5:  # (1,N,h,w,C) -> (N,h,w,C)
            sf = sf[0]; tf = tf[0]
        sf = sf / (sf.norm(dim=-1, keepdim=True) + 1e-6)
        tf = tf / (tf.norm(dim=-1, keepdim=True) + 1e-6)
        total = total + (sf - tf).abs().mean()
        n += 1
    return total / max(n, 1)


def l_conf(conf, resid, weight=0.1):
    """Confidence-as-correctness calibration (BCE on residual-median split)."""
    if conf is None:
        return conf.new_zeros(()) if torch.is_tensor(conf) else torch.zeros((), device=resid.device)
    c = torch.sigmoid(conf)
    B = resid.shape[0]
    med = resid.flatten(1).median(dim=1).values[:, None, None]
    y = (resid < med).float()
    bce = -(y * torch.log(c.clamp(1e-4, 1)) + (1 - y) * torch.log((1 - c).clamp(1e-4, 1)))
    return weight * bce.mean()
