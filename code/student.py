"""Student (DA3-BASE) and teacher (DA3MONO-LARGE) wrappers for mono training."""
from __future__ import annotations
import os, sys, torch
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
# vendored DA3 sources ship with this repo (da3_src/, Apache-2.0, lightly patched
# to make pycolmap/evo/trimesh/moviepy optional); DA3_SRC env or the original
# training-server layout work as fallbacks.
for _p in (_REPO / "da3_src",
           Path(os.environ.get("DA3_SRC", "/nonexistent")),
           Path("/root/autodl-tmp/Depth-Anything-3/src")):
    if (_p / "depth_anything_3" / "api.py").exists():
        sys.path.insert(0, str(_p))
        break
from depth_anything_3.api import DepthAnything3
from safetensors.torch import load_file


def _resolve_ckpt_dir():
    env = os.environ.get("DA3_CKPT_DIR")
    if env:
        return env
    # fresh checkout: put downloaded DA3-BASE/DA3MONO safetensors into <repo>/ckpt/
    if (_REPO / "ckpt" / "model.safetensors").exists():
        return str(_REPO / "ckpt")
    return "/root/autodl-tmp/Depth-Anything-3/ckpt"  # training-server layout


DA3_CKPT = _resolve_ckpt_dir()


def _load(model_name, ckpt_file):
    m = DepthAnything3(model_name=model_name)
    sd = load_file(f"{DA3_CKPT}/{ckpt_file}")
    miss, unexp = m.load_state_dict(sd, strict=False)
    # official ckpts omit the DPT aux (training-only) branch -> tolerated
    real_miss = [k for k in miss if "output_conv2_aux" not in k]
    assert not real_miss and not unexp, (real_miss[:3], unexp[:3])
    return m


def load_sd(path, ema=True):
    """Accepts a training .pt ({model, ema, proj, iter}) or a raw safetensors state_dict."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(str(path))
    ck = torch.load(path, map_location="cpu")
    return ck["ema"] if ema else ck["model"]


def load_student():
    """DA3-BASE, trainable."""
    return _load("da3-base", "model.safetensors").cuda().train()


def load_teacher():
    """DA3MONO-LARGE, frozen feature anchor."""
    m = _load("da3mono-large", "DA3MONO-LARGE.safetensors").cuda().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def forward_train(model, imgs, feat_layers):
    """Differentiable mono forward. imgs: (N,3,H,W) normalized cuda tensor.

    Calls the inner network directly: the facade forward() is decorated with
    @torch.inference_mode() (api.py:99), which silently cut all gradients
    in the first 25K-iter attempt (weights stayed bit-identical to init).
    """
    return model.model(imgs[None], None, None, list(feat_layers), False, False, "saddle_balanced")


def probe(model, feat_layers=(2, 5, 8, 11)):
    """Introspect output keys / aux shapes. Run once at startup."""
    x = torch.randn(2, 3, 448, 448, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        out = forward_train(model, x, feat_layers)
    print("[probe] output keys:", list(out.keys()))
    for k in out.keys():
        v = out[k]
        if torch.is_tensor(v):
            print(f"[probe]  {k}: {tuple(v.shape)} {v.dtype}")
    aux = getattr(out, "aux", None)
    if aux:
        for k, v in aux.items():
            print(f"[probe] aux[{k}]: {tuple(v.shape)}")
    return out
