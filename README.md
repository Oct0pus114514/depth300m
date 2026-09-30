# Depth300M — Lightweight Monocular Depth via Dual-Teacher Distillation

A **0.12B-parameter** monocular depth estimation model with strong boundary quality on
portraits and complex scenes. Fine-tuned from [Depth Anything 3](https://github.com/YvanYin530/Depth-Anything-3) (DA3-BASE)
by distilling two teachers, with the whole pipeline (single forward pass, no VAE, no text encoder)
comfortably inside a 300M budget.

## Method

**Student**: DA3-BASE (DINOv2 ViT-B + DualDPT head, ~120M), full-parameter fine-tune.

**Dual teachers**:

| Teacher | Role | Where it enters |
|---|---|---|
| T1 — Marigold V2 (20B DiT, offline pseudo-GT) | main label-space supervision (edge quality) | SSI-L1 / gradient / boundary-contrastive / confidence losses |
| T2 — DA3MONO-LARGE (0.35B, online forward) | consistency judge + feature anchor | weight map `w = exp(−avg₇ₓ₇\|T1−T2\|/τ)` on all pixel losses; LAPTOP-style normalized L1 on ViT features (student layers [2,5,8,11] ↔ teacher [5,11,17,23]) |

Loss: `1.0·L_ssi + 0.7·L_grad + 0.5·L_edge + 0.5·L_feat + L_conf`, single stage, 25K iters,
mixed-resolution crops (448–1008, all multiples of patch 14), EMA 0.999, bf16, AdamW lr 2e-5.

Data: COCO 2017 (118K) + PPR10K subset (3.0K, phone portrait domain, target-domain proxy),
both pseudo-labeled by T1 and T2 offline; per-image 2–98% quantile normalized log-depth in [-1,1].

## Results

**NYUv2 654** (official test, Eigen crop, per-image log-affine alignment):

| Model | Params | AbsRel mean ↓ | AbsRel median ↓ | δ₁ ↑ | RMSE ↓ |
|---|---|---|---|---|---|
| DA3MONO-LARGE (teacher T2) | 0.35B | 0.041 | 0.032 | 0.979 | — |
| DA3-BASE (init) | 0.12B | 0.056 | 0.045 | 0.962 | — |
| **Ours** | **0.12B** | **0.0581** | **0.0472** | **0.963** | 0.239 |

**HW-55** (internal phone EDOF portrait/scene set; Pearson corr of normalized log-depth vs
SD3-based reference):

| Model | corr median ↑ |
|---|---|
| Marigold V2 (teacher T1) | 0.827 |
| DA3MONO-LARGE (teacher T2) | 0.855 |
| DA3-BASE (init) | 0.883 |
| **Ours** | **0.903** |

The 0.12B student surpasses both teachers and its initialization on the target domain while
staying within +0.003 AbsRel of its init on NYU (guardrail: ≤ +0.005).

Full logs: `results/`.

## Architecture code & weights

**Model architecture code is vendored**: `da3_src/` contains the Depth Anything 3
sources (Apache-2.0, upstream commit `3d835ec`, `bench/app/services/cli` removed, and a
small patch marks `evo/pycolmap/trimesh/moviepy/gsplat` lazy — they are only needed for
GS / multi-view / mesh export, not mono depth). No separate DA3 clone is needed;
`DA3_SRC` env var overrides the location if you want to use your own copy.

**Weights** (not included except ours):

| Weight | Get from | Put at |
|---|---|---|
| DA3-BASE (student init / architecture container) | Depth-Anything-3 repo release links | `ckpt/model.safetensors` |
| DA3MONO-LARGE (teacher T2) | Depth-Anything-3 repo release links | `ckpt/DA3MONO-LARGE.safetensors` |
| **Ours (final EMA)** | **Hugging Face: `Oct0pus/depth300m`** (private repo) | `ckpt/depth300m_ema_iter25000.fp16.safetensors` |
| Marigold V2 Log-stage2 (teacher T1) | `huawei-bayerlab/marigold-v2` (HF) | only if you re-label data |

```bash
# download ours (private repo - pass your HF token; or `hf auth login` first)
# huggingface_hub >= 0.34 / 2.x renamed the CLI: use `hf` (older versions: `huggingface-cli`)
hf download Oct0pus/depth300m depth300m_ema_iter25000.fp16.safetensors \
    --local-dir ckpt --token $HF_TOKEN

# or from Python:
# from huggingface_hub import hf_hub_download
# hf_hub_download("Oct0pus/depth300m", "depth300m_ema_iter25000.fp16.safetensors",
#                 local_dir="ckpt", token=...)
```

Weights are NOT stored in this git repo on purpose (git-lfs pulls are unreliable
behind some networks; a ZIP download would give you pointer files, not weights).

Our fine-tuned weights are loaded on top of the DA3-BASE checkpoint
(`load_state_dict(..., strict=False)`); see `code/eval_*.py` for the exact recipe.

## Quickstart

Paths in the scripts default to our training server layout (`/root/autodl-tmp/...`); adjust the
constants at the top of each file for your machine.

```bash
# 1) environment (vendored DA3 code needs no extra install)
pip install -r requirements.txt

# 2) download DA3-BASE into ckpt/model.safetensors (table above)

# 3) inference on the bundled HW set (55 phone EDOF images, native aspect, res 1024)
python code/infer_model.py --ckpt ckpt/depth300m_ema_iter25000.fp16.safetensors --images data/hw --out infer_out

# 4) evaluations
python code/eval_nyu.py --ckpt ckpt/depth300m_ema_iter25000.fp16.safetensors --ema
python code/eval_hw.py --ckpt ... --ema

# 5) regenerate the 4-panel comparison demos (RGB | Marigold V2 | DA3-BASE | Ours)
python code/demo_hw_final.py
```

### Regenerating T1 pseudo-labels (optional)

T1 labels come from offline Marigold V2 inference; the marigold-v2 repo is NOT vendored
(no code here imports it). To re-label your own data:

```bash
git clone https://github.com/huawei-bayerlab/marigold-v2
python marigold-v2/scripts/infer.py --modality depth \
    --checkpoint <path>/Marigold-V2/depth/Log-stage2 \
    --image_dir <your_images> --output_dir <labels_out>   # native resolution
```

## Training reproduction

```bash
# pseudo-label your data first (T1 offline via marigold-v2 repo; T2 via code in this repo's
# history or the DA3MONO checkpoint), then:
python code/train.py --iters 25000 --batch 8 --accum 2 --lr 2e-5 --tau 0.3 \
    --eval_every 2000 --save_every 1000 --out runs/s1
```

Numerical-stability notes baked into the code (each one earned the hard way):
`forward_train` bypasses the DA3 facade's `@torch.inference_mode` decorator (otherwise zero
gradients reach the student); log-depth is clamped on both sides before `log`; `sqrt` terms
carry an epsilon (sqrt(0) has Inf gradient, and grad-clip converts a single Inf into global NaN);
optimizer steps are skipped when loss or grads are non-finite; `align_logdepth` fits in
float32 outside autocast.

## Repo structure

```
code/        training / inference / eval / demo scripts (single-file each)
da3_src/     vendored Depth-Anything-3 sources (Apache-2.0, pruned + lazy-patch, see above)
ckpt/        weights land here by download (ours from HF, DA3-BASE from its repo); git-ignored
data/hw/     the 55 HW RGB images (phone EDOF portraits/scenes) used by the quickstart
demo/        hw_final/ — 55 comparison sheets (RGB | Marigold V2 | DA3-BASE | Ours)
results/     final eval logs, baseline jsons, training log.jsonl
```

## Known limitations

- Transparent objects (glass, films) are predicted "see-through" — inherited from the whole
  supervision chain (both teachers and the SD3 reference behave the same). Fixing it needs
  transparent-surface GT data (e.g. ClearGrasp / TransCG), planned for a future round.
- HW-55 reference is itself model-generated; corr numbers are meaningful only relative to
  models evaluated under the same protocol.

## License & credits

- This repo: code under Apache-2.0; weights for research use.
- Depth Anything 3 (code & DA3-BASE / DA3MONO checkpoints): Apache-2.0.
- Marigold V2 (teacher pseudo-labels): Apache-2.0.
- Training data: COCO 2017 (CC-BY 4.0), PPR10K (MIT). HW-55 is internal.
