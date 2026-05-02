# RL-Guided Inference for Text-to-Image Diffusion

CS748 Project — IIT Bombay  
Harshitha Inampudi (23B1071) & Chavali Daya Swaroop (22B0939)

## Overview

This codebase combines methods from multiple papers to optimize inference-time
control of text-to-image diffusion models using reinforcement learning:

| Paper | Method | What We Use |
|-------|--------|-------------|
| [2] DPOK (Lee et al., 2023) | RL fine-tuning with KL reg | REINFORCE training loop, LoRA, log-prob tracking |
| [3] RLG (Luo et al., 2025) | RL Guidance via model blending | Base vs LoRA model output interpolation |
| [4] Miao et al., 2024 | Diversity reward | CLIP feature-space diversity score |
| [5] Papalampidi et al., 2025 | Dynamic CFG via online feedback | Per-step greedy γ search with latent CLIP |
| [6] CLIP (Radford et al., 2021) | Text-image alignment | CLIP cosine similarity reward |
| [7] LPIPS (Zhang et al., 2018) | Perceptual quality | LPIPS quality score |

**Key idea:** A lightweight policy network π_ϕ learns to output per-step
guidance scale (γ_t) and stochasticity (η_t), trained via REINFORCE with
a composite reward. The U-Net is also fine-tuned with LoRA.

## Setup

```bash
# Create conda environment
conda create -n rldiff python=3.10 -y
conda activate rldiff

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

The first run downloads the external model weights unless they are already in
your local caches:

- `runwayml/stable-diffusion-v1-5` from Hugging Face, used by
  `pipeline_extended.py`
- OpenCLIP ViT-B/32 and LPIPS VGG weights, used by `reward_model.py`

Run training before evaluation, or explicitly opt into untrained evaluation:
`python evaluate.py --checkpoint ./checkpoints/best --allow-untrained`.
Evaluation without `--allow-untrained` expects a saved `policy.pt` and `lora/`
directory under the checkpoint path.

On machines without CUDA, the pipeline automatically uses fp32 even though the
default config requests fp16 for GPU memory savings.

### Train (single prompt, fast test)
```bash
python train.py --prompt "A green colored rabbit." --max-steps 100
```

### Train (full, ~2-3 hours on 5070 Ti)
```bash
python train.py
```

### Train with ablations (disable specific methods)
```bash
# No dynamic CFG [5]
python train.py --no-dynamic-cfg

# No RLG blending [3]
python train.py --no-rlg

# No diversity reward [4]
python train.py --no-diversity

# No policy network (LoRA-only, like vanilla DPOK)
python train.py --no-policy
```

### Evaluate
```bash
# Basic evaluation with comparisons
python evaluate.py --checkpoint ./checkpoints/best

# Three-stage progress comparison:
#   1. fixed baseline before RL control
#   2. dense RL-controlled diffusion
#   3. sparse RL intervention on top-K denoising steps
python evaluate.py --checkpoint ./checkpoints/best --sparse-top-k 8 --num-seeds 1

# Full ablation study
python evaluate.py --checkpoint ./checkpoints/best --ablation
```

### Validate the Theory
```bash
# Lightweight synthetic MDP test for the proof claims
python3 proof_validation.py
```

This writes reproducible proof-validation artifacts to `results/proof_validation/`:

| Claim | What the script checks |
|-------|------------------------|
| Expressivity gap | State-conditioned policy reward is strictly larger than the best time-only schedule. |
| Sparse intervention | The threshold rule `Delta_t > epsilon / T` gives a `K<T` policy whose error is at most `epsilon`. |
| Guaranteed gain | Top-`K` cumulative gains match the computable lower-bound construction when interference is zero. |
| Diminishing returns | Sorted marginal gains are non-increasing and identify the saturation budget `K*`. |

Default result on this repo:

| Metric | Value |
|--------|-------|
| State-conditioned reward | `1.0000` |
| Best time-only reward | `0.5016` |
| Observed expressivity gap | `0.4984` |
| Sparse budget `K_epsilon` | `18 / 20` |
| Sparse error | `0.008625 <= epsilon 0.1` |
| Saturation budget `K*` | `18` |

Raw outputs:
- `results/proof_validation/proof_validation_results.json`
- `results/proof_validation/step_values.csv`
- `results/proof_validation/proof_validation.svg`
- `results/proof_validation/proof_validation.png` when `matplotlib` is installed

## Output Structure

```
logs/              # TensorBoard logs (run: tensorboard --logdir logs)
checkpoints/       # Model weights (policy + LoRA)
samples/           # Generated images during training
results/           # Final evaluation results
  comparison_*.png # Three-way: baseline vs dense RL vs sparse RL
  comparison_*_labeled.png # Labeled three-way comparisons with CLIP scores
  schedules.png    # Learned γ_t, η_t curves
  clip_comparison.png  # Bar chart of CLIP scores
  three_stage_comparison_results.json # Raw baseline/dense/sparse metrics
  three_stage_summary.json # Mean deltas and sparse settings
  three_stage_scores.csv # Per-prompt plotting table
  schedules_long.csv # Per-prompt, per-step gamma/eta table
  sparse_intervention_mask.csv # Per-step sparse gate table
  plotting_arrays.npz # Numpy arrays for notebooks
  ablation.png     # Ablation study results
  proof_validation/     # Synthetic proof validation JSON/CSV/plot
```

## VRAM Notes (12GB GPU)

The code is configured for 12GB VRAM:
- SD 1.5 in fp16 (~3.5GB)
- LoRA rank 4 (minimal overhead)
- Batch size 2 for generation
- 20 denoising steps (not 50)
- CLIP ViT-B/32 (not L/14)

If you hit OOM, reduce `gen_batch_size` to 1 in `config.py`.

## Architecture

```
x_t ──► U-Net_θ ──► ε̂_θ ──► Sampler f ──► x_{t-1}
 │         ▲                     ▲
 │      Text c, t            γ_t, η_t
 │                               │
 └──────► Policy π_ϕ ───────────┘
          (MLP controller)
```

The policy observes the current latent state and timestep, outputs
adjustments to guidance and noise parameters. The U-Net remains
primarily frozen (with small LoRA updates). During training, LoRA receives a
REINFORCE signal through stochastic DDIM transition log-probabilities; the
policy network receives the action log-probability signal.
