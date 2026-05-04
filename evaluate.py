"""
Evaluation & Results Generation.

Generates:
1. Comparison images: fixed baseline vs dense RL vs sparse RL
2. CLIP score comparison table
3. Dense/sparse gamma/eta schedule plots
4. Ablation results (with/without each method)

Usage:
    python evaluate.py --checkpoint ./checkpoints/best
    python evaluate.py --checkpoint ./checkpoints/best --sparse-top-k 8
    python evaluate.py --checkpoint ./checkpoints/best --ablation
"""

import os
import json
import argparse
import csv
from contextlib import contextmanager
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from config import Config
from reward_model import CompositeReward
from policy_network import GuidancePolicy
from pipeline_extended import RLGuidedPipeline


@contextmanager
def temporary_config(config, **overrides):
    """Temporarily override config flags for one generation stage."""
    old_values = {key: getattr(config, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(config, key, value)
        yield
    finally:
        for key, value in old_values.items():
            setattr(config, key, value)


def get_sparse_intervention_steps(num_steps, top_k, strategy):
    """Return 0-indexed denoising step indices used by sparse intervention."""
    top_k = max(0, min(top_k, num_steps))
    if top_k == 0:
        return []
    if strategy == "uniform":
        return sorted(set(np.linspace(0, num_steps - 1, top_k, dtype=int).tolist()))
    if strategy != "early":
        raise ValueError(f"Unknown sparse intervention strategy: {strategy}")
    return list(range(top_k))


def load_checkpoint(pipeline, policy, checkpoint_dir, config, allow_untrained=False):
    """Load saved model weights."""
    ckpt = Path(checkpoint_dir)
    
    if policy is not None:
        if (ckpt / "policy.pt").exists():
            policy.load_state_dict(torch.load(ckpt / "policy.pt", map_location=pipeline.device))
            print(f"Loaded policy from {ckpt / 'policy.pt'}")
        elif not allow_untrained:
            raise FileNotFoundError(
                f"Policy checkpoint not found at {ckpt / 'policy.pt'}. "
                "Use --allow-untrained only if you intentionally want random policy results."
            )
    
    if config.use_lora:
        if (ckpt / "lora").exists():
            from peft import PeftModel
            pipeline.unet = PeftModel.from_pretrained(
                pipeline.unet,
                ckpt / "lora",
                is_trainable=False,
            ).to(pipeline.device)
            pipeline.unet.eval()
            print(f"Loaded LoRA from {ckpt / 'lora'}")
        elif not allow_untrained:
            raise FileNotFoundError(
                f"LoRA checkpoint not found at {ckpt / 'lora'}. "
                "Use --allow-untrained only if you intentionally want base U-Net results."
            )
    
    return pipeline, policy


def _score_images(reward_model, images, prompt):
    rewards = reward_model.compute_terminal_reward(images, [prompt])
    return {
        "clip": rewards["clip"].item(),
        "total": rewards["total"].item() if "total" in rewards else rewards["clip"].item(),
    }


def save_labeled_comparison(row, output_path):
    """Save a labeled three-stage image comparison for one prompt."""
    stages = [
        ("Baseline", row["baseline_img"], row["baseline_clip"]),
        ("Dense RL", row["dense_rl_img"], row["dense_rl_clip"]),
        ("Sparse RL", row["sparse_rl_img"], row["sparse_rl_clip"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, image, clip_score) in zip(axes, stages):
        img = image[0].detach().float().cpu()
        img = ((img + 1.0) / 2.0).clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"{title}\nCLIP={clip_score:.4f}", fontsize=10)
        ax.axis("off")
    fig.suptitle(row["prompt"], fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_comparison(pipeline, policy, reward_model, prompts, config, output_dir,
                        num_seeds=1):
    """Generate three-way comparison: baseline, dense RL, sparse RL."""
    print("\n=== Generating Comparisons ===")
    
    results = []
    sparse_steps = get_sparse_intervention_steps(
        config.num_inference_steps,
        config.sparse_intervention_top_k,
        config.sparse_intervention_strategy,
    )
    print(f"  Sparse intervention steps: {[s + 1 for s in sparse_steps]}")
    
    cases = [
        (prompt_idx, prompt, seed_idx, config.seed + seed_idx)
        for seed_idx in range(num_seeds)
        for prompt_idx, prompt in enumerate(prompts)
    ]

    for i, (prompt_idx, prompt, seed_idx, seed) in enumerate(tqdm(cases, desc="Generating")):
        row = {
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "seed_idx": seed_idx,
            "seed": seed,
            "sparse_steps_1_indexed": [s + 1 for s in sparse_steps],
        }
        
        # Stage 1: original fixed diffusion baseline, before RL-controlled inference.
        with torch.no_grad():
            with temporary_config(
                config,
                use_policy_network=False,
                use_dynamic_cfg=False,
                use_rlg_blending=False,
                use_lora=False,
            ):
                result_baseline = pipeline.generate_with_trajectory(
                    prompts=[prompt],
                    policy=None,
                    reward_model=None,
                    seed=seed,
                )
            baseline_scores = _score_images(reward_model, result_baseline["images"], prompt)
            row["baseline_clip"] = baseline_scores["clip"]
            row["baseline_total"] = baseline_scores["total"]
            row["baseline_img"] = result_baseline["images"]
        
        # Stage 2: dense RL-controlled diffusion, before sparse intervention.
        with torch.no_grad():
            result_dense = pipeline.generate_with_trajectory(
                prompts=[prompt],
                policy=policy,
                reward_model=reward_model if config.use_dynamic_cfg else None,
                seed=seed,
            )
            dense_scores = _score_images(reward_model, result_dense["images"], prompt)
            row["dense_rl_clip"] = dense_scores["clip"]
            row["dense_rl_total"] = dense_scores["total"]
            row["dense_rl_img"] = result_dense["images"]
            row["dense_gammas"] = [g.mean().item() for g in result_dense["gammas"]]
            row["dense_etas"] = [e.mean().item() for e in result_dense["etas"]]
        
        # Stage 3: sparse intervention, using the same RL controller only at top-K steps.
        with torch.no_grad():
            result_sparse = pipeline.generate_with_trajectory(
                prompts=[prompt],
                policy=policy,
                reward_model=reward_model if config.use_dynamic_cfg else None,
                seed=seed,
                intervention_steps=sparse_steps,
            )
            sparse_scores = _score_images(reward_model, result_sparse["images"], prompt)
            row["sparse_rl_clip"] = sparse_scores["clip"]
            row["sparse_rl_total"] = sparse_scores["total"]
            row["sparse_rl_img"] = result_sparse["images"]
            row["sparse_gammas"] = [g.mean().item() for g in result_sparse["gammas"]]
            row["sparse_etas"] = [e.mean().item() for e in result_sparse["etas"]]
            row["sparse_intervention_mask"] = result_sparse["intervention_mask"]
        
        results.append(row)
        
        # Save individual three-stage comparison.
        comparison = torch.cat([
            result_baseline["images"],
            result_dense["images"],
            result_sparse["images"],
        ], dim=0)
        save_image(
            comparison, 
            Path(output_dir) / f"comparison_{i}.png",
            nrow=3, normalize=True, value_range=(-1, 1)
        )
        save_labeled_comparison(row, Path(output_dir) / f"comparison_{i}_labeled.png")
    
    return results


def plot_schedules(results, output_dir):
    """Plot the learned γ_t and η_t schedules across timesteps."""
    print("\n=== Plotting Schedules ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gamma schedules
    ax = axes[0]
    for i, row in enumerate(results):
        if "dense_gammas" in row:
            ax.plot(row["dense_gammas"], alpha=0.45, linestyle="-",
                    label="Dense: " + row["prompt"][:24] + "...")
        if "sparse_gammas" in row:
            ax.plot(row["sparse_gammas"], alpha=0.55, linestyle="--",
                    label="Sparse: " + row["prompt"][:23] + "...")
    ax.axhline(y=7.5, color="red", linestyle="--", label="Default CFG=7.5")
    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Guidance Scale γ_t")
    ax.set_title("Learned Guidance Schedule")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    
    # Eta schedules
    ax = axes[1]
    for i, row in enumerate(results):
        if "dense_etas" in row:
            ax.plot(row["dense_etas"], alpha=0.45, linestyle="-",
                    label="Dense: " + row["prompt"][:24] + "...")
        if "sparse_etas" in row:
            ax.plot(row["sparse_etas"], alpha=0.55, linestyle="--",
                    label="Sparse: " + row["prompt"][:23] + "...")
    ax.axhline(y=0.0, color="red", linestyle="--", label="Default η=0.0")
    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Stochasticity η_t")
    ax.set_title("Learned Stochasticity Schedule")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "schedules.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved schedule plot to {output_dir}/schedules.png")


def plot_clip_comparison(results, output_dir):
    """Bar chart comparing CLIP scores: baseline vs dense RL vs sparse RL."""
    print("\n=== Plotting CLIP Comparison ===")
    
    prompts_short = [r["prompt"][:25] + "..." for r in results]
    baseline_clips = [r["baseline_clip"] for r in results]
    dense_clips = [r["dense_rl_clip"] for r in results]
    sparse_clips = [r["sparse_rl_clip"] for r in results]
    
    x = np.arange(len(prompts_short))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, baseline_clips, width, label="Baseline (Fixed CFG)", 
                   color="#4A90D9", alpha=0.8)
    ax.bar(x, dense_clips, width, label="Dense RL-Controlled",
                   color="#E74C3C", alpha=0.8)
    ax.bar(x + width, sparse_clips, width, label="Sparse RL Intervention",
                   color="#2ECC71", alpha=0.8)
    
    ax.set_xlabel("Prompt")
    ax.set_ylabel("CLIP Score")
    ax.set_title("CLIP Score: Baseline vs Dense RL vs Sparse RL")
    ax.set_xticks(x)
    ax.set_xticklabels(prompts_short, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add improvement annotation
    mean_baseline = np.mean(baseline_clips)
    mean_dense = np.mean(dense_clips)
    mean_sparse = np.mean(sparse_clips)
    dense_improvement = (mean_dense - mean_baseline) / abs(mean_baseline) * 100
    sparse_improvement = (mean_sparse - mean_baseline) / abs(mean_baseline) * 100
    ax.text(0.02, 0.98, 
            f"Mean: Base={mean_baseline:.4f}, Dense={mean_dense:.4f} "
            f"({dense_improvement:+.1f}%), Sparse={mean_sparse:.4f} "
            f"({sparse_improvement:+.1f}%)",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "clip_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved CLIP comparison to {output_dir}/clip_comparison.png")


def save_results_json(results, output_dir):
    """Save scalar comparison data without image tensors."""
    serializable = []
    for row in results:
        serializable.append({
            key: value for key, value in row.items()
            if not torch.is_tensor(value)
        })
    with open(Path(output_dir) / "three_stage_comparison_results.json", "w") as f:
        json.dump(serializable, f, indent=2)


def save_plotting_artifacts(results, config, output_dir):
    """Save flat CSV/NPZ/JSON files for plotting and paper tables."""
    output_dir = Path(output_dir)
    if not results:
        raise ValueError("No evaluation results were produced; cannot save plotting artifacts.")

    score_rows = []
    schedule_rows = []
    mask_rows = []
    for row_idx, row in enumerate(results):
        prompt_idx = row.get("prompt_idx", row_idx)
        dense_delta = row["dense_rl_clip"] - row["baseline_clip"]
        sparse_delta = row["sparse_rl_clip"] - row["baseline_clip"]
        sparse_vs_dense = row["sparse_rl_clip"] - row["dense_rl_clip"]
        score_rows.append({
            "prompt_idx": prompt_idx,
            "prompt": row["prompt"],
            "seed_idx": row.get("seed_idx", 0),
            "seed": row.get("seed"),
            "baseline_clip": row["baseline_clip"],
            "dense_rl_clip": row["dense_rl_clip"],
            "sparse_rl_clip": row["sparse_rl_clip"],
            "baseline_total": row["baseline_total"],
            "dense_rl_total": row["dense_rl_total"],
            "sparse_rl_total": row["sparse_rl_total"],
            "dense_minus_baseline_clip": dense_delta,
            "sparse_minus_baseline_clip": sparse_delta,
            "sparse_minus_dense_clip": sparse_vs_dense,
            "sparse_top_k": config.sparse_intervention_top_k,
            "sparse_strategy": config.sparse_intervention_strategy,
        })

        for step_idx, (dense_gamma, dense_eta, sparse_gamma, sparse_eta, is_intervention) in enumerate(
            zip(
                row["dense_gammas"],
                row["dense_etas"],
                row["sparse_gammas"],
                row["sparse_etas"],
                row["sparse_intervention_mask"],
            )
        ):
            schedule_rows.append({
                "prompt_idx": prompt_idx,
                "prompt": row["prompt"],
                "seed_idx": row.get("seed_idx", 0),
                "seed": row.get("seed"),
                "step_idx": step_idx,
                "step_1_indexed": step_idx + 1,
                "dense_gamma": dense_gamma,
                "dense_eta": dense_eta,
                "sparse_gamma": sparse_gamma,
                "sparse_eta": sparse_eta,
                "sparse_intervention": int(is_intervention),
            })
            mask_rows.append({
                "prompt_idx": prompt_idx,
                "seed_idx": row.get("seed_idx", 0),
                "seed": row.get("seed"),
                "step_idx": step_idx,
                "step_1_indexed": step_idx + 1,
                "sparse_intervention": int(is_intervention),
            })

    with open(output_dir / "three_stage_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
        writer.writeheader()
        writer.writerows(score_rows)

    with open(output_dir / "schedules_long.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(schedule_rows[0].keys()))
        writer.writeheader()
        writer.writerows(schedule_rows)

    with open(output_dir / "sparse_intervention_mask.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mask_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mask_rows)

    baseline = np.array([row["baseline_clip"] for row in results])
    dense = np.array([row["dense_rl_clip"] for row in results])
    sparse = np.array([row["sparse_rl_clip"] for row in results])
    dense_gammas = np.array([row["dense_gammas"] for row in results])
    sparse_gammas = np.array([row["sparse_gammas"] for row in results])
    dense_etas = np.array([row["dense_etas"] for row in results])
    sparse_etas = np.array([row["sparse_etas"] for row in results])
    sparse_mask = np.array([row["sparse_intervention_mask"] for row in results], dtype=np.int32)
    np.savez(
        output_dir / "plotting_arrays.npz",
        baseline_clip=baseline,
        dense_rl_clip=dense,
        sparse_rl_clip=sparse,
        dense_gammas=dense_gammas,
        sparse_gammas=sparse_gammas,
        dense_etas=dense_etas,
        sparse_etas=sparse_etas,
        sparse_intervention_mask=sparse_mask,
    )

    summary = {
        "num_prompts": len(results),
        "num_unique_prompts": len({row["prompt"] for row in results}),
        "num_seeds": len({row.get("seed_idx", 0) for row in results}),
        "num_inference_steps": config.num_inference_steps,
        "sparse_top_k": config.sparse_intervention_top_k,
        "sparse_strategy": config.sparse_intervention_strategy,
        "sparse_gate_dynamic_cfg": config.sparse_gate_dynamic_cfg,
        "sparse_gate_rlg": config.sparse_gate_rlg,
        "mean_baseline_clip": float(baseline.mean()),
        "mean_dense_rl_clip": float(dense.mean()),
        "mean_sparse_rl_clip": float(sparse.mean()),
        "mean_dense_minus_baseline_clip": float((dense - baseline).mean()),
        "mean_sparse_minus_baseline_clip": float((sparse - baseline).mean()),
        "mean_sparse_minus_dense_clip": float((sparse - dense).mean()),
        "std_baseline_clip": float(baseline.std()),
        "std_dense_rl_clip": float(dense.std()),
        "std_sparse_rl_clip": float(sparse.std()),
    }
    with open(output_dir / "three_stage_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved plotting data to {output_dir}/three_stage_scores.csv")
    print(f"  Saved schedules to {output_dir}/schedules_long.csv")
    print(f"  Saved arrays to {output_dir}/plotting_arrays.npz")


def score_prompt_set(pipeline, policy, reward_model, prompts, config, output_dir,
                     split_name, num_seeds=1):
    """Generate one image per prompt/seed and compute full reward breakdown."""
    results = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"Scoring {split_name}")):
        for seed_idx in range(num_seeds):
            seed = config.seed + seed_idx
            use_policy = policy if config.use_policy_network else None
            use_reward = reward_model if config.use_dynamic_cfg else None

            with torch.no_grad():
                result = pipeline.generate_with_trajectory(
                    prompts=[prompt],
                    policy=use_policy,
                    reward_model=use_reward,
                    seed=seed,
                )

            rewards = reward_model.compute_terminal_reward(result["images"], [prompt])
            filename = f"{split_name}_prompt_{prompt_idx}_seed_{seed}.png"
            save_image(
                result["images"],
                output_dir / filename,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

            results.append({
                "split": split_name,
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "seed_idx": seed_idx,
                "seed": seed,
                "filename": filename,
                "clip_score": rewards["clip"].item(),
                "lpips_score": rewards["lpips"].item(),
                "diversity_score": rewards["diversity"].item(),
                "total_score": rewards["total"].item(),
            })

    return results


def print_score_results(results, header):
    print("\n--- " + header + " Evaluation Summary ---")
    for res in results:
        print(f"Prompt: {res['prompt']}")
        print(f"  Generated Image: {res['filename']}")
        print(f"  CLIP Score: {res['clip_score']:.4f}")
        print(f"  LPIPS Score: {res['lpips_score']:.4f}")
        print(f"  Diversity Score: {res['diversity_score']:.4f}")
        print(f"  Total Modified Score: {res['total_score']:.4f}")
        print("-" * 30)


def run_ablation(pipeline, policy, reward_model, prompts, config, output_dir):
    """Run ablation study: disable each method and measure impact."""
    print("\n=== Running Ablation Study ===")
    
    ablation_configs = [
        ("Full Model", {}),
        ("No Dynamic CFG [5]", {"use_dynamic_cfg": False}),
        ("No RLG [3]", {"use_rlg_blending": False}),
        ("No Diversity [4]", {"lambda_diversity": 0.0}),
        ("No Policy Network", {"use_policy_network": False}),
        ("Baseline (all off)", {
            "use_dynamic_cfg": False, "use_rlg_blending": False,
            "lambda_diversity": 0.0, "use_policy_network": False
        }),
    ]
    
    ablation_results = {}
    
    for name, overrides in ablation_configs:
        print(f"\n  Testing: {name}")
        
        # Apply overrides
        for k, v in overrides.items():
            setattr(config, k, v)
        
        clips = []
        for prompt in tqdm(prompts, desc=f"  {name}", leave=False):
            use_policy = policy if config.use_policy_network else None
            use_reward = reward_model if config.use_dynamic_cfg else None
            
            with torch.no_grad():
                result = pipeline.generate_with_trajectory(
                    prompts=[prompt],
                    policy=use_policy,
                    reward_model=use_reward,
                    seed=config.seed,
                )
                rewards = reward_model.compute_terminal_reward(
                    result["images"], [prompt]
                )
                clips.append(rewards["clip"].item())
        
        ablation_results[name] = {
            "mean_clip": np.mean(clips),
            "std_clip": np.std(clips),
            "clips": clips,
        }
        print(f"    CLIP: {np.mean(clips):.4f} ± {np.std(clips):.4f}")
        
        # Restore config
        config_default = Config()
        for k in overrides:
            setattr(config, k, getattr(config_default, k))
    
    # Plot ablation results
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(ablation_results.keys())
    means = [ablation_results[n]["mean_clip"] for n in names]
    stds = [ablation_results[n]["std_clip"] for n in names]
    
    colors = ["#2ECC71", "#3498DB", "#9B59B6", "#E67E22", "#E74C3C", "#95A5A6"]
    bars = ax.barh(names, means, xerr=stds, color=colors[:len(names)], alpha=0.8)
    ax.set_xlabel("Mean CLIP Score")
    ax.set_title("Ablation Study: Contribution of Each Method")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Save raw results
    with open(Path(output_dir) / "ablation_results.json", "w") as f:
        json.dump({k: {"mean": v["mean_clip"], "std": v["std_clip"]} 
                  for k, v in ablation_results.items()}, f, indent=2)
    
    print(f"\n  Saved ablation results to {output_dir}/ablation.png")
    return ablation_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best",
                       help="Path to checkpoint directory")
    parser.add_argument("--output", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--ablation", action="store_true",
                       help="Run full ablation study")
    parser.add_argument("--num-seeds", type=int, default=1,
                       help="Number of seeds for evaluation")
    parser.add_argument("--sparse-top-k", type=int, default=None,
                       help="Number of denoising steps where sparse RL can intervene")
    parser.add_argument("--sparse-strategy", type=str, default=None,
                       choices=["early", "uniform"],
                       help="How to choose sparse intervention steps")
    parser.add_argument("--no-sparse-gate-dynamic-cfg", action="store_true",
                       help="Let dynamic CFG run on every step even for sparse RL")
    parser.add_argument("--no-sparse-gate-rlg", action="store_true",
                       help="Let RLG blending run on every step even for sparse RL")
    parser.add_argument("--allow-untrained", action="store_true",
                       help="Allow evaluation with missing checkpoint pieces")
    parser.add_argument("--score-train-test", action="store_true",
                       help="Score train and eval prompts with full reward breakdown")
    parser.add_argument("--skip-comparisons", action="store_true",
                       help="Skip baseline/dense/sparse comparison plots and tables")
    args = parser.parse_args()
    if args.num_seeds < 1:
        parser.error("--num-seeds must be at least 1")
    
    config = Config()
    if args.sparse_top_k is not None:
        config.sparse_intervention_top_k = args.sparse_top_k
    if args.sparse_strategy is not None:
        config.sparse_intervention_strategy = args.sparse_strategy
    if args.no_sparse_gate_dynamic_cfg:
        config.sparse_gate_dynamic_cfg = False
    if args.no_sparse_gate_rlg:
        config.sparse_gate_rlg = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("Loading pipeline...")
    pipeline = RLGuidedPipeline(config, device=device)
    
    policy = None
    if config.use_policy_network:
        policy = GuidancePolicy(config).to(device)
    
    reward_model = CompositeReward(config, device=device)
    
    # Load checkpoint
    if Path(args.checkpoint).exists():
        pipeline, policy = load_checkpoint(
            pipeline, policy, args.checkpoint, config,
            allow_untrained=args.allow_untrained,
        )
    elif args.allow_untrained:
        print(f"WARNING: Checkpoint {args.checkpoint} not found, using untrained model")
    else:
        raise FileNotFoundError(
            f"Checkpoint directory {args.checkpoint} not found. "
            "Use --allow-untrained only if you intentionally want untrained results."
        )
    if policy is not None:
        policy.eval()
    
    # Shared prompt bundle for comparisons/ablation
    all_prompts = config.eval_prompts + config.train_prompts[:5]

    if args.score_train_test:
        train_scores = score_prompt_set(
            pipeline, policy, reward_model, config.train_prompts,
            config, output_dir / "scored", "train", num_seeds=args.num_seeds,
        )
        eval_scores = score_prompt_set(
            pipeline, policy, reward_model, config.eval_prompts,
            config, output_dir / "scored", "eval", num_seeds=args.num_seeds,
        )
        print_score_results(train_scores, "Train")
        print_score_results(eval_scores, "Eval")

        with open(output_dir / "train_prompt_scores.json", "w") as f:
            json.dump(train_scores, f, indent=2)
        with open(output_dir / "eval_prompt_scores.json", "w") as f:
            json.dump(eval_scores, f, indent=2)

    if not args.skip_comparisons:
        # Generate comparisons
        results = generate_comparison(
            pipeline, policy, reward_model, all_prompts, config, output_dir,
            num_seeds=args.num_seeds
        )
        save_results_json(results, output_dir)
        save_plotting_artifacts(results, config, output_dir)

        # Plot schedule
        plot_schedules(results, output_dir)

        # Plot CLIP comparison
        plot_clip_comparison(results, output_dir)

        # Summary table
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"{'Prompt':<34} {'Baseline':>9} {'Dense RL':>9} {'Sparse':>9} {'S-B':>8} {'S-D':>8}")
        print("-" * 86)
        for r in results:
            sparse_vs_base = r["sparse_rl_clip"] - r["baseline_clip"]
            sparse_vs_dense = r["sparse_rl_clip"] - r["dense_rl_clip"]
            print(f"{r['prompt'][:32]:<34} {r['baseline_clip']:>9.4f} "
                  f"{r['dense_rl_clip']:>9.4f} {r['sparse_rl_clip']:>9.4f} "
                  f"{sparse_vs_base:>+8.4f} {sparse_vs_dense:>+8.4f}")

        mean_b = np.mean([r["baseline_clip"] for r in results])
        mean_d = np.mean([r["dense_rl_clip"] for r in results])
        mean_s = np.mean([r["sparse_rl_clip"] for r in results])
        print("-" * 86)
        print(f"{'MEAN':<34} {mean_b:>9.4f} {mean_d:>9.4f} {mean_s:>9.4f} "
              f"{mean_s-mean_b:>+8.4f} {mean_s-mean_d:>+8.4f}")
    
    # Ablation
    if args.ablation:
        run_ablation(pipeline, policy, reward_model, all_prompts, config, output_dir)
    
    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
