"""
RL-Guided Diffusion Inference Training.

Main training loop adapted from DPOK [2] (REINFORCE with KL regularization),
extended with:
  - Policy network for (γ_t, η_t) control [Proposal]
  - Composite reward (CLIP + LPIPS + Diversity) [4, 6, 7]
  - Dynamic CFG during rollouts [5]
  - RLG blending [3]

Usage:
    python train.py
    
    # Disable features for ablation:
    python train.py --no-dynamic-cfg --no-rlg --no-diversity
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import numpy as np

from config import Config
from reward_model import CompositeReward
from policy_network import GuidancePolicy
from pipeline_extended import RLGuidedPipeline


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_lora(pipeline, config):
    """Add LoRA adapters to U-Net (from DPOK [2])."""
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)
    
    # Count trainable params
    trainable = sum(p.numel() for p in pipeline.unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in pipeline.unet.parameters())
    print(f"LoRA: {trainable:,} trainable / {total:,} total params "
          f"({100*trainable/total:.2f}%)")
    
    return pipeline


def collect_rollout(pipeline, policy, reward_model, prompts, config, step):
    """Collect a batch of rollouts (trajectories) for RL training.
    
    This follows DPOK [2]'s online data collection:
    1. Sample prompts
    2. Generate images with current policy/model
    3. Compute rewards
    4. Store trajectories
    
    IMPORTANT: The policy forward passes run WITH gradients (needed for
    REINFORCE). When LoRA is enabled, the active U-Net also runs with
    gradients so DDIM transition log-probs can train the adapter. The
    VAE, reward model, and frozen base U-Net stay out of the graph.
    """
    # Select batch of prompts
    batch_prompts = random.sample(prompts, min(config.gen_batch_size, len(prompts)))
    
    # Generate with trajectory tracking
    # Policy runs with grad, everything else without
    result = generate_training_trajectory(
        pipeline, policy, reward_model, batch_prompts, config, 
        seed=config.seed + step
    )
    
    # Compute terminal reward (no grad needed for reward computation)
    images = result["images"]
    with torch.no_grad():
        reward_info = reward_model.compute_terminal_reward(images, batch_prompts)
        smoothness = CompositeReward.compute_smoothness_reward(result["latents_history"])
        # ``latents_history`` may now live on CPU (memory saver); align device
        # with the terminal reward so the addition below is well-defined.
        smoothness = smoothness.to(reward_info["total"].device)
    
    # Total reward
    total_reward = (
        reward_info["total"] + 
        config.lambda_smooth * smoothness
    )
    
    return {
        "prompts": batch_prompts,
        "images": images.detach(),
        "log_probs": result["log_probs"],  # has grad from policy
        "entropies": result["entropies"],   # has grad from policy
        "rewards": total_reward.detach(),   # detached (REINFORCE doesn't backprop through reward)
        "reward_info": {k: v.detach() for k, v in reward_info.items()},
        "gammas": result["gammas"],
        "etas": result["etas"],
        "smoothness": smoothness.detach(),
    }


def generate_training_trajectory(pipeline, policy, reward_model, prompts, config, seed=None):
    """Generate images for training with proper gradient handling.
    
    Key: Policy forward passes have gradients. If LoRA is enabled, the
    active U-Net forward and DDIM transition log-prob also keep gradients
    so the adapter receives a REINFORCE signal. Latents are detached
    between denoising steps to avoid backpropagating through the whole
    sampler trajectory.
    """
    B = len(prompts)
    device = pipeline.device
    
    # Encode prompts (no grad)
    with torch.no_grad():
        cond_embeds, uncond_embeds = pipeline.encode_prompt(prompts)
    
    # Initial noise
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    latent_shape = (B, 4, config.image_size // 8, config.image_size // 8)
    latents = torch.randn(latent_shape, device=device, 
                         dtype=cond_embeds.dtype, generator=generator)
    latents = latents * pipeline.scheduler.init_noise_sigma
    
    # Tracking. See pipeline.generate_with_trajectory for the rationale —
    # the smoothness reward is the only consumer of latents_history and runs
    # after the rollout completes, so the GPU copies are dead weight.
    history_to_cpu = getattr(config, "latent_history_to_cpu", False)
    latents_history = [
        latents.detach().to("cpu", non_blocking=True) if history_to_cpu
        else latents.detach().clone()
    ]
    total_log_prob = torch.zeros(B, device=device)
    total_entropy = torch.zeros(B, device=device)
    gamma_history = []
    eta_history = []
    
    timesteps = pipeline.scheduler.timesteps
    
    for step_idx, t in enumerate(timesteps):
        t_tensor = t.unsqueeze(0).to(device) if t.dim() == 0 else t.to(device)
        
        is_control_step = (step_idx % config.control_every_n_steps == 0)
        
        # === Policy forward (WITH gradient for REINFORCE) ===
        if policy is not None and config.use_policy_network and is_control_step:
            gamma_t, eta_t, log_prob, entropy = policy(
                latents.detach().float(),  # detach latent from U-Net graph
                t_tensor.expand(B)
            )
            total_log_prob = total_log_prob + log_prob
            total_entropy = total_entropy + entropy
        else:
            gamma_t = torch.full((B,), config.default_guidance_scale, device=device)
            eta_t = torch.full((B,), config.default_eta, device=device)
        
        train_lora = config.use_lora
        step_latents = latents.detach()

        # === U-Net forward + denoising step ===
        # Keep gradients only for the active LoRA U-Net. All reward/search/base
        # model work is still no-grad.
        with nullcontext() if train_lora else torch.no_grad():
            if config.use_dynamic_cfg and reward_model is not None:
                with torch.no_grad():
                    _, best_gamma = pipeline._dynamic_cfg_search(
                        step_latents, t_tensor, cond_embeds, uncond_embeds,
                        prompts, reward_model, config.cfg_candidates
                    )
                if policy is not None and is_control_step:
                    gamma_t_val = 0.5 * gamma_t.detach() + 0.5 * best_gamma
                    noise_pred, _, _ = pipeline._predict_noise(
                        step_latents, t_tensor, cond_embeds, uncond_embeds,
                        gamma_t_val.mean().item()
                    )
                    gamma_t = gamma_t_val
                else:
                    gamma_t_val = best_gamma
                    gamma_t = gamma_t_val
                    noise_pred, _, _ = pipeline._predict_noise(
                        step_latents, t_tensor, cond_embeds, uncond_embeds,
                        gamma_t_val.mean().item()
                    )
            else:
                mean_gamma = gamma_t.detach().mean().item()
                noise_pred, _, _ = pipeline._predict_noise(
                    step_latents, t_tensor, cond_embeds, uncond_embeds, mean_gamma
                )
            
            # RLG blending
            if config.use_rlg_blending and pipeline._base_unet is not None:
                with torch.no_grad():
                    noise_base = pipeline._predict_noise_base(
                        step_latents, t_tensor, cond_embeds, uncond_embeds,
                        gamma_t.detach().mean().item()
                    )
                noise_pred = pipeline._apply_rlg_blending(
                    noise_pred, noise_base, config.rlg_scale
                )
            
            # DDIM step
            mean_eta = eta_t.detach().mean().item()
            if train_lora and mean_eta <= 0:
                mean_eta = config.lora_train_eta
            next_latents, step_log_prob = pipeline._ddim_step_with_logprob(
                noise_pred, t_tensor, step_latents, eta=mean_eta,
                generator=generator
            )

            if train_lora:
                total_log_prob = total_log_prob + step_log_prob
            latents = next_latents.detach()
        
        latents_history.append(
            latents.detach().to("cpu", non_blocking=True) if history_to_cpu
            else latents.detach().clone()
        )
        gamma_history.append(gamma_t.detach().cpu())
        eta_history.append(eta_t.detach().cpu())
    
    # Decode final image (no grad)
    with torch.no_grad():
        images = pipeline.decode_latents(latents)
    
    return {
        "images": images,
        "latents_history": latents_history,
        "log_probs": total_log_prob,       # has grad through policy
        "gammas": gamma_history,
        "etas": eta_history,
        "entropies": total_entropy,         # has grad through policy
    }


def compute_policy_loss(rollout, config):
    """REINFORCE loss with KL regularization (from DPOK [2]).
    
    L = -E[R * log π(a|s)] + β * KL(π || π_ref)
    
    In DPOK, the KL term prevents the model from deviating too far
    from the pretrained weights. Here we apply it to both the
    policy network and the LoRA parameters.
    """
    log_probs = rollout["log_probs"]           # (B,)
    rewards = rollout["rewards"]                # (B,)
    entropies = rollout["entropies"]            # (B,)
    
    # Normalize rewards (variance reduction)
    if rewards.shape[0] > 1:
        rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    else:
        rewards_normalized = rewards
    
    # REINFORCE: -R * log π
    policy_loss = -(rewards_normalized * log_probs).mean()
    
    # KL regularization (DPOK [2]): encourage entropy to prevent collapse
    entropy_bonus = -config.kl_weight * entropies.mean()
    
    # Total
    total_loss = config.reward_weight * policy_loss + entropy_bonus
    
    return total_loss, {
        "policy_loss": policy_loss.item(),
        "entropy_bonus": entropy_bonus.item(),
        "total_loss": total_loss.item(),
        "mean_reward": rewards.mean().item(),
        "mean_entropy": entropies.mean().item(),
    }


def evaluate(pipeline, policy, reward_model, prompts, config, step, writer, sample_dir):
    """Generate evaluation images and compute metrics."""
    print(f"\n--- Evaluation at step {step} ---")
    
    all_images = []
    all_clip = []
    all_gammas = []
    
    for prompt in prompts:
        with torch.no_grad():
            # With policy
            result = pipeline.generate_with_trajectory(
                prompts=[prompt],
                policy=policy,
                reward_model=reward_model if config.use_dynamic_cfg else None,
                seed=config.seed,
            )
            
            images = result["images"]
            rewards = reward_model.compute_terminal_reward(images, [prompt])
            
            all_images.append(images)
            all_clip.append(rewards["clip"].item())
            all_gammas.append([g.mean().item() for g in result["gammas"]])
            
            # Also generate baseline (no policy, fixed CFG)
            result_baseline = pipeline.generate_with_trajectory(
                prompts=[prompt],
                policy=None,
                reward_model=None,
                seed=config.seed,
            )
            baseline_rewards = reward_model.compute_terminal_reward(
                result_baseline["images"], [prompt]
            )
    
    # Save sample images
    if all_images:
        grid = make_grid(torch.cat(all_images, dim=0), nrow=5, normalize=True, value_range=(-1, 1))
        save_path = Path(sample_dir) / f"step_{step}.png"
        save_image(grid, save_path)
        print(f"  Saved samples to {save_path}")
    
    # Log metrics
    mean_clip = np.mean(all_clip)
    print(f"  Mean CLIP score: {mean_clip:.4f}")
    
    if writer:
        writer.add_scalar("eval/mean_clip", mean_clip, step)
        if all_images:
            writer.add_image("eval/samples", grid, step)
    
    # Log gamma schedules
    if all_gammas:
        mean_gamma_schedule = np.mean(all_gammas, axis=0)
        print(f"  Mean γ schedule: {[f'{g:.1f}' for g in mean_gamma_schedule[:5]]}...")
        if writer:
            for t_idx, g in enumerate(mean_gamma_schedule):
                writer.add_scalar(f"eval/gamma_t{t_idx}", g, step)
    
    return mean_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dynamic-cfg", action="store_true", 
                       help="Disable dynamic CFG [5]")
    parser.add_argument("--no-rlg", action="store_true",
                       help="Disable RLG blending [3]")
    parser.add_argument("--no-diversity", action="store_true",
                       help="Disable diversity reward [4]")
    parser.add_argument("--no-policy", action="store_true",
                       help="Disable policy network (use only LoRA)")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Override max training steps")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Single prompt mode")
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Apply CLI overrides
    if args.no_dynamic_cfg:
        config.use_dynamic_cfg = False
    if args.no_rlg:
        config.use_rlg_blending = False
    if args.no_diversity:
        config.lambda_diversity = 0.0
    if args.no_policy:
        config.use_policy_network = False
    if args.max_steps:
        config.max_train_steps = args.max_steps
    if args.prompt:
        config.train_prompts = [args.prompt]
    
    # Print config summary
    print("=" * 60)
    print("RL-Guided Diffusion Inference Training")
    print("=" * 60)
    print(f"  Dynamic CFG [5]:     {'ON' if config.use_dynamic_cfg else 'OFF'}")
    print(f"  RLG Blending [3]:    {'ON' if config.use_rlg_blending else 'OFF'}")
    print(f"  Diversity Reward [4]: {'ON' if config.lambda_diversity > 0 else 'OFF'}")
    print(f"  Policy Network:      {'ON' if config.use_policy_network else 'OFF'}")
    print(f"  LoRA Fine-tuning [2]: {'ON' if config.use_lora else 'OFF'}")
    print(f"  Train prompts: {len(config.train_prompts)}")
    print(f"  Max steps: {config.max_train_steps}")
    print("=" * 60)
    
    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # === Setup ===
    # 1. Pipeline
    pipeline = RLGuidedPipeline(config, device=device)
    
    # 2. LoRA on U-Net (DPOK [2])
    if config.use_lora:
        pipeline = setup_lora(pipeline, config)
        # Gradient checkpointing only activates when ``model.training`` is True.
        # SD's U-Net has no dropout / no batchnorm (only GroupNorm), and the
        # LoRA dropout is 0.0 — so train vs eval is numerically identical for
        # this model. Flipping to train() costs nothing but lets the
        # checkpointing path run during the with-grad rollout forwards.
        if getattr(config, "gradient_checkpointing", False):
            pipeline.unet.train()
    
    # 3. Policy network (Proposal)
    policy = None
    if config.use_policy_network:
        policy = GuidancePolicy(config).to(device)
        print(f"Policy network: {sum(p.numel() for p in policy.parameters()):,} params")
    
    # 4. Reward model
    reward_model = CompositeReward(config, device=device)
    
    # 5. Optimizers
    param_groups = []
    if config.use_lora:
        lora_params = [p for p in pipeline.unet.parameters() if p.requires_grad]
        param_groups.append({"params": lora_params, "lr": config.lora_lr})
    if policy is not None:
        param_groups.append({"params": policy.parameters(), "lr": config.learning_rate})
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    
    # 6. Logging
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)
    writer = SummaryWriter(config.log_dir)
    
    # Save config
    with open(os.path.join(config.log_dir, "config.json"), "w") as f:
        json.dump({k: str(v) for k, v in vars(config).items()}, f, indent=2)
    
    # === Training Loop (DPOK [2] style REINFORCE) ===
    print("\nStarting training...")
    best_clip = -float("inf")
    
    for step in tqdm(range(1, config.max_train_steps + 1), desc="Training"):
        # Collect rollout
        if policy is not None:
            policy.eval()
        
        rollout = collect_rollout(
            pipeline, policy, reward_model,
            config.train_prompts, config, step
        )
        
        # Compute loss and update
        if policy is not None:
            policy.train()
        
        loss, loss_info = compute_policy_loss(rollout, config)
        
        # Backward + clip + step
        loss.backward()
        
        if (step % config.gradient_accumulation_steps) == 0:
            if config.clip_norm > 0:
                all_params = []
                if config.use_lora:
                    all_params.extend([p for p in pipeline.unet.parameters() if p.requires_grad])
                if policy is not None:
                    all_params.extend(policy.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, config.clip_norm)
            
            optimizer.step()
            optimizer.zero_grad()
        
        # Optional periodic VRAM defragmentation. ``empty_cache`` is itself
        # not free, so this is opt-in via config (default disabled).
        empty_every = getattr(config, "empty_cache_every_n_steps", 0)
        if empty_every and torch.cuda.is_available() and step % empty_every == 0:
            torch.cuda.empty_cache()
        
        # === Logging ===
        if step % config.log_every == 0:
            tqdm.write(
                f"  Step {step}: loss={loss_info['total_loss']:.4f} "
                f"reward={loss_info['mean_reward']:.4f} "
                f"entropy={loss_info['mean_entropy']:.4f}"
            )
            writer.add_scalar("train/total_loss", loss_info["total_loss"], step)
            writer.add_scalar("train/policy_loss", loss_info["policy_loss"], step)
            writer.add_scalar("train/mean_reward", loss_info["mean_reward"], step)
            writer.add_scalar("train/mean_entropy", loss_info["mean_entropy"], step)
            
            # Log individual reward components
            for key, val in rollout["reward_info"].items():
                writer.add_scalar(f"train/reward_{key}", val.mean().item(), step)
            writer.add_scalar("train/smoothness", rollout["smoothness"].mean().item(), step)
        
        # === Evaluation ===
        if step % config.eval_every == 0:
            mean_clip = evaluate(
                pipeline, policy, reward_model,
                config.eval_prompts, config, step, writer, config.sample_dir
            )
            
            if mean_clip > best_clip:
                best_clip = mean_clip
                # Save best model
                save_path = Path(config.save_dir) / "best"
                save_path.mkdir(parents=True, exist_ok=True)
                if policy is not None:
                    torch.save(policy.state_dict(), save_path / "policy.pt")
                if config.use_lora:
                    pipeline.unet.save_pretrained(save_path / "lora")
                print(f"  New best CLIP: {best_clip:.4f}")
        
        # === Periodic save ===
        if step % config.save_every == 0:
            save_path = Path(config.save_dir) / f"step_{step}"
            save_path.mkdir(parents=True, exist_ok=True)
            if policy is not None:
                torch.save(policy.state_dict(), save_path / "policy.pt")
            if config.use_lora:
                pipeline.unet.save_pretrained(save_path / "lora")
    
    # Final save
    print("\nTraining complete!")
    print(f"Best CLIP score: {best_clip:.4f}")
    
    save_path = Path(config.save_dir) / "final"
    save_path.mkdir(parents=True, exist_ok=True)
    if policy is not None:
        torch.save(policy.state_dict(), save_path / "policy.pt")
    if config.use_lora:
        pipeline.unet.save_pretrained(save_path / "lora")
    
    writer.close()
    print(f"Logs: {config.log_dir}")
    print(f"Checkpoints: {config.save_dir}")
    print(f"Samples: {config.sample_dir}")


if __name__ == "__main__":
    main()
