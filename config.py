"""
Configuration for RL-Guided Diffusion Inference.
Combines methods from:
  [2] DPOK - RL fine-tuning with KL regularization
  [3] RLG  - RL guidance via model blending
  [4] Miao - Diversity reward
  [5] Papalampidi - Dynamic CFG via online feedback
  [6] CLIP - Text-image alignment reward
  [7] LPIPS - Perceptual quality reward
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # === Model ===
    model_id: str = "runwayml/stable-diffusion-v1-5"
    dtype: str = "fp16"  # fp16 for 12GB VRAM
    low_cpu_mem_usage: bool = True
    use_safetensors: bool = True
    # Memory-saving knobs (especially for CPU runs)
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False
    # Cache a separate base U-Net for RLG/baseline; disable to save RAM
    cache_base_unet: bool = False

    # === Extra VRAM-saving knobs (logic-preserving) ===
    # Enable U-Net gradient checkpointing: recomputes activations in backward
    # instead of storing them. Biggest single win when LoRA is being trained
    # (each of the N inference steps would otherwise keep a full activation
    # graph alive until backward). Slows backward ~30%.
    gradient_checkpointing: bool = True
    # Run the unconditional and conditional CFG forwards as two sequential
    # batch-B passes instead of one concatenated batch-2B pass. Halves peak
    # U-Net activation memory at the cost of one extra kernel launch per step.
    # GroupNorm is per-sample, so this is bit-identical to the concat path.
    cfg_split_batch: bool = True
    # Stash the latents trajectory on CPU instead of GPU. The smoothness
    # reward only needs them after the rollout completes, so the GPU copies
    # are dead weight during the rest of the step.
    latent_history_to_cpu: bool = True
    # Periodically release cached blocks back to the allocator. 0 disables.
    # Helps with fragmentation on long runs but adds a small per-call cost.
    empty_cache_every_n_steps: int = 0
    
    # === Diffusion Sampling ===
    num_inference_steps: int = 20       # fewer steps = faster RL episodes
    image_size: int = 512
    default_guidance_scale: float = 7.5
    default_eta: float = 0.0            # DDIM default (deterministic)
    
    # === LoRA (from DPOK [2]) ===
    use_lora: bool = True
    lora_rank: int = 4
    lora_alpha: int = 4
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0"
    ])
    
    # === Policy Network (our addition from proposal) ===
    use_policy_network: bool = True
    policy_hidden_dim: int = 256
    policy_input_dim: int = 1 + 64      # timestep (1) + pooled latent features (64)
    gamma_range: tuple = (1.0, 20.0)     # allowed guidance scale range
    eta_range: tuple = (0.0, 1.0)        # allowed stochasticity range
    # Control timesteps (from proposal: apply policy at selected steps)
    control_every_n_steps: int = 2       # apply policy every N steps
    # Sparse intervention evaluation: use the controller only on a top-K subset
    sparse_intervention_top_k: int = 8
    sparse_intervention_strategy: str = "early"  # early | uniform
    sparse_gate_dynamic_cfg: bool = True
    sparse_gate_rlg: bool = True
    
    # === RL Training (DPOK [2]) ===
    learning_rate: float = 5e-5
    lora_lr: float = 1e-5
    lora_train_eta: float = 1.0          # stochastic DDIM eta used for LoRA-only training
    reward_weight: float = 1.0
    kl_weight: float = 0.01
    max_train_steps: int = 500
    gradient_accumulation_steps: int = 4
    clip_norm: float = 0.05
    gen_batch_size: int = 1              # images per generation (12GB VRAM)
    policy_batch_size: int = 2           # batch for policy update
    buffer_size: int = 50                # replay buffer size
    
    # === Reward Weights ===
    # Terminal reward: R = λ1*CLIP - λ2*LPIPS + λ3*Diversity  (from proposal + [4])
    lambda_clip: float = 1.0            # [6] CLIP alignment weight
    lambda_lpips: float = 0.5           # [7] LPIPS perceptual quality weight
    lambda_diversity: float = 0.3       # [4] Miao diversity reward weight
    # Auxiliary shaping reward (from proposal)
    lambda_smooth: float = 0.01         # smoothness penalty weight
    
    # === Dynamic CFG (Papalampidi [5]) ===
    use_dynamic_cfg: bool = True
    cfg_candidates: List[float] = field(default_factory=lambda: [
        3.0, 5.0, 7.5, 10.0, 12.5, 15.0
    ])
    # Whether to use latent CLIP for per-step CFG selection
    use_latent_clip_feedback: bool = True
    
    # === RLG Blending ([3]) ===
    use_rlg_blending: bool = True
    rlg_scale: float = 1.5              # interpolation scale for RL-tuned vs base
    
    # === Diversity Reward (Miao [4]) ===
    diversity_num_samples: int = 4       # samples per prompt for diversity calc
    diversity_feature_layer: str = "clip" # use CLIP features for diversity
    
    # === Prompts ===
    train_prompts: List[str] = field(default_factory=lambda: [
        "A green colored rabbit.",
        "A cat sitting on a red chair.",
        "A dog wearing sunglasses at the beach.",
        "A castle on top of a mountain at sunset.",
        "A robot painting a picture in an art studio.",
        "A blue bird flying over a snowy forest.",
        "An astronaut riding a horse on the moon.",
        "A teddy bear skateboarding in Times Square.",
        "A bowl of soup that looks like a swimming pool.",
        "A photograph of a confused grizzly bear in calculus class.",
        "A raccoon playing guitar in a park.",
        "A golden retriever wearing a top hat.",
        "Two cats playing chess on a rainy day.",
        "A panda making latte art.",
        "A steampunk owl reading a book.",
    ])
    
    eval_prompts: List[str] = field(default_factory=lambda: [
        "A corgi wearing a crown sitting on a throne.",
        "A fox in a spacesuit floating in space.",
        "A lighthouse during a thunderstorm.",
        "Three penguins having a tea party.",
        "A dragon made of flowers.",
    ])
    
    # === Logging ===
    log_dir: str = "/kaggle/working/logs"
    save_dir: str = "/kaggle/working/checkpoints"
    sample_dir: str = "/kaggle/working/samples"
    log_every: int = 10
    save_every: int = 50
    eval_every: int = 50
    
    # === Seed ===
    seed: int = 42
