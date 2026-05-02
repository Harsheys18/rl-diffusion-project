"""
Extended Stable Diffusion Pipeline for RL-Guided Inference.

Integrates:
  [2] DPOK  - LoRA fine-tuning + log-prob tracking for REINFORCE
  [3] RLG   - Blending base and RL-tuned model predictions  
  [5] Papalampidi - Dynamic CFG via greedy search with latent feedback
  Proposal  - Policy-controlled (γ_t, η_t) at each denoising step
"""

import copy
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler
from typing import Optional, List, Tuple
import numpy as np


class RLGuidedPipeline:
    """Wrapper around SD pipeline with RL-guided inference controls."""
    
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device
        
        # Load SD 1.5
        dtype = self._resolve_dtype(config.dtype, device)
        self.torch_dtype = dtype
        print(f"Loading Stable Diffusion from {config.model_id} ({dtype})...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            safety_checker=None,
            low_cpu_mem_usage=config.low_cpu_mem_usage,
            use_safetensors=config.use_safetensors,
        ).to(device)

        # Memory-saving features (especially on CPU)
        if config.enable_attention_slicing:
            self.pipe.enable_attention_slicing()
        if config.enable_vae_slicing:
            self.pipe.enable_vae_slicing()
        if config.enable_vae_tiling:
            self.pipe.enable_vae_tiling()

        # Gradient checkpointing on the U-Net: recompute activations in
        # backward instead of holding the whole step's worth in VRAM.
        # Set BEFORE LoRA wrapping so PEFT inherits the flag from the
        # underlying modules. No-op for inference (no backward).
        if getattr(config, "gradient_checkpointing", False):
            try:
                self.pipe.unet.enable_gradient_checkpointing()
                print("Gradient checkpointing enabled on U-Net.")
            except Exception as exc:  # older diffusers / non-standard U-Net
                print(f"Could not enable gradient checkpointing: {exc}")
        
        # Set up DDIM scheduler (matches DPOK)
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            rescale_betas_zero_snr=False,
        )
        self.pipe.scheduler.set_timesteps(config.num_inference_steps)
        
        # Shortcuts
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = self.pipe.scheduler
        
        # Keep an independent original U-Net for true pre-RL baselines and RLG.
        # PEFT can mutate the wrapped model, so a reference is not enough here.
        self._base_unet = None
        cache_base_unet = config.cache_base_unet or config.use_rlg_blending
        if cache_base_unet:
            print("Caching independent base U-Net for baseline/RLG...")
            self._base_unet = copy.deepcopy(self.pipe.unet).to(device).eval()
            for p in self._base_unet.parameters():
                p.requires_grad = False
        
        print("Pipeline ready.")

    @staticmethod
    def _resolve_dtype(dtype_name, device):
        """Return a torch dtype that is safe for the selected device."""
        if device == "cpu" and dtype_name == "fp16":
            print("Config requested fp16 on CPU; using fp32 instead.")
            return torch.float32
        if dtype_name == "fp16":
            return torch.float16
        if dtype_name == "bf16":
            return torch.bfloat16
        if dtype_name == "fp32":
            return torch.float32
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Use fp16, bf16, or fp32.")
    
    def encode_prompt(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts into conditioning embeddings.
        
        Returns:
            cond_embeds: (B, seq_len, dim) conditional embeddings
            uncond_embeds: (B, seq_len, dim) unconditional embeddings
        """
        # Conditional
        text_inputs = self.tokenizer(
            prompts, padding="max_length", 
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(self.device)
        cond_embeds = self.text_encoder(text_inputs.input_ids)[0]
        
        # Unconditional (empty prompt)
        uncond_inputs = self.tokenizer(
            [""] * len(prompts), padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(self.device)
        uncond_embeds = self.text_encoder(uncond_inputs.input_ids)[0]
        
        return cond_embeds, uncond_embeds
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to pixel images in [-1, 1]."""
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            images = self.vae.decode(latents).sample
        return images.clamp(-1, 1)
    
    def _active_unet(self):
        """Return the U-Net for the current stage: base or LoRA/RL-tuned."""
        if self.config.use_lora:
            return self.unet
        return self._base_unet if self._base_unet is not None else self.unet

    def _predict_noise_with_unet(self, unet, latents, timestep, cond_embeds, uncond_embeds,
                                 guidance_scale):
        """Run classifier-free guidance prediction.
        
        Implements: ε̂_t = ε_uncond + γ_t * (ε_cond - ε_uncond)
        From the proposal slide 6.

        With ``cfg_split_batch=True`` (default) the unconditional and
        conditional branches are run as two sequential batch-B forwards
        instead of a single concatenated batch-2B forward. The U-Net only
        contains GroupNorm (which is per-sample), so the two paths are
        bit-identical; this just halves peak activation memory at the cost
        of one extra forward dispatch.
        """
        if getattr(self.config, "cfg_split_batch", False):
            # Two sequential forwards at batch B.
            noise_uncond = unet(
                latents, timestep, encoder_hidden_states=uncond_embeds
            ).sample
            noise_cond = unet(
                latents, timestep, encoder_hidden_states=cond_embeds
            ).sample
        else:
            # Original single forward at batch 2B.
            latent_input = torch.cat([latents, latents], dim=0)
            embed_input = torch.cat([uncond_embeds, cond_embeds], dim=0)

            noise_pred = unet(
                latent_input, timestep, encoder_hidden_states=embed_input
            ).sample

            noise_uncond, noise_cond = noise_pred.chunk(2)

        # CFG: ε̂ = ε_uncond + γ * (ε_cond - ε_uncond)
        noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        return noise_guided, noise_uncond, noise_cond

    def _predict_noise(self, latents, timestep, cond_embeds, uncond_embeds,
                       guidance_scale):
        return self._predict_noise_with_unet(
            self._active_unet(), latents, timestep, cond_embeds, uncond_embeds,
            guidance_scale
        )
    
    def _predict_noise_base(self, latents, timestep, cond_embeds, uncond_embeds,
                            guidance_scale):
        """Predict noise using BASE (non-LoRA) U-Net weights for RLG [3]."""
        if self._base_unet is None:
            return None

        noise_guided, _, _ = self._predict_noise_with_unet(
            self._base_unet,
            latents, timestep, cond_embeds, uncond_embeds, guidance_scale
        )

        return noise_guided
    
    def _apply_rlg_blending(self, noise_lora, noise_base, rlg_scale):
        """RLG blending from Luo et al. [3].
        
        Blends predictions from base and RL-tuned models:
          ε̂_final = ε_base + rlg_scale * (ε_lora - ε_base)
          
        This is like classifier guidance but using the RL-tuned model
        as the "classifier".
        """
        if noise_base is None:
            return noise_lora
        
        return noise_base + rlg_scale * (noise_lora - noise_base)
    
    def _dynamic_cfg_search(self, latents, timestep, cond_embeds, uncond_embeds,
                            prompts, reward_model, cfg_candidates):
        """Dynamic CFG from Papalampidi et al. [5].
        
        For each candidate γ, compute the denoised latent and score it
        with a latent-space evaluator. Pick the best γ.
        
        Key insight from [5]: this requires NO extra NFEs since we reuse
        the same unconditional/conditional predictions.
        """
        # Get unconditional and conditional predictions (single forward pass).
        # Mirrors the split-vs-concat policy used by ``_predict_noise_with_unet``.
        unet_for_search = self._active_unet()
        if getattr(self.config, "cfg_split_batch", False):
            noise_uncond = unet_for_search(
                latents, timestep, encoder_hidden_states=uncond_embeds
            ).sample
            noise_cond = unet_for_search(
                latents, timestep, encoder_hidden_states=cond_embeds
            ).sample
        else:
            latent_input = torch.cat([latents, latents], dim=0)
            embed_input = torch.cat([uncond_embeds, cond_embeds], dim=0)
            noise_pred = unet_for_search(
                latent_input, timestep, encoder_hidden_states=embed_input
            ).sample
            noise_uncond, noise_cond = noise_pred.chunk(2)
        
        B = latents.shape[0]
        best_scores = torch.full((B,), -float('inf'), device=self.device)
        best_gamma = torch.full((B,), self.config.default_guidance_scale, 
                               device=self.device)
        best_noise = None
        
        for gamma in cfg_candidates:
            # Apply CFG with this candidate
            noise_guided = noise_uncond + gamma * (noise_cond - noise_uncond)
            
            # Compute what the denoised latent would look like
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep.cpu().long().item()]
            alpha_prod_t = torch.as_tensor(alpha_prod_t, device=self.device, dtype=latents.dtype)
            
            # Predicted x0 from noise prediction
            pred_x0 = (latents - (1 - alpha_prod_t).sqrt() * noise_guided) / \
                       alpha_prod_t.sqrt()
            
            # Score using latent CLIP
            if reward_model is not None:
                scores = reward_model.get_latent_clip_score(
                    pred_x0, prompts, 
                    vae_decode_fn=self.decode_latents
                )
            else:
                # Fallback: prefer lower-norm predictions (less extreme)
                scores = -pred_x0.abs().mean(dim=(1, 2, 3))
            
            # Update best per sample
            improved = scores > best_scores
            best_scores[improved] = scores[improved]
            best_gamma[improved] = gamma
            if best_noise is None:
                best_noise = noise_guided.clone()
            else:
                best_noise[improved] = noise_guided[improved]
        
        return best_noise, best_gamma
    
    def _ddim_step_with_logprob(self, noise_pred, timestep, latents, eta=0.0,
                                generator=None):
        """DDIM step with log-probability computation (from DPOK [2]).
        
        This is needed for the REINFORCE policy gradient.
        
        Returns:
            next_latents: denoised latents
            log_prob: log probability of this transition
        """
        t = timestep
        # Get previous timestep
        num_steps = self.scheduler.config.num_train_timesteps
        step_ratio = num_steps // self.scheduler.num_inference_steps
        prev_t = t - step_ratio
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t.cpu().long().item()]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[max(prev_t.cpu().long().item(), 0)] \
            if prev_t >= 0 else torch.tensor(1.0)
        
        alpha_prod_t = torch.as_tensor(alpha_prod_t, device=self.device, dtype=latents.dtype)
        alpha_prod_t_prev = torch.as_tensor(alpha_prod_t_prev, device=self.device, dtype=latents.dtype)
        
        # Predicted x0
        pred_x0 = (latents - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()
        
        # Variance for stochastic DDIM
        sigma_t = eta * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)).sqrt() * \
                  (1 - alpha_prod_t / alpha_prod_t_prev).sqrt()
        
        # Direction pointing to x_t  
        pred_dir_coeff = (1 - alpha_prod_t_prev - sigma_t**2).clamp(min=0).sqrt()
        
        # Mean of the transition
        mean = alpha_prod_t_prev.sqrt() * pred_x0 + pred_dir_coeff * noise_pred
        
        if eta > 0:
            noise = torch.randn(
                latents.shape,
                device=latents.device,
                dtype=latents.dtype,
                generator=generator,
            )
            next_latents = mean + sigma_t * noise
            
            # Log probability of sampling this transition
            # Treat the sampled next latent as the action. If it stays attached
            # to mean through reparameterization, the score-function gradient
            # through the U-Net transition cancels out.
            sampled_next_latents = next_latents.detach()
            var = (sigma_t ** 2).clamp(min=1e-10)
            log_prob = -0.5 * (
                (sampled_next_latents - mean) ** 2 / var
                + torch.log(var)
                + np.log(2 * np.pi)
            )
            log_prob = log_prob.sum(dim=(1, 2, 3))  # (B,)
        else:
            next_latents = mean
            # Deterministic step: log_prob is 0 (delta distribution)
            log_prob = torch.zeros(latents.shape[0], device=self.device)
        
        return next_latents, log_prob
    
    @torch.no_grad()
    def generate_with_trajectory(
        self,
        prompts: List[str],
        policy=None,
        reward_model=None,
        seed: Optional[int] = None,
        intervention_steps: Optional[List[int]] = None,
    ) -> dict:
        """Full generation loop with RL-guided inference.
        
        This is the main inference function that combines all methods.
        
        Returns:
            dict with keys:
                'images': (B, 3, H, W) decoded images
                'latents_history': list of latents at each step
                'log_probs': (B,) total log probability of trajectory
                'gammas': list of (B,) guidance scales used
                'etas': list of (B,) stochasticity values used
                'entropies': (B,) total entropy
                'intervention_mask': list of bools indicating gated intervention steps
        """
        B = len(prompts)
        intervention_step_set = None
        if intervention_steps is not None:
            intervention_step_set = set(intervention_steps)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Encode prompts
        cond_embeds, uncond_embeds = self.encode_prompt(prompts)
        
        # Initial noise
        generator = torch.Generator(device=self.device).manual_seed(seed) \
            if seed is not None else None
        latent_shape = (B, 4, 
                       self.config.image_size // 8, 
                       self.config.image_size // 8)
        latents = torch.randn(latent_shape, device=self.device, 
                             dtype=cond_embeds.dtype, generator=generator)
        latents = latents * self.scheduler.init_noise_sigma
        
        # Tracking. Stash latents on CPU when allowed: the smoothness reward
        # is the only consumer and runs after the rollout. ``.detach()`` is
        # enough — the storage is already independent of the autograd graph
        # and the next iteration produces a brand-new tensor for ``latents``.
        history_to_cpu = getattr(self.config, "latent_history_to_cpu", False)
        latents_history = [
            latents.detach().to("cpu", non_blocking=True) if history_to_cpu
            else latents.detach().clone()
        ]
        total_log_prob = torch.zeros(B, device=self.device)
        total_entropy = torch.zeros(B, device=self.device)
        gamma_history = []
        eta_history = []
        
        timesteps = self.scheduler.timesteps
        
        for step_idx, t in enumerate(timesteps):
            t_tensor = t.unsqueeze(0).to(self.device) if t.dim() == 0 else t.to(self.device)
            
            # === Decide control parameters ===
            is_sparse_intervention_step = (
                intervention_step_set is None or step_idx in intervention_step_set
            )
            if intervention_step_set is None:
                is_control_step = step_idx % self.config.control_every_n_steps == 0
            else:
                is_control_step = step_idx in intervention_step_set
            
            if policy is not None and self.config.use_policy_network and is_control_step:
                # Policy network decides (γ_t, η_t) [Proposal]
                gamma_t, eta_t, log_prob, entropy = policy(
                    latents.float(), t_tensor.expand(B)
                )
                total_log_prob += log_prob
                total_entropy += entropy
            else:
                gamma_t = torch.full((B,), self.config.default_guidance_scale, 
                                    device=self.device)
                eta_t = torch.full((B,), self.config.default_eta, device=self.device)
            
            # === Dynamic CFG override (Papalampidi [5]) ===
            use_dynamic_cfg = self.config.use_dynamic_cfg and reward_model is not None
            if intervention_step_set is not None and self.config.sparse_gate_dynamic_cfg:
                use_dynamic_cfg = use_dynamic_cfg and is_sparse_intervention_step

            if use_dynamic_cfg:
                noise_pred, best_gamma = self._dynamic_cfg_search(
                    latents, t_tensor, cond_embeds, uncond_embeds,
                    prompts, reward_model, self.config.cfg_candidates
                )
                # Blend policy gamma with dynamic CFG: use dynamic as a constraint
                # Policy can adjust within a range around the dynamic optimum
                if policy is not None and is_control_step:
                    # Policy adjustment relative to dynamic CFG choice
                    gamma_t = 0.5 * gamma_t + 0.5 * best_gamma
                    noise_pred, _, _ = self._predict_noise(
                        latents, t_tensor, cond_embeds, uncond_embeds,
                        gamma_t.mean().item()
                    )
            else:
                # Standard CFG with policy-chosen gamma
                # Use per-sample gamma (take mean for the batch CFG call)
                mean_gamma = gamma_t.mean().item()
                noise_pred, _, _ = self._predict_noise(
                    latents, t_tensor, cond_embeds, uncond_embeds, mean_gamma
                )
            
            # === RLG Blending ([3]) ===
            use_rlg_blending = self.config.use_rlg_blending
            if intervention_step_set is not None and self.config.sparse_gate_rlg:
                use_rlg_blending = use_rlg_blending and is_sparse_intervention_step

            if use_rlg_blending and self._base_unet is not None:
                noise_base = self._predict_noise_base(
                    latents, t_tensor, cond_embeds, uncond_embeds,
                    gamma_t.mean().item()
                )
                noise_pred = self._apply_rlg_blending(
                    noise_pred, noise_base, self.config.rlg_scale
                )
            
            # === DDIM step with log-prob tracking (DPOK [2]) ===
            mean_eta = eta_t.mean().item()
            latents, step_log_prob = self._ddim_step_with_logprob(
                noise_pred, t_tensor, latents, eta=mean_eta, generator=generator
            )
            
            # Track
            latents_history.append(
                latents.detach().to("cpu", non_blocking=True) if history_to_cpu
                else latents.detach().clone()
            )
            gamma_history.append(gamma_t.detach().cpu())
            eta_history.append(eta_t.detach().cpu())
        
        # Decode final image
        images = self.decode_latents(latents)
        
        return {
            "images": images,
            "latents_history": latents_history,
            "log_probs": total_log_prob,
            "gammas": gamma_history,
            "etas": eta_history,
            "entropies": total_entropy,
            "intervention_mask": [
                intervention_step_set is None or idx in intervention_step_set
                for idx in range(len(timesteps))
            ],
        }
