"""
Policy Network πϕ(a_t | s_t) from the proposal.

Observes: current noisy latent x_t (pooled), timestep t, text conditioning c
Outputs: deviations (Δγ_t, Δη_t) for guidance scale and stochasticity

Key design choices from the proposal:
  - Outputs DEVIATIONS from defaults, not absolute values
  - Applied only at selected timesteps (T_control)
  - Uses global-average-pooled latent (not full spatial) for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math


class GuidancePolicy(nn.Module):
    """Lightweight MLP policy that controls inference-time parameters.
    
    State: s_t = (pooled_latent, timestep_embedding)
    Action: a_t = (Δγ_t, Δη_t) ~ Normal(μ, σ)
    
    The policy outputs mean and log_std for a diagonal Gaussian.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Timestep embedding (sinusoidal, like in diffusion models)
        self.time_embed_dim = 64
        
        # Latent pooling: from (B, 4, H/8, W/8) -> (B, 64) via learned projection
        self.latent_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # (B, 4, 1, 1)
            nn.Flatten(),                       # (B, 4)
            nn.Linear(4, 64),
            nn.SiLU(),
        )
        
        # Input dim = time_embed (64) + latent_pool (64) = 128
        input_dim = self.time_embed_dim + 64
        hidden_dim = config.policy_hidden_dim
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Output heads: mean and log_std for (Δγ, Δη)
        self.mean_head = nn.Linear(hidden_dim, 2)
        self.log_std_head = nn.Linear(hidden_dim, 2)
        
        # Initialize output layers to near-zero (start with default schedule)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.constant_(self.log_std_head.bias, -1.0)  # small initial std
        
        # Ranges for clamping
        self.gamma_min, self.gamma_max = config.gamma_range
        self.eta_min, self.eta_max = config.eta_range
        self.default_gamma = config.default_guidance_scale
        self.default_eta = config.default_eta
    
    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embedding."""
        half_dim = self.time_embed_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        # Normalize timestep to [0, 1]
        t_normalized = timesteps.float() / 1000.0
        emb = t_normalized.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor):
        """
        Args:
            latents: (B, 4, H/8, W/8) current noisy latent
            timesteps: (B,) or scalar, current timestep
            
        Returns:
            gamma_t: (B,) guidance scale for this step
            eta_t: (B,) stochasticity for this step
            log_prob: (B,) log probability of the sampled action
            entropy: (B,) entropy of the action distribution
        """
        B = latents.shape[0]
        
        # Ensure timesteps is (B,)
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0).expand(B)
        
        # Encode state
        t_emb = self.get_timestep_embedding(timesteps)     # (B, 64)
        latents = latents / (latents.abs().mean(dim=(1,2,3), keepdim=True) + 1e-6)
        l_emb = self.latent_pool(latents)                   # (B, 64)
        state = torch.cat([t_emb, l_emb], dim=-1)           # (B, 128)
        
        # Forward through backbone
        h = self.backbone(state)                             # (B, hidden_dim)
        h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # Get distribution parameters
        mean = self.mean_head(h)
        mean = torch.tanh(mean) * 5.0      
        log_std = self.log_std_head(h).clamp(-5, 2)          # (B, 2)
        std = torch.exp(log_std)
        std = torch.clamp(std, 1e-3, 2.0)   
        std = torch.clamp(std, min=1e-3, max=2.0)   
        
        # Sample actions (Δγ, Δη)
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("⚠️ NaN detected in policy, resetting")
            mean = torch.zeros_like(mean)
            std = torch.ones_like(std) * 0.1
        dist = Normal(mean, std)
        raw_action = dist.rsample()                          # (B, 2) - reparameterized
        
        # Log probability
        log_prob = dist.log_prob(raw_action).sum(dim=-1)     # (B,)
        entropy = dist.entropy().sum(dim=-1)                 # (B,)
        
        # Split into Δγ and Δη
        delta_gamma = raw_action[:, 0]
        delta_eta = raw_action[:, 1]
        
        # Apply deviations to defaults and clamp
        gamma_t = torch.clamp(
            self.default_gamma + delta_gamma,
            self.gamma_min, self.gamma_max
        )
        eta_t = torch.clamp(
            self.default_eta + torch.sigmoid(delta_eta),  # sigmoid to keep positive
            self.eta_min, self.eta_max
        )
        
        return gamma_t, eta_t, log_prob, entropy
    
    def get_default_actions(self, batch_size: int, device: torch.device):
        """Return default (non-policy) actions for non-control timesteps."""
        gamma = torch.full((batch_size,), self.default_gamma, device=device)
        eta = torch.full((batch_size,), self.default_eta, device=device)
        log_prob = torch.zeros(batch_size, device=device)
        entropy = torch.zeros(batch_size, device=device)
        return gamma, eta, log_prob, entropy
