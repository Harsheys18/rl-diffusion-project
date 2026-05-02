"""
Reward Model combining:
  [6] CLIP  - text-image alignment
  [7] LPIPS - perceptual quality (lower = better, so we negate)
  [4] Miao  - diversity reward via feature-space coverage
  Proposal  - trajectory smoothness auxiliary reward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import lpips
from PIL import Image
from torchvision import transforms


class CLIPReward(nn.Module):
    """CLIP-based text-image alignment reward [6]."""
    
    def __init__(self, device="cuda"):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model = self.model.to(device).eval()
        self.device = device
        
        # Freeze CLIP
        for p in self.model.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Get CLIP image features from tensor images (B, 3, H, W) in [-1, 1]."""
        images = images.float()
        # Resize and normalize for CLIP
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        images = (images + 1.0) / 2.0  # [-1,1] -> [0,1]
        # CLIP normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        images = (images - mean) / std
        return self.model.encode_image(images)
    
    @torch.no_grad()
    def get_text_features(self, texts: list) -> torch.Tensor:
        """Get CLIP text features."""
        tokens = self.tokenizer(texts).to(self.device)
        return self.model.encode_text(tokens)
    
    @torch.no_grad()
    def score(self, images: torch.Tensor, texts: list) -> torch.Tensor:
        """Compute CLIP cosine similarity between images and texts.
        
        Args:
            images: (B, 3, H, W) tensor in [-1, 1]
            texts: list of B text prompts
            
        Returns:
            (B,) tensor of CLIP scores
        """
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(texts)
        
        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Cosine similarity
        return (image_features * text_features).sum(dim=-1)
    
    @torch.no_grad()
    def get_latent_clip_score(self, latent_images: torch.Tensor, texts: list,
                              vae_decode_fn=None) -> torch.Tensor:
        """Score latent images using CLIP (for Papalampidi [5] dynamic CFG).
        
        If vae_decode_fn is provided, decode latents first.
        Otherwise, apply a simple pooling heuristic on latent space.
        """
        if vae_decode_fn is not None:
            # Decode latent to pixel space for accurate scoring
            with torch.no_grad():
                pixel_images = vae_decode_fn(latent_images)
            return self.score(pixel_images, texts)
        else:
            # Fallback: pool latent features and compare with text
            # This is a rough approximation for speed
            pooled = latent_images.mean(dim=(2, 3))  # (B, 4)
            text_features = self.get_text_features(texts)
            text_features = F.normalize(text_features, dim=-1)
            pooled = F.normalize(pooled, dim=-1)
            # Not very meaningful, but serves as a cheap proxy
            return (pooled[:, :text_features.shape[1]] * text_features).sum(dim=-1)


class LPIPSReward(nn.Module):
    """LPIPS perceptual quality reward [7].
    
    Lower LPIPS = more realistic, so we negate it for reward.
    We compare against a "reference" which is the mean of generated images
    or can be self-referential (intra-batch diversity signal).
    """
    
    def __init__(self, device="cuda"):
        super().__init__()
        self.loss_fn = lpips.LPIPS(net="vgg").to(device).eval()
        self.device = device
        
        for p in self.loss_fn.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def quality_score(self, images: torch.Tensor) -> torch.Tensor:
        """Compute a quality heuristic based on LPIPS self-similarity.
        
        Higher internal consistency within each image suggests realism.
        We use random crops to estimate this.
        
        Args:
            images: (B, 3, H, W) tensor in [-1, 1]
        Returns:
            (B,) tensor of quality scores (higher = better)
        """
        images = images.float()
        B = images.shape[0]
        scores = []
        
        for i in range(B):
            img = images[i:i+1]  # (1, 3, H, W)
            # Compare flipped versions (a simple realism proxy)
            flipped = torch.flip(img, dims=[3])
            dist = self.loss_fn(img, flipped)
            # Lower LPIPS distance to flipped = more symmetric/realistic
            # Negate so higher = better
            scores.append(-dist.squeeze())
        
        return torch.stack(scores)


class DiversityReward(nn.Module):
    """Diversity reward from Miao et al. [4].
    
    Measures how well a batch of generated images covers the feature space.
    Uses pairwise distances in CLIP feature space.
    Higher diversity = higher reward.
    """
    
    def __init__(self, clip_reward: CLIPReward, device="cuda"):
        super().__init__()
        self.clip_reward = clip_reward
        self.device = device
    
    @torch.no_grad()
    def score(self, images: torch.Tensor) -> torch.Tensor:
        """Compute diversity score for a batch of images.
        
        Uses mean pairwise distance in CLIP feature space.
        Each image gets a score = its average distance to all other images.
        
        Args:
            images: (B, 3, H, W) tensor in [-1, 1], B >= 2
        Returns:
            (B,) tensor of diversity scores
        """
        B = images.shape[0]
        if B < 2:
            return torch.zeros(B, device=self.device)
        
        # Get CLIP features for all images
        features = self.clip_reward.get_image_features(images)
        features = F.normalize(features, dim=-1)  # (B, D)
        
        # Pairwise cosine similarity matrix
        sim_matrix = features @ features.T  # (B, B)
        
        # Mask diagonal
        mask = ~torch.eye(B, dtype=torch.bool, device=self.device)
        
        # Mean distance (1 - similarity) to other images
        # Higher distance = more diverse
        distances = (1.0 - sim_matrix) * mask.float()
        diversity_scores = distances.sum(dim=1) / (B - 1)
        
        return diversity_scores


class CompositeReward(nn.Module):
    """Combined reward function as described in the proposal.
    
    R_term(x0, c) = λ1·CLIP(x0, c) - λ2·LPIPS(x0) + λ3·Diversity(x0)
    r_aux(t)      = -||x_t - x_{t-1}||^2  (smoothness)
    """
    
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        
        print("Loading CLIP reward model...")
        self.clip_reward = CLIPReward(device=device)
        print("Loading LPIPS reward model...")
        self.lpips_reward = LPIPSReward(device=device)
        print("Loading Diversity reward model...")
        self.diversity_reward = DiversityReward(self.clip_reward, device=device)
        print("All reward models loaded.")
    
    @torch.no_grad()
    def compute_terminal_reward(self, images: torch.Tensor, prompts: list) -> dict:
        """Compute the full terminal reward for generated images.
        
        Args:
            images: (B, 3, H, W) tensor in [-1, 1]
            prompts: list of B text prompts
            
        Returns:
            dict with 'total', 'clip', 'lpips', 'diversity' keys
        """
        clip_scores = self.clip_reward.score(images, prompts)
        lpips_scores = self.lpips_reward.quality_score(images)
        diversity_scores = self.diversity_reward.score(images)
        
        total = (
            self.config.lambda_clip * clip_scores
            + self.config.lambda_lpips * lpips_scores
            + self.config.lambda_diversity * diversity_scores
        )
        
        return {
            "total": total,
            "clip": clip_scores,
            "lpips": lpips_scores,
            "diversity": diversity_scores,
        }
    
    @staticmethod
    def compute_smoothness_reward(latents_history: list) -> torch.Tensor:
        """Auxiliary smoothness reward from the proposal.
        
        r_aux(t) = -||x_t - x_{t-1}||^2
        
        Args:
            latents_history: list of (B, C, H, W) tensors at each timestep
        Returns:
            (B,) tensor of summed smoothness penalties
        """
        if len(latents_history) < 2:
            return torch.zeros(latents_history[0].shape[0], 
                             device=latents_history[0].device)
        
        total_penalty = torch.zeros(latents_history[0].shape[0],
                                   device=latents_history[0].device)
        
        for t in range(1, len(latents_history)):
            diff = latents_history[t] - latents_history[t-1]
            penalty = -(diff ** 2).mean(dim=(1, 2, 3))  # (B,)
            total_penalty += penalty
        
        return total_penalty / (len(latents_history) - 1)
    
    def get_latent_clip_score(self, latent_images, texts, vae_decode_fn=None):
        """Expose latent CLIP scoring for dynamic CFG (Papalampidi [5])."""
        return self.clip_reward.get_latent_clip_score(
            latent_images, texts, vae_decode_fn
        )
