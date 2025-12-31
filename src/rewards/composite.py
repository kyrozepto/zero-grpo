"""
Composite Reward with Adaptive Curriculum Weighting.

Combines all reward components (depth warping, epipolar, semantic)
with time-dependent adaptive weighting for curriculum learning.

Reference: Algorithm 1 in the paper.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .depth_warping import DepthWarpingReward
from .epipolar import EpipolarReward
from .semantic import SemanticReward


class CompositeReward(nn.Module):
    """
    Combined multi-view consistency reward with curriculum learning.
    
    R_total = (r_geom)^(γ_t) · (r_sem)^(1-γ_t)
    
    Where:
    - r_geom = r_warp * r_epi (geometric consistency)
    - r_sem = semantic identity reward
    - γ_t = min(γ_max, γ_base + η * step) (adaptive weight)
    
    The curriculum starts with higher weight on semantic reward (to avoid
    Janus problem early in training) and gradually shifts to geometric
    reward (to enforce 3D structure at convergence).
    
    Args:
        gamma_base: Initial geometric weight (default: 0.3)
        eta: Learning rate for weight increase (default: 0.0001)
        gamma_max: Maximum geometric weight (default: 0.8)
        device: Torch device for computation
    """
    
    def __init__(
        self,
        gamma_base: float = 0.3,
        eta: float = 0.0001,
        gamma_max: float = 0.8,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.gamma_base = gamma_base
        self.eta = eta
        self.gamma_max = gamma_max
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize sub-reward modules
        self.depth_warp = DepthWarpingReward(device=self.device)
        self.epipolar = EpipolarReward(device=self.device)
        self.semantic = SemanticReward(device=self.device)
        
        # Track current training step
        self.register_buffer('current_step', torch.tensor(0, dtype=torch.long))
    
    def compute_gamma_t(self, step: Optional[int] = None) -> float:
        """
        Compute adaptive geometric weight at training step t.
        
        γ_t = min(γ_max, γ_base + η * t)
        
        Args:
            step: Training step (uses internal counter if None)
            
        Returns:
            gamma_t: Current geometric weight
        """
        if step is None:
            step = self.current_step.item()
        
        gamma_t = min(self.gamma_max, self.gamma_base + self.eta * step)
        return gamma_t
    
    def update_step(self, step: int):
        """Update the current training step."""
        self.current_step.fill_(step)
    
    def increment_step(self):
        """Increment the training step by 1."""
        self.current_step += 1
    
    def forward(
        self,
        images: torch.Tensor,
        condition: torch.Tensor,
        cameras: Optional[Dict] = None,
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite reward for multi-view images.
        
        Args:
            images: Multi-view generated images (B, num_views, C, H, W)
            condition: Conditioning image (B, C, H, W)
            cameras: Optional dict with camera parameters
            step: Training step (for adaptive weighting)
            
        Returns:
            Dictionary with all reward components:
            - 'total': R_total (combined reward)
            - 'warp': R_warp (depth warping)
            - 'epipolar': R_epi (epipolar constraint)
            - 'semantic': R_sem (semantic identity)
            - 'geometric': r_geom (warp * epi)
            - 'gamma_t': Current geometric weight
        """
        # Compute individual rewards
        warp_result = self.depth_warp.forward(images, **(cameras or {}))
        epi_result = self.epipolar.forward(images, **(cameras or {}))
        sem_result = self.semantic.forward(images, condition)
        
        r_warp = warp_result['reward']
        r_epi = epi_result['reward']
        r_sem = sem_result['reward']
        
        # Combine geometric rewards (multiplicative)
        r_geom = r_warp * r_epi
        
        # Get adaptive weight
        gamma_t = self.compute_gamma_t(step)
        
        # Compute total reward: R = (r_geom)^(γ_t) · (r_sem)^(1-γ_t)
        # Add small epsilon to avoid log(0) issues
        eps = 1e-6
        r_geom_safe = torch.clamp(r_geom, min=eps, max=1.0)
        r_sem_safe = torch.clamp(r_sem, min=eps, max=1.0)
        
        # Using log-space for numerical stability
        log_total = gamma_t * torch.log(r_geom_safe) + (1 - gamma_t) * torch.log(r_sem_safe)
        r_total = torch.exp(log_total)
        r_total = torch.clamp(r_total, min=0.0, max=1.0)
        
        return {
            'total': r_total,
            'warp': r_warp,
            'epipolar': r_epi,
            'semantic': r_sem,
            'geometric': r_geom,
            'gamma_t': gamma_t,
            # Additional details from sub-rewards
            'warp_details': warp_result,
            'epi_details': epi_result,
            'sem_details': sem_result
        }
    
    def __call__(
        self,
        images: torch.Tensor,
        condition: torch.Tensor,
        cameras: Optional[Dict] = None,
        step: Optional[int] = None
    ) -> torch.Tensor:
        """
        Simplified interface returning just the total reward.
        
        Args:
            images: Multi-view generated images (B, num_views, C, H, W)
            condition: Conditioning image (B, C, H, W)
            cameras: Optional camera parameters
            step: Optional training step
            
        Returns:
            reward: R_total score
        """
        result = self.forward(images, condition, cameras, step)
        return result['total']
    
    def get_curriculum_schedule(self, max_steps: int = 10000) -> Dict[str, list]:
        """
        Generate the curriculum schedule for visualization.
        
        Args:
            max_steps: Maximum training steps
            
        Returns:
            Dictionary with steps and corresponding weights
        """
        steps = list(range(0, max_steps + 1, 100))
        gamma_values = [self.compute_gamma_t(s) for s in steps]
        
        return {
            'steps': steps,
            'gamma_t': gamma_values,
            'semantic_weight': [1 - g for g in gamma_values],
            'geometric_weight': gamma_values
        }


class CompositeRewardWithPenalties(CompositeReward):
    """
    Extended composite reward with additional penalty terms.
    
    Can add penalties for:
    - Extreme depth values
    - Low keypoint match counts
    - High variance in per-view similarities
    """
    
    def __init__(
        self,
        gamma_base: float = 0.3,
        eta: float = 0.0001,
        gamma_max: float = 0.8,
        depth_penalty_weight: float = 0.1,
        match_penalty_weight: float = 0.1,
        variance_penalty_weight: float = 0.05,
        device: Optional[torch.device] = None
    ):
        super().__init__(gamma_base, eta, gamma_max, device)
        
        self.depth_penalty_weight = depth_penalty_weight
        self.match_penalty_weight = match_penalty_weight
        self.variance_penalty_weight = variance_penalty_weight
    
    def forward(
        self,
        images: torch.Tensor,
        condition: torch.Tensor,
        cameras: Optional[Dict] = None,
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute composite reward with penalty terms."""
        # Get base rewards
        result = super().forward(images, condition, cameras, step)
        
        total = result['total']
        
        # Penalty 1: Depth variance penalty (too extreme depths indicate problems)
        depths = result['warp_details'].get('depths')
        if depths is not None:
            depth_var = depths.var()
            depth_penalty = self.depth_penalty_weight * torch.tanh(depth_var)
            total = total - depth_penalty
            result['depth_penalty'] = depth_penalty
        
        # Penalty 2: Low match count penalty
        num_matches = result['epi_details'].get('num_matches', 0)
        if num_matches < 50:  # Too few matches indicate poor features
            match_penalty = self.match_penalty_weight * (1 - num_matches / 50)
            total = total - match_penalty
            result['match_penalty'] = match_penalty
        
        # Penalty 3: Semantic variance penalty (views should be equally similar)
        per_view_sim = result['sem_details'].get('per_view_similarity')
        if per_view_sim is not None:
            sim_variance = per_view_sim.var(dim=1).mean()
            variance_penalty = self.variance_penalty_weight * sim_variance
            total = total - variance_penalty
            result['variance_penalty'] = variance_penalty
        
        # Clamp final result
        result['total'] = torch.clamp(total, min=0.0, max=1.0)
        
        return result
