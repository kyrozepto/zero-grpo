"""
Reward function modules for multi-view consistency.

This package implements three reward components for GRPO training:
1. DepthWarpingReward (R_warp): Uncertainty-aware depth warping
2. EpipolarReward (R_epi): Epipolar Sampson constraint
3. SemanticReward (R_sem): Semantic identity via DINOv2
4. CompositeReward: Combined reward with adaptive curriculum weighting
"""

from .depth_warping import DepthWarpingReward
from .epipolar import EpipolarReward
from .semantic import SemanticReward
from .composite import CompositeReward

__all__ = [
    'DepthWarpingReward',
    'EpipolarReward', 
    'SemanticReward',
    'CompositeReward',
]
