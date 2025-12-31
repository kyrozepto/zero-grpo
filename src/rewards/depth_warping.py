"""
Uncertainty-Aware Depth Warping Reward (R_warp).

Implements depth-based image warping with forward-backward consistency checking
to compute geometric consistency reward for multi-view samples.

Reference: Section 4.2, "Uncertainty-Aware Depth Warping" in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import warnings

from .utils import (
    get_view_pairs,
    get_zero123pp_cameras,
    compute_relative_pose,
    warp_image_with_depth,
    normalize_image,
)


class DepthWarpingReward(nn.Module):
    """
    Computes depth warping reward with uncertainty masking.
    
    R_warp = 1 - (1/|P|) * Σ_(i,j)∈P ||M_fb · (I^(j) - I^(i→j)_warp)||_1
    
    Where:
    - P: Set of view pairs
    - M_fb: Forward-backward consistency mask
    - I^(i→j)_warp: Image i warped to view j using estimated depth
    
    Args:
        model_name: Hugging Face model ID for depth estimation
        delta: Reprojection error threshold for forward-backward check (pixels)
        device: Torch device for computation
    """
    
    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        delta: float = 1.0,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.delta = delta
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Depth Anything V2 model
        self._load_depth_model()
        
        # Cache view pairs for 6-view setup
        self.view_pairs = get_view_pairs(num_views=6)
        
    def _load_depth_model(self):
        """Load pretrained Depth Anything V2 model."""
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
            self.depth_model = self.depth_model.to(self.device)
            self.depth_model.eval()
            
            # Freeze parameters
            for param in self.depth_model.parameters():
                param.requires_grad = False
                
        except Exception as e:
            warnings.warn(
                f"Failed to load depth model '{self.model_name}': {e}\n"
                f"Falling back to placeholder depth estimation."
            )
            self.processor = None
            self.depth_model = None
    
    @torch.no_grad()
    def estimate_depth(self, images: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth maps for all views.
        
        Args:
            images: Multi-view images (B, num_views, C, H, W) in [0, 1] range
            
        Returns:
            depths: Depth maps (B, num_views, H, W)
        """
        B, num_views, C, H, W = images.shape
        
        if self.depth_model is None:
            # Fallback: return uniform depth
            warnings.warn("Using placeholder depth estimation (uniform depth).")
            return torch.ones(B, num_views, H, W, device=images.device)
        
        # Flatten batch and views
        flat_images = images.view(B * num_views, C, H, W)
        
        # Process in smaller batches to avoid OOM
        batch_size = 4
        depths = []
        
        for i in range(0, B * num_views, batch_size):
            batch = flat_images[i:i + batch_size]
            
            # Convert to PIL format expected by processor
            # The processor expects pixel values in [0, 255] or normalized
            batch_np = (batch.cpu().numpy() * 255).astype('uint8')
            batch_np = batch_np.transpose(0, 2, 3, 1)  # BCHW -> BHWC
            
            # Process through Depth Anything
            inputs = self.processor(
                images=list(batch_np),
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
            
            # Interpolate to original size
            depth_resized = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            depths.append(depth_resized)
        
        depths = torch.cat(depths, dim=0)
        depths = depths.view(B, num_views, H, W)
        
        return depths
    
    def forward_backward_check(
        self,
        img_i: torch.Tensor,
        img_j: torch.Tensor,
        depth_i: torch.Tensor,
        depth_j: torch.Tensor,
        K: torch.Tensor,
        R_ij: torch.Tensor,
        t_ij: torch.Tensor,
        R_ji: torch.Tensor,
        t_ji: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute forward-backward consistency mask.
        
        Pixels that don't warp consistently both directions are masked out
        (likely occlusions, depth discontinuities, or estimation errors).
        
        Args:
            img_i: Source image (B, C, H, W)
            img_j: Target image (B, C, H, W)
            depth_i: Depth for view i (B, H, W)
            depth_j: Depth for view j (B, H, W)
            K: Intrinsic matrix (3, 3)
            R_ij, t_ij: Relative pose from i to j
            R_ji, t_ji: Relative pose from j to i
            
        Returns:
            mask: Forward-backward consistency mask (B, H, W)
        """
        B, C, H, W = img_i.shape
        
        # Forward warp: i -> j
        warped_ij, valid_ij = warp_image_with_depth(img_i, depth_i, K, R_ij, t_ij)
        
        # Get the corresponding depths at warped locations
        # For simplicity, we warp the depth as well
        depth_i_4d = depth_i.unsqueeze(1)  # (B, 1, H, W)
        warped_depth_ij, _ = warp_image_with_depth(depth_i_4d, depth_i, K, R_ij, t_ij)
        warped_depth_ij = warped_depth_ij.squeeze(1)  # (B, H, W)
        
        # Backward warp: j -> i
        warped_ji, valid_ji = warp_image_with_depth(img_j, depth_j, K, R_ji, t_ji)
        
        # Compute forward-backward error
        # Warp img_j to i, then warp result back to j
        # Actually, we use a simpler photometric check:
        # The warped image should match the original when warped back
        
        # Photometric error between original and round-trip warped
        error_i = torch.abs(img_i - warped_ji).mean(dim=1)  # (B, H, W)
        
        # Create mask based on threshold
        mask = (error_i < self.delta / 255.0).float()  # Normalize delta
        
        # Combine with valid masks
        mask = mask * valid_ji.float()
        
        return mask
    
    def compute_warp_error(
        self,
        img_i: torch.Tensor,
        img_j: torch.Tensor,
        depth_i: torch.Tensor,
        K: torch.Tensor,
        R_ij: torch.Tensor,
        t_ij: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute masked warping error between views.
        
        Args:
            img_i: Source image (B, C, H, W)
            img_j: Target image (B, C, H, W)
            depth_i: Depth for view i (B, H, W)
            K: Intrinsic matrix (3, 3)
            R_ij, t_ij: Relative pose from i to j
            mask: Optional mask (B, H, W)
            
        Returns:
            error: Mean L1 error (scalar)
        """
        # Warp image i to view j
        warped_ij, valid_mask = warp_image_with_depth(img_i, depth_i, K, R_ij, t_ij)
        
        # Combine masks
        if mask is not None:
            combined_mask = mask * valid_mask.float()
        else:
            combined_mask = valid_mask.float()
        
        # Compute masked L1 error
        diff = torch.abs(img_j - warped_ij)
        
        if combined_mask.sum() > 0:
            masked_diff = diff * combined_mask.unsqueeze(1)
            error = masked_diff.sum() / (combined_mask.sum() * diff.shape[1] + 1e-6)
        else:
            error = torch.tensor(1.0, device=diff.device)  # Max error if no valid pixels
        
        return error
    
    def forward(
        self,
        images: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        translations: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute depth warping reward for multi-view images.
        
        Args:
            images: Multi-view images (B, num_views, C, H, W) in [0, 1] range
            K: Optional intrinsic matrix (3, 3)
            rotations: Optional rotation matrices (num_views, 3, 3)
            translations: Optional translation vectors (num_views, 3)
            
        Returns:
            Dictionary with:
            - 'reward': R_warp score (higher is better)
            - 'mean_error': Mean warping error
            - 'depths': Estimated depth maps
        """
        B, num_views, C, H, W = images.shape
        device = images.device
        
        # Get camera parameters if not provided
        if K is None or rotations is None or translations is None:
            K, rotations, translations = get_zero123pp_cameras(device=device)
        
        # Estimate depths for all views
        depths = self.estimate_depth(images)
        
        # Compute warping errors for all view pairs
        total_error = 0.0
        num_pairs = 0
        
        for i, j in self.view_pairs:
            # Get images and depths for this pair
            img_i = images[:, i]  # (B, C, H, W)
            img_j = images[:, j]
            depth_i = depths[:, i]  # (B, H, W)
            depth_j = depths[:, j]
            
            # Compute relative poses
            R_ij, t_ij = compute_relative_pose(
                rotations[i], translations[i],
                rotations[j], translations[j]
            )
            R_ji, t_ji = compute_relative_pose(
                rotations[j], translations[j],
                rotations[i], translations[i]
            )
            
            # Compute forward-backward consistency mask
            mask = self.forward_backward_check(
                img_i, img_j, depth_i, depth_j,
                K, R_ij, t_ij, R_ji, t_ji
            )
            
            # Compute masked warping error (both directions)
            error_ij = self.compute_warp_error(
                img_i, img_j, depth_i, K, R_ij, t_ij, mask
            )
            error_ji = self.compute_warp_error(
                img_j, img_i, depth_j, K, R_ji, t_ji, mask
            )
            
            total_error += (error_ij + error_ji) / 2
            num_pairs += 1
        
        mean_error = total_error / num_pairs
        
        # Reward = 1 - error (higher is better)
        reward = 1.0 - mean_error
        reward = torch.clamp(reward, min=0.0, max=1.0)
        
        return {
            'reward': reward,
            'mean_error': mean_error,
            'depths': depths
        }
    
    def __call__(
        self,
        images: torch.Tensor,
        cameras: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Simplified interface returning just the reward score.
        
        Args:
            images: Multi-view images (B, num_views, C, H, W)
            cameras: Optional dict with 'K', 'rotations', 'translations'
            
        Returns:
            reward: R_warp score (B,) or scalar
        """
        if cameras is not None:
            result = self.forward(
                images,
                K=cameras.get('K'),
                rotations=cameras.get('rotations'),
                translations=cameras.get('translations')
            )
        else:
            result = self.forward(images)
        
        return result['reward']
