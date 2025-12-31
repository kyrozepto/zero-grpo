"""
Utility functions for multi-view reward computation.

This module provides helper functions for:
- Camera intrinsic/extrinsic parameter handling
- Zero123++ canonical view configurations
- Skew-symmetric matrix computation
- Differentiable image warping
- View pair generation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import math


# Zero123++ canonical camera configurations
# 6 views in 3x2 grid layout
ZERO123PP_ELEVATIONS = [20.0, -10.0, 20.0, -10.0, 20.0, -10.0]  # degrees (v1.2)
ZERO123PP_AZIMUTHS = [30.0, 90.0, 150.0, 210.0, 270.0, 330.0]   # degrees
ZERO123PP_FOV = 30.0  # degrees (v1.2)
ZERO123PP_VIEW_RESOLUTION = 320  # pixels per view


def get_intrinsic_matrix(
    fov: float = ZERO123PP_FOV,
    height: int = ZERO123PP_VIEW_RESOLUTION,
    width: int = ZERO123PP_VIEW_RESOLUTION,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Compute camera intrinsic matrix K from field of view.
    
    Args:
        fov: Field of view in degrees
        height: Image height in pixels
        width: Image width in pixels
        device: Torch device
        
    Returns:
        K: Intrinsic matrix (3, 3)
    """
    fov_rad = math.radians(fov)
    focal_length = width / (2 * math.tan(fov_rad / 2))
    
    cx = width / 2
    cy = height / 2
    
    K = torch.tensor([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    return K


def elevation_azimuth_to_rotation(
    elevation: float,
    azimuth: float,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Convert elevation and azimuth angles to rotation matrix.
    
    The camera looks at the origin from a point on a sphere.
    Elevation is angle from XY plane, azimuth is angle around Z axis.
    
    Args:
        elevation: Elevation angle in degrees
        azimuth: Azimuth angle in degrees
        device: Torch device
        
    Returns:
        R: Rotation matrix (3, 3)
    """
    elev_rad = math.radians(elevation)
    azim_rad = math.radians(azimuth)
    
    # Camera position on unit sphere
    x = math.cos(elev_rad) * math.cos(azim_rad)
    y = math.cos(elev_rad) * math.sin(azim_rad)
    z = math.sin(elev_rad)
    
    # Camera looks at origin
    forward = torch.tensor([-x, -y, -z], dtype=torch.float32, device=device)
    forward = forward / torch.norm(forward)
    
    # Up vector (world Z axis, adjusted for elevation)
    world_up = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
    
    # Right vector
    right = torch.cross(forward, world_up)
    right_norm = torch.norm(right)
    if right_norm < 1e-6:
        # Handle degenerate case (looking straight up/down)
        right = torch.tensor([1, 0, 0], dtype=torch.float32, device=device)
    else:
        right = right / right_norm
    
    # Recompute up vector
    up = torch.cross(right, forward)
    up = up / torch.norm(up)
    
    # Rotation matrix (camera to world)
    R = torch.stack([right, up, -forward], dim=1)
    
    return R


def get_zero123pp_cameras(
    radius: float = 1.5,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get camera parameters for Zero123++ canonical 6 views.
    
    Args:
        radius: Distance from camera to origin
        device: Torch device
        
    Returns:
        K: Intrinsic matrix (3, 3)
        rotations: Rotation matrices (6, 3, 3)
        translations: Translation vectors (6, 3)
    """
    K = get_intrinsic_matrix(device=device)
    
    rotations = []
    translations = []
    
    for elev, azim in zip(ZERO123PP_ELEVATIONS, ZERO123PP_AZIMUTHS):
        R = elevation_azimuth_to_rotation(elev, azim, device)
        rotations.append(R)
        
        # Camera position
        elev_rad = math.radians(elev)
        azim_rad = math.radians(azim)
        cam_x = radius * math.cos(elev_rad) * math.cos(azim_rad)
        cam_y = radius * math.cos(elev_rad) * math.sin(azim_rad)
        cam_z = radius * math.sin(elev_rad)
        
        # Translation: t = -R @ camera_position
        cam_pos = torch.tensor([cam_x, cam_y, cam_z], dtype=torch.float32, device=device)
        t = -R @ cam_pos
        translations.append(t)
    
    rotations = torch.stack(rotations, dim=0)
    translations = torch.stack(translations, dim=0)
    
    return K, rotations, translations


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """
    Compute skew-symmetric matrix [v]_x from vector v.
    
    [v]_x = | 0   -v_z  v_y |
            | v_z  0   -v_x |
            |-v_y  v_x  0   |
    
    Args:
        v: Vector (3,) or (B, 3)
        
    Returns:
        Skew-symmetric matrix (3, 3) or (B, 3, 3)
    """
    if v.dim() == 1:
        return torch.tensor([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ], dtype=v.dtype, device=v.device)
    else:
        B = v.shape[0]
        zero = torch.zeros(B, dtype=v.dtype, device=v.device)
        skew = torch.stack([
            torch.stack([zero, -v[:, 2], v[:, 1]], dim=1),
            torch.stack([v[:, 2], zero, -v[:, 0]], dim=1),
            torch.stack([-v[:, 1], v[:, 0], zero], dim=1)
        ], dim=1)
        return skew


def compute_essential_matrix(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute essential matrix E = [t]_x @ R.
    
    Args:
        R: Relative rotation matrix (3, 3)
        t: Relative translation vector (3,)
        
    Returns:
        E: Essential matrix (3, 3)
    """
    t_skew = skew_symmetric(t)
    E = t_skew @ R
    return E


def compute_fundamental_matrix(
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """
    Compute fundamental matrix F = K^(-T) @ E @ K^(-1).
    
    Args:
        K: Intrinsic matrix (3, 3)
        R: Relative rotation matrix (3, 3)
        t: Relative translation vector (3,)
        
    Returns:
        F: Fundamental matrix (3, 3)
    """
    E = compute_essential_matrix(R, t)
    K_inv = torch.inverse(K)
    F = K_inv.T @ E @ K_inv
    return F


def compute_relative_pose(
    R_i: torch.Tensor,
    t_i: torch.Tensor,
    R_j: torch.Tensor,
    t_j: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute relative pose from view i to view j.
    
    Args:
        R_i, t_i: Camera pose for view i
        R_j, t_j: Camera pose for view j
        
    Returns:
        R_ij: Relative rotation (3, 3)
        t_ij: Relative translation (3,)
    """
    R_ij = R_j @ R_i.T
    t_ij = t_j - R_ij @ t_i
    return R_ij, t_ij


def get_view_pairs(num_views: int = 6) -> List[Tuple[int, int]]:
    """
    Generate all unique view pairs for multi-view comparison.
    
    Args:
        num_views: Number of views (default 6 for Zero123++)
        
    Returns:
        List of (i, j) tuples where i < j
    """
    pairs = []
    for i in range(num_views):
        for j in range(i + 1, num_views):
            pairs.append((i, j))
    return pairs


def warp_image_with_depth(
    image: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    R_rel: torch.Tensor,
    t_rel: torch.Tensor,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Warp image from source view to target view using depth.
    
    Uses inverse depth-based warping:
    1. For each pixel in source, compute 3D point using depth
    2. Transform to target camera coordinates
    3. Project to target image plane
    
    Args:
        image: Source image (C, H, W) or (B, C, H, W)
        depth: Depth map (H, W) or (B, H, W)
        K: Intrinsic matrix (3, 3)
        R_rel: Relative rotation from source to target (3, 3)
        t_rel: Relative translation from source to target (3,)
        target_height: Output height (default: same as input)
        target_width: Output width (default: same as input)
        
    Returns:
        warped_image: Warped image (same shape as input)
        valid_mask: Binary mask of valid pixels (H, W) or (B, H, W)
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
        depth = depth.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, C, H, W = image.shape
    target_height = target_height or H
    target_width = target_width or W
    device = image.device
    
    # Create pixel grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Homogeneous pixel coordinates
    ones = torch.ones_like(x_coords)
    pixels_hom = torch.stack([x_coords, y_coords, ones], dim=-1)  # (H, W, 3)
    pixels_hom = pixels_hom.view(-1, 3).T  # (3, H*W)
    
    # Unproject to 3D
    K_inv = torch.inverse(K)
    rays = K_inv @ pixels_hom  # (3, H*W)
    
    # Scale by depth to get 3D points
    depth_flat = depth.view(B, -1)  # (B, H*W)
    points_3d = rays.unsqueeze(0) * depth_flat.unsqueeze(1)  # (B, 3, H*W)
    
    # Transform to target camera
    # P_target = R_rel @ P_source + t_rel
    points_target = R_rel @ points_3d + t_rel.view(1, 3, 1)  # (B, 3, H*W)
    
    # Project to target image
    points_proj = K @ points_target  # (B, 3, H*W)
    
    # Normalize by depth
    z = points_proj[:, 2:3, :]  # (B, 1, H*W)
    z = torch.clamp(z, min=1e-6)  # Avoid division by zero
    
    uv = points_proj[:, :2, :] / z  # (B, 2, H*W)
    
    # Normalize to [-1, 1] for grid_sample
    u_norm = 2 * uv[:, 0, :] / (target_width - 1) - 1
    v_norm = 2 * uv[:, 1, :] / (target_height - 1) - 1
    
    grid = torch.stack([u_norm, v_norm], dim=-1)  # (B, H*W, 2)
    grid = grid.view(B, H, W, 2)
    
    # Warp image
    warped = F.grid_sample(
        image, grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    
    # Create valid mask (points that project within bounds and have positive depth)
    valid_u = (uv[:, 0, :] >= 0) & (uv[:, 0, :] < target_width)
    valid_v = (uv[:, 1, :] >= 0) & (uv[:, 1, :] < target_height)
    valid_z = z.squeeze(1) > 0
    valid_mask = (valid_u & valid_v & valid_z).view(B, H, W)
    
    if squeeze_output:
        warped = warped.squeeze(0)
        valid_mask = valid_mask.squeeze(0)
    
    return warped, valid_mask


def compute_reprojection_error(
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    warped_image: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute L1 reprojection error between warped and target images.
    
    Args:
        source_image: Original source image (not used, for reference)
        target_image: Ground truth target image (C, H, W) or (B, C, H, W)
        warped_image: Source warped to target (same shape)
        mask: Optional valid pixel mask (H, W) or (B, H, W)
        
    Returns:
        error: Mean L1 error (scalar or (B,))
    """
    diff = torch.abs(target_image - warped_image)
    
    if mask is not None:
        # Expand mask to match image channels
        if diff.dim() == 4:
            mask = mask.unsqueeze(1).expand_as(diff)
        else:
            mask = mask.unsqueeze(0).expand_as(diff)
        
        # Masked mean
        masked_diff = diff * mask.float()
        error = masked_diff.sum() / (mask.float().sum() + 1e-6)
    else:
        error = diff.mean()
    
    return error


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Normalize image to [0, 1] range if not already.
    
    Args:
        image: Input image tensor
        
    Returns:
        Normalized image in [0, 1]
    """
    if image.max() > 1.0:
        return image / 255.0
    return image


def split_multiview_grid(
    grid_image: torch.Tensor,
    num_rows: int = 2,
    num_cols: int = 3
) -> torch.Tensor:
    """
    Split a multi-view grid image into individual views.
    
    Zero123++ outputs 6 views in a 3x2 grid (960x640 -> 6 x 320x320).
    
    Args:
        grid_image: Grid image (C, H, W) or (B, C, H, W)
        num_rows: Number of rows in grid
        num_cols: Number of columns in grid
        
    Returns:
        views: Individual views (num_views, C, view_H, view_W) or (B, ...)
    """
    if grid_image.dim() == 3:
        grid_image = grid_image.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    B, C, H, W = grid_image.shape
    view_h = H // num_rows
    view_w = W // num_cols
    
    views = []
    for row in range(num_rows):
        for col in range(num_cols):
            view = grid_image[
                :, :,
                row * view_h:(row + 1) * view_h,
                col * view_w:(col + 1) * view_w
            ]
            views.append(view)
    
    views = torch.stack(views, dim=1)  # (B, num_views, C, view_H, view_W)
    
    if squeeze:
        views = views.squeeze(0)
    
    return views
