"""
Semantic Identity Reward (R_sem).

Implements semantic consistency measurement using DINOv2 features
to ensure generated views maintain identity with the conditioning image.

Reference: Section 4.2, "Semantic Identity" in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import warnings

try:
    from transformers import AutoImageProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SemanticReward(nn.Module):
    """
    Computes semantic identity reward using DINOv2.
    
    R_sem = (1/6) * Σ_(v=1)^6 CosineSim(φ(y^(v)), φ(c))
    
    Where:
    - φ: DINOv2 [CLS] token extractor
    - y^(v): v-th generated view
    - c: Input conditioning image
    
    Args:
        model_name: Hugging Face model ID for DINOv2
        device: Torch device for computation
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model specifications
        self.input_size = 224  # DINOv2 expects 224x224
        self.feature_dim = 1024  # ViT-L output dimension
        
        # Load DINOv2 model
        self._load_model()
    
    def _load_model(self):
        """Load pretrained DINOv2 model."""
        if not TRANSFORMERS_AVAILABLE:
            warnings.warn("transformers library not available. Using fallback.")
            self.processor = None
            self.model = None
            return
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Update feature dim based on actual model
            if hasattr(self.model.config, 'hidden_size'):
                self.feature_dim = self.model.config.hidden_size
                
        except Exception as e:
            warnings.warn(f"Failed to load DINOv2 model '{self.model_name}': {e}")
            self.processor = None
            self.model = None
    
    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for DINOv2.
        
        Args:
            images: Input images (B, C, H, W) in [0, 1] range
            
        Returns:
            Preprocessed images (B, C, 224, 224)
        """
        # Resize to 224x224
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize with ImageNet stats (DINOv2 uses ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        
        normalized = (images - mean) / std
        
        return normalized
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract [CLS] token features from images.
        
        Args:
            images: Input images (B, C, H, W) in [0, 1] range
            
        Returns:
            features: [CLS] token features (B, feature_dim)
        """
        B = images.shape[0]
        
        if self.model is None:
            # Fallback: return random features
            warnings.warn("Using placeholder feature extraction.")
            return torch.randn(B, self.feature_dim, device=images.device)
        
        # Preprocess images
        preprocessed = self._preprocess(images)
        
        # Forward through DINOv2
        outputs = self.model(preprocessed)
        
        # Extract [CLS] token (first token)
        if hasattr(outputs, 'last_hidden_state'):
            cls_features = outputs.last_hidden_state[:, 0, :]  # (B, feature_dim)
        elif hasattr(outputs, 'pooler_output'):
            cls_features = outputs.pooler_output
        else:
            # Fallback to last hidden state
            cls_features = outputs[0][:, 0, :]
        
        return cls_features
    
    def cosine_similarity(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between feature vectors.
        
        Args:
            feat_a: Features (B, D) or (D,)
            feat_b: Features (B, D) or (D,)
            
        Returns:
            similarity: Cosine similarity scores (B,) or scalar
        """
        # Normalize
        feat_a = F.normalize(feat_a, p=2, dim=-1)
        feat_b = F.normalize(feat_b, p=2, dim=-1)
        
        # Dot product
        similarity = (feat_a * feat_b).sum(dim=-1)
        
        return similarity
    
    def forward(
        self,
        generated_views: torch.Tensor,
        condition_image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute semantic identity reward.
        
        Args:
            generated_views: Generated multi-view images (B, num_views, C, H, W)
            condition_image: Conditioning image (B, C, H, W)
            
        Returns:
            Dictionary with:
            - 'reward': R_sem score (average similarity)
            - 'per_view_similarity': Similarity for each view (B, num_views)
            - 'condition_features': Condition image features (B, D)
        """
        B, num_views, C, H, W = generated_views.shape
        device = generated_views.device
        
        # Extract features from condition image
        condition_features = self.extract_features(condition_image)  # (B, D)
        
        # Extract features from each generated view
        # Reshape to (B * num_views, C, H, W)
        flat_views = generated_views.view(B * num_views, C, H, W)
        view_features = self.extract_features(flat_views)  # (B * num_views, D)
        view_features = view_features.view(B, num_views, -1)  # (B, num_views, D)
        
        # Compute cosine similarity for each view with condition
        similarities = []
        for v in range(num_views):
            sim = self.cosine_similarity(
                view_features[:, v, :],  # (B, D)
                condition_features       # (B, D)
            )
            similarities.append(sim)
        
        per_view_similarity = torch.stack(similarities, dim=1)  # (B, num_views)
        
        # Average similarity across views
        mean_similarity = per_view_similarity.mean(dim=1)  # (B,)
        
        # Return mean across batch for scalar reward
        reward = mean_similarity.mean()
        
        return {
            'reward': reward,
            'per_view_similarity': per_view_similarity,
            'condition_features': condition_features,
            'view_features': view_features
        }
    
    def __call__(
        self,
        generated_views: torch.Tensor,
        condition_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified interface returning just the reward score.
        
        Args:
            generated_views: Multi-view images (B, num_views, C, H, W)
            condition_image: Conditioning image (B, C, H, W)
            
        Returns:
            reward: R_sem score (scalar)
        """
        result = self.forward(generated_views, condition_image)
        return result['reward']


class PatchSemanticReward(SemanticReward):
    """
    Extended semantic reward using patch-level features for finer-grained comparison.
    
    Uses all patch tokens from DINOv2, not just [CLS] token.
    This can capture local semantic details better but is more expensive.
    """
    
    @torch.no_grad()
    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract all patch token features from images.
        
        Args:
            images: Input images (B, C, H, W)
            
        Returns:
            features: Patch features (B, num_patches, feature_dim)
        """
        if self.model is None:
            warnings.warn("Using placeholder patch feature extraction.")
            num_patches = (self.input_size // 14) ** 2  # DINOv2 uses 14x14 patches
            return torch.randn(images.shape[0], num_patches, self.feature_dim, device=images.device)
        
        preprocessed = self._preprocess(images)
        outputs = self.model(preprocessed)
        
        # Get all tokens except [CLS]
        if hasattr(outputs, 'last_hidden_state'):
            patch_features = outputs.last_hidden_state[:, 1:, :]  # Skip CLS token
        else:
            patch_features = outputs[0][:, 1:, :]
        
        return patch_features
    
    def forward_patch(
        self,
        generated_views: torch.Tensor,
        condition_image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute patch-level semantic reward.
        
        Uses maximum similarity across patches for robustness.
        """
        B, num_views, C, H, W = generated_views.shape
        
        # Extract patch features from condition
        condition_patches = self.extract_patch_features(condition_image)  # (B, P, D)
        
        # Extract for each view
        flat_views = generated_views.view(B * num_views, C, H, W)
        view_patches = self.extract_patch_features(flat_views)
        view_patches = view_patches.view(B, num_views, -1, self.feature_dim)  # (B, V, P, D)
        
        # Compute patch-level similarities
        # For each view, compute similarity matrix with condition patches
        per_view_similarity = []
        
        for v in range(num_views):
            # (B, P, D) vs (B, P, D) -> cosine similarity matrix
            cond_norm = F.normalize(condition_patches, p=2, dim=-1)  # (B, P, D)
            view_norm = F.normalize(view_patches[:, v], p=2, dim=-1)  # (B, P, D)
            
            # Compute all pairwise similarities
            sim_matrix = torch.bmm(view_norm, cond_norm.transpose(1, 2))  # (B, P, P)
            
            # Max similarity for each view patch to any condition patch
            max_sim = sim_matrix.max(dim=-1).values.mean(dim=-1)  # (B,)
            per_view_similarity.append(max_sim)
        
        per_view_similarity = torch.stack(per_view_similarity, dim=1)  # (B, V)
        reward = per_view_similarity.mean()
        
        return {
            'reward': reward,
            'per_view_similarity': per_view_similarity
        }
