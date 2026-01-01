"""
Quality Scoring and Feature Fusion Module.

This module implements quality-aware fusion of multiple rectified frames,
weighting frames by their estimated quality to produce a single high-quality
fused representation for super-resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class QualityScorer(nn.Module):
    """
    Quality scoring network that estimates the quality of each frame.
    
    The quality score reflects factors like sharpness, contrast, focus,
    and absence of motion blur. Higher quality frames contribute more
    to the fused representation.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        use_global_context: bool = True
    ):
        """
        Initialize the quality scorer.
        
        Args:
            in_channels: Number of input feature channels.
            hidden_dim: Hidden dimension for scoring network.
            use_global_context: Whether to use global context for scoring.
        """
        super().__init__()
        
        self.use_global_context = use_global_context
        
        # Local quality features
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Score prediction
        score_input_dim = hidden_dim * 2 if use_global_context else hidden_dim
        self.score_head = nn.Sequential(
            nn.Linear(score_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Quality score in [0, 1]
        )
        
        if use_global_context:
            # Global context aggregation
            self.global_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate quality score for a single frame.
        
        Args:
            x: Input feature map of shape (B, C, H, W).
            
        Returns:
            Quality scores of shape (B, 1) in range [0, 1].
        """
        batch_size = x.size(0)
        
        # Local features
        local_feat = self.local_conv(x)
        local_pooled = self.gap(local_feat).view(batch_size, -1)
        
        if self.use_global_context:
            # Global context
            global_feat = self.global_conv(x)
            global_pooled = self.gap(global_feat).view(batch_size, -1)
            
            # Concatenate
            features = torch.cat([local_pooled, global_pooled], dim=-1)
        else:
            features = local_pooled
        
        # Predict quality score
        score = self.score_head(features)
        
        return score


class FeatureFusion(nn.Module):
    """
    Feature fusion module that combines multiple frame features.
    
    Supports multiple fusion strategies:
        - Weighted average (using quality scores)
        - Attention-based fusion
        - Max pooling
    """
    
    def __init__(
        self,
        in_channels: int,
        fusion_type: str = 'weighted_avg',
        hidden_dim: int = 128
    ):
        """
        Initialize the feature fusion module.
        
        Args:
            in_channels: Number of input feature channels.
            fusion_type: Type of fusion ('weighted_avg', 'attention', 'max').
            hidden_dim: Hidden dimension for fusion operations.
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        self.in_channels = in_channels
        
        if fusion_type == 'attention':
            # Cross-frame attention for fusion
            self.query = nn.Conv2d(in_channels, hidden_dim, 1)
            self.key = nn.Conv2d(in_channels, hidden_dim, 1)
            self.value = nn.Conv2d(in_channels, in_channels, 1)
            self.scale = hidden_dim ** -0.5
    
    def forward(
        self, 
        features: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse features from multiple frames.
        
        Args:
            features: Feature tensor of shape (B, T, C, H, W).
            quality_scores: Quality scores of shape (B, T, 1) for weighted fusion.
            
        Returns:
            Fused feature map of shape (B, C, H, W).
        """
        B, T, C, H, W = features.shape
        
        if self.fusion_type == 'weighted_avg':
            if quality_scores is None:
                # Uniform weights if no quality scores
                weights = torch.ones(B, T, 1, 1, 1, device=features.device) / T
            else:
                # Normalize quality scores to sum to 1
                weights = F.softmax(quality_scores, dim=1)  # (B, T, 1)
                weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1, 1)
            
            # Weighted sum
            fused = (features * weights).sum(dim=1)  # (B, C, H, W)
            
        elif self.fusion_type == 'attention':
            # Reshape for attention computation
            features_flat = features.view(B * T, C, H, W)
            
            # Compute queries, keys, values
            q = self.query(features_flat).view(B, T, -1, H * W)  # (B, T, hidden, N)
            k = self.key(features_flat).view(B, T, -1, H * W)    # (B, T, hidden, N)
            v = self.value(features_flat).view(B, T, -1, H * W)  # (B, T, C, N)
            
            # Use first frame as query
            q_0 = q[:, 0]  # (B, hidden, N)
            
            # Compute attention across frames
            attn_weights = []
            for t in range(T):
                attn = torch.einsum('bhw,bhw->bw', q_0, k[:, t]) * self.scale
                attn_weights.append(attn)
            
            attn_weights = torch.stack(attn_weights, dim=1)  # (B, T, N)
            attn_weights = F.softmax(attn_weights, dim=1)
            
            # Weighted sum of values
            fused = torch.zeros(B, C, H * W, device=features.device)
            for t in range(T):
                weight = attn_weights[:, t].unsqueeze(1)  # (B, 1, N)
                fused = fused + v[:, t] * weight
            
            fused = fused.view(B, C, H, W)
            
        elif self.fusion_type == 'max':
            # Max pooling across frames
            fused = features.max(dim=1)[0]  # (B, C, H, W)
            
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        return fused


class QualityScorerFusion(nn.Module):
    """
    Combined module for quality scoring and feature fusion.
    
    This module first estimates the quality of each input frame,
    then fuses the frames using quality-weighted averaging to produce
    a single high-quality feature representation.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        fusion_type: str = 'weighted_avg',
        use_global_context: bool = True
    ):
        """
        Initialize the quality scorer and fusion module.
        
        Args:
            in_channels: Number of input feature channels.
            hidden_dim: Hidden dimension for quality scorer.
            fusion_type: Type of feature fusion.
            use_global_context: Whether to use global context for scoring.
        """
        super().__init__()
        
        self.quality_scorer = QualityScorer(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            use_global_context=use_global_context
        )
        
        self.feature_fusion = FeatureFusion(
            in_channels=in_channels,
            fusion_type=fusion_type,
            hidden_dim=hidden_dim
        )
    
    def forward(
        self, 
        features: torch.Tensor,
        return_scores: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Score and fuse multiple frame features.
        
        Args:
            features: Feature tensor of shape (B, T, C, H, W).
            return_scores: Whether to return quality scores.
            
        Returns:
            Tuple containing:
                - Fused feature map of shape (B, C, H, W)
                - (optional) Quality scores of shape (B, T, 1)
        """
        B, T, C, H, W = features.shape
        
        # Compute quality scores for each frame
        features_flat = features.view(B * T, C, H, W)
        scores = self.quality_scorer(features_flat)  # (B*T, 1)
        scores = scores.view(B, T, 1)
        
        # Fuse features using quality scores
        fused = self.feature_fusion(features, scores)
        
        if return_scores:
            return fused, scores
        return fused, None


class TemporalFusion(nn.Module):
    """
    Temporal fusion using 3D convolutions.
    
    Alternative fusion approach that uses 3D convolutions to capture
    temporal relationships across frames before collapsing to a single
    feature map.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_frames: int = 5,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim, (num_frames, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Conv2d(hidden_dim, in_channels, 1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Fuse frames using temporal convolutions.
        
        Args:
            features: Feature tensor of shape (B, T, C, H, W).
            
        Returns:
            Fused feature map of shape (B, C, H, W).
        """
        # Rearrange to (B, C, T, H, W) for 3D conv
        features = features.permute(0, 2, 1, 3, 4)
        
        # Temporal convolutions
        fused = self.temporal_conv(features)  # (B, hidden_dim, 1, H, W)
        fused = fused.squeeze(2)  # (B, hidden_dim, H, W)
        
        # Project back to original channels
        fused = self.project(fused)
        
        return fused
