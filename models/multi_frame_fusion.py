"""
Multi-Frame Fusion Module for License Plate Super-Resolution.

This module implements the Inter-frame Cross-Attention Module (ICAM) from LP-Diff,
which fuses temporal information across consecutive frames using cross-attention.

Reference: LP-Diff (CVPR 2025) - "LP-Diff: Towards Improved Restoration of
Real-World Degraded License Plate"

Author: New Architecture Implementation (2026-01-27)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class InterFrameCrossAttentionModule(nn.Module):
    """
    Inter-frame Cross-Attention Module (ICAM) from LP-Diff.

    Fuses temporal information across consecutive frames using cross-attention.
    This is more effective than simple averaging because it learns which
    frames contribute most to each spatial location.

    Args:
        channels: Number of feature channels
        num_frames: Number of input frames (typically 5)
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        channels: int = 64,
        num_frames: int = 5,
        num_heads: int = 4
    ):
        super().__init__()

        self.channels = channels
        self.num_frames = num_frames
        self.num_heads = num_heads

        # Multi-head temporal attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(channels)

        # Fusion convolution
        # Input: concatenated frames (T*C) + attention output (T*C) = 2*T*C channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * num_frames * 2, channels * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, 1)
        )

        # Residual connection
        self.residual_conv = nn.Conv2d(channels * num_frames, channels, 1)

    def forward(
        self,
        frames: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse multi-frame features using temporal cross-attention.

        Args:
            frames: Input frames of shape (B, T, C, H, W) where T is num_frames
            return_attention: Whether to return attention weights for visualization

        Returns:
            Fused features of shape (B, C, H, W)
            Optional: Attention weights of shape (B, H*W, T, T)
        """
        B, T, C, H, W = frames.shape

        # Method 1: Concatenate all frames (simple baseline)
        # This is used as input to the attention mechanism
        frames_concat = frames.reshape(B, T * C, H, W)

        # Method 2: Temporal attention (learn which frames to use)
        # Reshape for attention: (B, T, C, H, W) -> (B, H*W, T, C)
        x = frames.permute(0, 3, 4, 1, 2)  # (B, H, W, T, C)
        x = x.reshape(B, H * W, T, C)  # (B, H*W, T, C)

        # Apply temporal self-attention at each spatial location
        # Each spatial location attends across all frames
        attn_input = x.reshape(B * H * W, T, C)

        attn_output, attn_weights = self.temporal_attn(
            attn_input, attn_input, attn_input,
            need_weights=True
        )  # (B*H*W, T, C), (B*H*W, T, T)

        # Reshape back
        attn_output = attn_output.reshape(B, H, W, T, C)
        attn_output = attn_output.permute(0, 3, 4, 1, 2)  # (B, T, C, H, W)

        # Layer norm - reshape to (B*T*H*W, C) for proper normalization
        attn_output = attn_output.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        attn_output = attn_output.reshape(B * T * H * W, C)
        attn_output = self.norm(attn_output)  # Normalize over C dimension
        attn_output = attn_output.reshape(B, T, H, W, C)
        attn_output = attn_output.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)

        # Reshape and concatenate
        attn_fused = attn_output.reshape(B, T * C, H, W)

        # Combine concatenation and attention
        combined = torch.cat([frames_concat, attn_fused], dim=1)

        # Final fusion
        output = self.fusion_conv(combined)

        # Residual connection from simple average
        avg_output = frames.mean(dim=1)  # (B, C, H, W)
        residual = self.residual_conv(frames_concat)

        output = output + residual + avg_output

        if return_attention:
            # Reshape attention weights for visualization
            attn_weights = attn_weights.reshape(B, H * W, T, T)
            return output, attn_weights

        return output


class TemporalFusionModule(nn.Module):
    """
    Simplified temporal fusion module using 3D convolution.

    Alternative to ICAM that uses 3D convolutions for temporal fusion.
    More computationally efficient but less expressive.

    Args:
        channels: Number of feature channels
        num_frames: Number of input frames
    """

    def __init__(self, channels: int = 64, num_frames: int = 5):
        super().__init__()

        self.num_frames = num_frames
        self.channels = channels

        # 3D convolution for temporal-spatial fusion
        # Kernel: (T, H, W) = (num_frames, 3, 3)
        self.conv3d = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(num_frames, 3, 3),
            padding=(0, 1, 1),
            bias=False
        )

        # Channel attention for temporal features
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Fuse multi-frame features using 3D convolution.

        Args:
            frames: Input frames of shape (B, T, C, H, W)

        Returns:
            Fused features of shape (B, C, H, W)
        """
        B, T, C, H, W = frames.shape

        # Reshape for 3D conv: (B, T, C, H, W) -> (B, C, T, H, W)
        x = frames.permute(0, 2, 1, 3, 4)

        # Apply 3D convolution
        x = self.conv3d(x)  # (B, C, 1, H, W)
        x = x.squeeze(2)  # (B, C, H, W)

        # Channel attention
        attn = self.channel_attn(x)
        x = x * attn

        return x


class AdaptiveFrameFusion(nn.Module):
    """
    Adaptive frame fusion that learns to weight each frame dynamically.

    Uses spatial attention to determine which frames contribute most
    to each spatial location.

    Args:
        channels: Number of feature channels
        num_frames: Number of input frames
    """

    def __init__(self, channels: int = 64, num_frames: int = 5):
        super().__init__()

        self.num_frames = num_frames
        self.channels = channels

        # Spatial attention for frame weighting
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels * num_frames, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Frame-wise feature extraction
        self.frame_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1)
            ) for _ in range(num_frames)
        ])

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * num_frames, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Fuse frames with adaptive spatial weighting.

        Args:
            frames: Input frames of shape (B, T, C, H, W)

        Returns:
            Fused features of shape (B, C, H, W)
        """
        B, T, C, H, W = frames.shape

        # Apply frame-wise convolutions
        frame_features = []
        for t in range(T):
            feat = self.frame_convs[t](frames[:, t])  # (B, C, H, W)
            frame_features.append(feat)

        # Concatenate all frames
        concat = torch.cat(frame_features, dim=1)  # (B, T*C, H, W)

        # Compute spatial attention weights
        attn_weights = self.spatial_attn(concat)  # (B, 1, H, W)

        # Apply attention weights to each frame
        weighted_frames = []
        for t in range(T):
            weighted = frames[:, t] * attn_weights
            weighted_frames.append(weighted)

        # Concatenate and fuse
        weighted_concat = torch.cat(weighted_frames, dim=1)
        output = self.fusion(weighted_concat)

        return output


def create_multi_frame_fusion(
    fusion_type: str = 'icam',
    channels: int = 64,
    num_frames: int = 5,
    num_heads: int = 4
) -> nn.Module:
    """
    Factory function to create multi-frame fusion module.

    Args:
        fusion_type: Type of fusion ('icam', '3d', 'adaptive')
        channels: Number of feature channels
        num_frames: Number of input frames
        num_heads: Number of attention heads (for icam)

    Returns:
        Multi-frame fusion module
    """
    if fusion_type == 'icam':
        return InterFrameCrossAttentionModule(channels, num_frames, num_heads)
    elif fusion_type == '3d':
        return TemporalFusionModule(channels, num_frames)
    elif fusion_type == 'adaptive':
        return AdaptiveFrameFusion(channels, num_frames)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


# Test function
def _test_multi_frame_fusion():
    """Quick test to verify multi-frame fusion works correctly."""
    import torch

    batch_size = 2
    num_frames = 5
    channels = 64
    h, w = 16, 48

    # Create test input
    frames = torch.randn(batch_size, num_frames, channels, h, w)

    # Test ICAM
    fusion = InterFrameCrossAttentionModule(channels, num_frames)
    output = fusion(frames)

    print(f"Input shape: {frames.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output requires grad: {output.requires_grad}")

    # Test with attention return
    output, attn = fusion(frames, return_attention=True)
    print(f"Attention weights shape: {attn.shape}")

    # Test 3D fusion
    fusion_3d = TemporalFusionModule(channels, num_frames)
    output_3d = fusion_3d(frames)
    print(f"3D fusion output shape: {output_3d.shape}")

    # Test adaptive fusion
    fusion_adaptive = AdaptiveFrameFusion(channels, num_frames)
    output_adaptive = fusion_adaptive(frames)
    print(f"Adaptive fusion output shape: {output_adaptive.shape}")

    print("✓ Multi-frame fusion test passed!")


if __name__ == "__main__":
    _test_multi_frame_fusion()
