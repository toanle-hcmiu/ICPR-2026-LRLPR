"""
Text-Prior Guided Generator for License Plate Super-Resolution.

This module implements the main generator that combines:
1. Multi-frame fusion (ICAM from LP-Diff)
2. Text prior extraction (from PARSeq)
3. Sequential Residual Blocks (from TSRN)
4. Cross-attention for text-image fusion

Based on TPGSR (IEEE TIP 2021) + LP-Diff (CVPR 2025) + TSRN (ECCV 2020)

Author: New Architecture Implementation (2026-01-27)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Import modules
from .multi_frame_fusion import InterFrameCrossAttentionModule
from .text_prior import TextPriorExtractor
from .sequential_blocks import SequentialResidualBlock, SequentialResidualBlocks


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention for text-image fusion.

    Uses text features to modulate image features at each spatial location.
    The text features provide character-level guidance for generation.

    Args:
        image_channels: Number of image feature channels
        text_channels: Number of text feature channels
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        image_channels: int = 64,
        text_channels: int = 256,
        num_heads: int = 4
    ):
        super().__init__()

        self.image_channels = image_channels
        self.text_channels = text_channels
        self.num_heads = num_heads

        # Query from image, Key/Value from text
        self.to_q = nn.Conv2d(image_channels, image_channels, 1)
        self.to_k = nn.Conv2d(text_channels, image_channels, 1)
        self.to_v = nn.Conv2d(text_channels, image_channels, 1)

        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(image_channels, image_channels, 1),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-attention from text to image.

        Args:
            image_features: Image features of shape (B, C, H, W)
            text_features: Text features of shape (B, text_dim, H', W')

        Returns:
            Text-modulated image features of shape (B, C, H, W)
        """
        B, C, H, W = image_features.shape

        # Reshape text features to match spatial size if needed
        if text_features.shape[2:] != (H, W):
            text_features = F.interpolate(
                text_features,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        # Q, K, V
        q = self.to_q(image_features)  # (B, C, H, W)
        k = self.to_k(text_features)   # (B, C, H, W)
        v = self.to_v(text_features)   # (B, C, H, W)

        # Reshape for attention
        # (B, C, H, W) -> (B, H*W, C)
        q_flat = q.reshape(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        k_flat = k.reshape(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        v_flat = v.reshape(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        # Scaled dot-product attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.bmm(q_flat, k_flat.transpose(1, 2)) * scale  # (B, H*W, H*W)
        attn = F.softmax(attn, dim=-1)

        # Apply attention: (B, H*W, H*W) @ (B, H*W, C) -> (B, H*W, C)
        out = torch.bmm(attn, v_flat)  # (B, H*W, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)

        # Output projection with residual
        out = self.out_conv(out) + image_features

        return out


class TextPriorGuidedGenerator(nn.Module):
    """
    Text-Prior Guided Generator for License Plate Super-Resolution.

    This is the main generator that combines all components:
    1. Multi-frame fusion (ICAM)
    2. Text prior extraction (from PARSeq)
    3. Sequential Residual Blocks (with BiLSTM)
    4. Cross-attention for text-image fusion

    Key design:
    - Training: Uses text prior from frozen PARSeq
    - Inference: No text input (pure LR -> HR)
    - This prevents mode collapse while maintaining practical inference

    Args:
        num_frames: Number of input frames (default: 5)
        num_srb: Number of Sequential Residual Blocks (default: 4)
        channels: Number of feature channels (default: 64)
        lstm_hidden: Hidden size for LSTM in SRB (default: 32)
        upscale_factor: Upscaling factor (default: 4 for 16x48 -> 64x192)
    """

    def __init__(
        self,
        num_frames: int = 5,
        num_srb: int = 4,
        channels: int = 64,
        lstm_hidden: int = 32,
        upscale_factor: int = 4
    ):
        super().__init__()

        self.num_frames = num_frames
        self.num_srb = num_srb
        self.channels = channels
        self.upscale_factor = upscale_factor

        # Initial feature extraction from first frame
        self.conv_in = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

        # Multi-frame fusion module
        self.frame_fusion = InterFrameCrossAttentionModule(
            channels=channels,
            num_frames=num_frames
        )

        # Text prior extractor (will be set during model initialization)
        self.text_extractor = None

        # Sequential Residual Blocks
        self.srb_blocks = nn.ModuleList([
            SequentialResidualBlock(channels, lstm_hidden)
            for _ in range(num_srb)
        ])

        # Cross-attention for text guidance (training only)
        self.text_cross_attn = CrossAttentionBlock(channels, 256)

        # Upsampling
        # 16x48 -> 64x192 = 4x upscale in each dimension
        # Two 2x upsampling steps
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 2x upscale
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # 2x upscale
            nn.ReLU(inplace=True),
        )

        # Final reconstruction layers
        self.conv_out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        )

        # For text-guided mode (alternative SRBs with text guidance)
        self.use_text_guided_srb = False

    def set_text_extractor(self, parseq_model: nn.Module) -> None:
        """
        Set the frozen PARSeq for text prior extraction.

        Args:
            parseq_model: Pretrained PARSeq model
        """
        self.text_extractor = TextPriorExtractor(parseq_model)

    def set_text_guided_mode(self, use: bool = True) -> None:
        """
        Enable/disable text-guided SRB mode.

        Args:
            use: Whether to use text-guided SRBs
        """
        self.use_text_guided_srb = use

    def forward(
        self,
        lr_frames: torch.Tensor,
        use_text_prior: bool = True,
        return_text_prior: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate HR image from LR frames.

        Args:
            lr_frames: Multi-frame LR input of shape (B, T, 3, H, W) or (B, 3, H, W)
            use_text_prior: Whether to use text prior guidance (default: True)
                           Should be True during training, False during inference
            return_text_prior: Whether to return text prior for visualization

        Returns:
            hr_image: Super-resolved image of shape (B, 3, 64, 192)
            text_prior: (Optional) Character probabilities of shape (B, 7, vocab_size)
        """
        # Handle both single frame and multi-frame input
        if lr_frames.dim() == 4:  # (B, 3, H, W)
            lr_frames = lr_frames.unsqueeze(1)  # (B, 1, 3, H, W)

        B, T, C, H_in, W_in = lr_frames.shape

        # Encode all frames first (needed for multi-frame fusion)
        encoded_frames = []
        for t in range(T):
            feat = self.conv_in(lr_frames[:, t])  # (B, channels, H, W)
            encoded_frames.append(feat)
        encoded_frames = torch.stack(encoded_frames, dim=1)  # (B, T, channels, H, W)

        # Use first frame as base
        x = encoded_frames[:, 0]  # (B, channels, H, W)

        # Multi-frame fusion (if multiple frames available)
        if T > 1:
            fused = self.frame_fusion(encoded_frames)
            x = x + fused  # Residual connection

        # Extract text prior (ONLY during training)
        text_prior, text_features = None, None
        if use_text_prior and self.text_extractor is not None:
            with torch.no_grad():  # Don't backprop through PARSeq
                text_prior, text_features = self.text_extractor(lr_frames)

        # Apply Sequential Residual Blocks
        for srb in self.srb_blocks:
            x = srb(x)

            # Apply text cross-attention (ONLY during training)
            if text_features is not None:
                x = self.text_cross_attn(x, text_features)

        # Upsample to HR dimensions
        x = self.upsample(x)  # (B, channels, H*4, W*4)

        # Final output
        hr_image = self.conv_out(x)  # (B, 3, H*4, W*4)

        # Apply tanh for [-1, 1] range
        hr_image = torch.tanh(hr_image)

        if return_text_prior and text_prior is not None:
            return hr_image, text_prior

        return hr_image

    def get_trainable_params(self) -> list:
        """
        Get trainable parameters for optimizer.

        Note: Text extractor parameters are NOT included as PARSeq is frozen.

        Returns:
            List of parameter groups
        """
        param_groups = [
            {'params': self.conv_in.parameters(), 'name': 'conv_in'},
            {'params': self.frame_fusion.parameters(), 'name': 'frame_fusion'},
            {'params': self.srb_blocks.parameters(), 'name': 'srb_blocks'},
            {'params': self.text_cross_attn.parameters(), 'name': 'text_cross_attn'},
            {'params': self.upsample.parameters(), 'name': 'upsample'},
            {'params': self.conv_out.parameters(), 'name': 'conv_out'},
        ]

        # Include text extractor projection layers (not PARSeq itself)
        if self.text_extractor is not None:
            param_groups.extend(self.text_extractor.get_trainable_params())

        return param_groups

    def init_weights(self) -> None:
        """
        Initialize network weights.

        Uses Xavier initialization for conv layers and zeros for biases.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param, gain=0.02)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)


class LightweightTPGenerator(nn.Module):
    """
    Lightweight variant of Text-Prior Guided Generator.

    Fewer SRB blocks and smaller channel dimensions for faster training.
    Useful for initial experiments.

    Args:
        num_frames: Number of input frames (default: 5)
        num_srb: Number of Sequential Residual Blocks (default: 2)
        channels: Number of feature channels (default: 32)
    """

    def __init__(
        self,
        num_frames: int = 5,
        num_srb: int = 2,
        channels: int = 32
    ):
        super().__init__()

        self.num_frames = num_frames
        self.channels = channels

        # Simplified architecture
        self.conv_in = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Multi-frame fusion
        self.frame_fusion = InterFrameCrossAttentionModule(channels, num_frames)

        # Fewer SRB blocks
        from .sequential_blocks import SequentialResidualBlock
        self.srb_blocks = nn.ModuleList([
            SequentialResidualBlock(channels, lstm_hidden=16)
            for _ in range(num_srb)
        ])

        # Text extractor placeholder
        self.text_extractor = None

        # Simple upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

        self.conv_out = nn.Conv2d(channels, 3, 3, padding=1)

    def set_text_extractor(self, parseq_model: nn.Module) -> None:
        """Set the frozen PARSeq for text prior extraction."""
        from .text_prior import TextPriorExtractor
        self.text_extractor = TextPriorExtractor(parseq_model, feature_dim=128)

    def forward(
        self,
        lr_frames: torch.Tensor,
        use_text_prior: bool = True
    ) -> torch.Tensor:
        """Forward pass."""
        if lr_frames.dim() == 4:
            lr_frames = lr_frames.unsqueeze(1)

        x = self.conv_in(lr_frames[:, 0])

        if lr_frames.shape[1] > 1:
            fused = self.frame_fusion(lr_frames)
            x = x + fused

        text_features = None
        if use_text_prior and self.text_extractor is not None:
            with torch.no_grad():
                _, text_features = self.text_extractor(lr_frames)

        for srb in self.srb_blocks:
            x = srb(x)
            if text_features is not None:
                # Simple feature modulation instead of full cross-attention
                B, C, H, W = x.shape
                text_feat = F.interpolate(text_features, size=(H, W), mode='bilinear')
                x = x + text_feat * 0.1  # Small modulation

        x = self.upsample(x)
        x = self.conv_out(x)
        return torch.tanh(x)


def create_tp_guided_generator(
    num_frames: int = 5,
    num_srb: int = 4,
    channels: int = 64,
    lstm_hidden: int = 32,
    lightweight: bool = False
) -> nn.Module:
    """
    Factory function to create text-prior guided generator.

    Args:
        num_frames: Number of input frames
        num_srb: Number of SRB blocks
        channels: Number of feature channels
        lstm_hidden: Hidden size for LSTM
        lightweight: Whether to use lightweight variant

    Returns:
        Generator module
    """
    if lightweight:
        return LightweightTPGenerator(num_frames, num_srb, channels)
    else:
        return TextPriorGuidedGenerator(num_frames, num_srb, channels, lstm_hidden)


# Test function
def _test_tp_guided_generator():
    """Quick test to verify generator works correctly."""
    import torch

    # Create generator
    gen = TextPriorGuidedGenerator(num_frames=5, num_srb=2, channels=32)

    # Test data
    B, T, C, H, W = 2, 5, 3, 16, 48
    lr_frames = torch.randn(B, T, C, H, W)

    # Forward pass (without text prior - inference mode)
    output = gen(lr_frames, use_text_prior=False)

    print(f"Input shape: {lr_frames.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Output requires grad: {output.requires_grad}")

    # Test parameter counting
    total_params = sum(p.numel() for p in gen.parameters())
    trainable_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("✓ Text-Prior Guided Generator test passed!")


if __name__ == "__main__":
    _test_tp_guided_generator()
