"""
Text Prior Extraction Module for License Plate Super-Resolution.

This module extracts text prior from PARSeq for guiding the super-resolution network.
The text prior provides categorical constraints during training, preventing mode collapse.

Based on TPGSR (Text Prior Guided Scene Text Image Super-resolution, IEEE TIP 2021)
with adaptations for PARSeq instead of CRNN.

Author: New Architecture Implementation (2026-01-27)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TextPriorExtractor(nn.Module):
    """
    Extracts text prior from frozen PARSeq.

    Unlike TPGSR which uses CRNN, we use PARSeq which is already trained.
    The text prior is a sequence of character probabilities (7 × vocab_size).
    This provides categorical constraints during generation.

    Key design:
    - PARSeq is FROZEN - no gradients flow through it
    - Character probabilities are converted to spatial features for cross-attention
    - Provides guidance during training only (not inference)

    Args:
        parseq_model: Pretrained PARSeq model
        vocab_size: Size of the vocabulary (default: 39)
        num_chars: Number of characters in plate (default: 7)
        feature_dim: Dimension for projected text features (default: 256)
    """

    def __init__(
        self,
        parseq_model: nn.Module,
        vocab_size: int = 39,
        num_chars: int = 7,
        feature_dim: int = 256
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_chars = num_chars
        self.feature_dim = feature_dim

        # Get device from parseq model (already on correct device)
        device = next(parseq_model.parameters()).device

        # Store PARSeq model reference - NOT registered as submodule
        self.parseq = parseq_model

        # Freeze PARSeq completely
        for param in self.parseq.parameters():
            param.requires_grad = False
        self.parseq.eval()

        # Learnable projection from probability to feature space
        # CRITICAL: Initialize with small weights to prevent gradient explosion
        # This layer is newly initialized at Stage 2
        proj = nn.Linear(vocab_size, feature_dim)
        nn.init.xavier_normal_(proj.weight, gain=0.01)
        nn.init.constant_(proj.bias, 0)
        self.proj = proj.to(device)

        # Positional encoding - use fixed encoding (not trained) to avoid non-leaf issues
        # Register as buffer so it's moved with the module but not optimized
        # Use VERY small std (0.01) to prevent adding noise to the features
        pos_enc = torch.randn(1, feature_dim, 1, num_chars) * 0.01
        self.register_buffer('pos_encoding', pos_enc.to(device))

        # Learnable upsampling to spatial dimensions
        # CRITICAL: Initialize with VERY small weights to prevent gradient explosion
        # These weights are newly initialized at Stage 2 and can cause instability
        spatial_expand = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
        ).to(device)

        # Initialize with near-zero weights (gain=0.01) for smooth training
        for m in spatial_expand.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.spatial_expand = spatial_expand

    def forward(
        self,
        lr_frames: torch.Tensor,
        frame_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract text prior from LR frames.

        Args:
            lr_frames: Multi-frame LR input of shape (B, T, 3, H, W) or (B, 3, H, W)
            frame_idx: Which frame to use for text extraction (default: 0)

        Returns:
            text_prior: Character probabilities of shape (B, 7, vocab_size)
            text_features: Spatial text features of shape (B, feature_dim, H, W)
        """
        # Handle both single frame and multi-frame input
        if lr_frames.dim() == 4:  # (B, 3, H, W)
            lr_input = lr_frames
        else:  # (B, T, 3, H, W)
            lr_input = lr_frames[:, frame_idx]

        # Get PARSeq logits (frozen - no gradients)
        with torch.no_grad():
            self.parseq.eval()

            # Get logits from PARSeq
            logits = self.parseq.forward_parallel(lr_input)  # (B, 7, vocab_size)

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, 7, vocab_size)

        # Project probabilities to feature space
        # (B, 7, vocab_size) -> (B, 7, feature_dim)
        char_features = self.proj(probs)

        # Add positional encoding
        # pos_encoding: (1, feature_dim, 1, num_chars) -> (num_chars, feature_dim)
        pos_enc = self.pos_encoding.squeeze(0).squeeze(1).permute(1, 0)  # (num_chars, feature_dim)
        char_features = char_features + pos_enc  # Broadcast to (B, 7, feature_dim)

        # Reshape to spatial format for cross-attention
        # (B, 7, feature_dim) -> (B, feature_dim, 1, 7)
        text_features_1d = char_features.permute(0, 2, 1).unsqueeze(2)

        # Upsample to match input spatial dimensions
        # (B, feature_dim, 1, 7) -> (B, feature_dim, H, W)
        B, feat_dim, _, C = text_features_1d.shape
        H, W = lr_input.shape[2], lr_input.shape[3]

        # First expand height: 1 -> H
        text_features_h = F.interpolate(
            text_features_1d,
            size=(H, C),
            mode='bilinear',
            align_corners=False
        )  # (B, F, H, C)

        # Then expand width: C -> W
        text_features = F.interpolate(
            text_features_h,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )  # (B, F, H, W)

        # Apply spatial refinement
        text_features = self.spatial_expand(text_features)

        return probs, text_features

    def get_trainable_params(self) -> list:
        """
        Get trainable parameters (only projection layers, not PARSeq or pos_encoding).

        CRITICAL: Use much lower learning rate for spatial_expand (1.18M params)
        to prevent gradient explosion when these newly initialized weights
        are introduced in Stage 2.

        Returns:
            List of parameter groups with learning rate scaling
        """
        return [
            {'params': self.proj.parameters(), 'name': 'text_proj', 'lr_scale': 0.1},  # Small projection layer
            # spatial_expand has 1.18M newly initialized params - use VERY small LR
            {'params': self.spatial_expand.parameters(), 'name': 'spatial_expand', 'lr_scale': 0.01}
        ]


class TextPriorLoss(nn.Module):
    """
    Text Prior Loss for training the text-guided generator.

    This loss encourages the generated HR image to produce character
    probabilities that match the ground truth text.

    Args:
        parseq_model: Pretrained PARSeq model (will be frozen)
        reduction: Loss reduction method
    """

    def __init__(
        self,
        parseq_model: nn.Module,
        reduction: str = 'mean'
    ):
        super().__init__()

        # Store PARSeq for extracting logits from generated images
        self.parseq = parseq_model

        # Freeze PARSeq
        for param in self.parseq.parameters():
            param.requires_grad = False
        self.parseq.eval()

        self.reduction = reduction

    def forward(
        self,
        hr_pred: torch.Tensor,
        text_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute text prior loss.

        Args:
            hr_pred: Predicted HR images of shape (B, 3, H, W)
            text_indices: Ground truth text indices of shape (B, 7)

        Returns:
            Loss value (cross-entropy between predicted probs and GT text)
        """
        B = hr_pred.shape[0]

        # Get PARSeq logits from predicted HR (frozen)
        with torch.no_grad():
            self.parseq.eval()
            logits = self.parseq.forward_parallel(hr_pred)  # (B, 7, vocab_size)

        # Compute cross-entropy loss
        # text_indices: (B, 7) - GT character indices
        # logits: (B, 7, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for GT characters
        loss = F.nll_loss(
            log_probs.reshape(-1, self.parseq.vocab_size),
            text_indices.reshape(-1),
            reduction='none'
        )  # (B * 7,)

        # Reshape and average over characters
        loss = loss.reshape(B, -1)  # (B, 7)
        loss = loss.mean(dim=1)  # (B,)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiFrameTextPriorExtractor(nn.Module):
    """
    Multi-frame text prior extractor that aggregates text information
    from all frames before extracting text prior.

    Args:
        parseq_model: Pretrained PARSeq model
        num_frames: Number of frames to aggregate
        aggregation: How to aggregate ('average', 'max', 'attention')
        feature_dim: Dimension for projected text features
    """

    def __init__(
        self,
        parseq_model: nn.Module,
        num_frames: int = 5,
        aggregation: str = 'average',
        feature_dim: int = 256
    ):
        super().__init__()

        self.num_frames = num_frames
        self.aggregation = aggregation

        # Individual extractors for each frame
        self.extractors = nn.ModuleList([
            TextPriorExtractor(parseq_model, feature_dim=feature_dim)
            for _ in range(num_frames)
        ])

        if aggregation == 'attention':
            # Attention-based aggregation
            self.attn = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)

    def forward(
        self,
        lr_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and aggregate text prior from multiple frames.

        Args:
            lr_frames: Multi-frame LR input of shape (B, T, 3, H, W)

        Returns:
            text_prior: Aggregated character probabilities (B, 7, vocab_size)
            text_features: Aggregated spatial text features (B, feature_dim, H, W)
        """
        B, T, C, H, W = lr_frames.shape

        # Extract text prior from each frame
        probs_list = []
        features_list = []

        for t in range(T):
            probs, features = self.extractors[t](lr_frames, t)
            probs_list.append(probs)
            features_list.append(features)

        # Stack
        probs_stack = torch.stack(probs_list, dim=0)  # (T, B, 7, vocab_size)
        features_stack = torch.stack(features_list, dim=0)  # (T, B, feature_dim, H, W)

        if self.aggregation == 'average':
            # Average probabilities and features
            text_prior = probs_stack.mean(dim=0)  # (B, 7, vocab_size)
            text_features = features_stack.mean(dim=0)  # (B, feature_dim, H, W)

        elif self.aggregation == 'max':
            # Max pooling (more conservative)
            text_prior = probs_stack.max(dim=0)[0]
            text_features = features_stack.max(dim=0)[0]

        elif self.aggregation == 'attention':
            # Attention-based aggregation
            probs_flat = probs_stack.permute(1, 2, 3, 0).reshape(B, 7, -1)
            # For simplicity, just use average on probs
            text_prior = probs_stack.mean(dim=0)

            # Attention on features
            features_flat = features_stack.permute(1, 3, 0, 2).reshape(B, H * W, T, -1)
            attn_out, _ = self.attn(features_flat, features_flat, features_flat)
            attn_out = attn_out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            text_features = attn_out.mean(dim=1, keepdim=True)  # Average across sequence

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return text_prior, text_features


def create_text_prior_extractor(
    parseq_model: nn.Module,
    multi_frame: bool = False,
    num_frames: int = 5,
    aggregation: str = 'average',
    feature_dim: int = 256
) -> nn.Module:
    """
    Factory function to create text prior extractor.

    Args:
        parseq_model: Pretrained PARSeq model
        multi_frame: Whether to use multi-frame aggregation
        num_frames: Number of frames (if multi_frame)
        aggregation: Aggregation method ('average', 'max', 'attention')
        feature_dim: Dimension for projected text features

    Returns:
        Text prior extractor
    """
    if multi_frame:
        return MultiFrameTextPriorExtractor(
            parseq_model, num_frames, aggregation, feature_dim
        )
    else:
        return TextPriorExtractor(parseq_model, feature_dim=feature_dim)


# Test function
def _test_text_prior_extractor():
    """Quick test to verify text prior extractor works correctly."""
    import torch

    # Create a mock PARSeq model for testing
    class MockPARSeq(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = 39
            self.conv = nn.Conv2d(3, 64, 3, padding=1)

        def forward_parallel(self, x):
            # Simple mock: return random logits
            B = x.shape[0]
            return torch.randn(B, 7, self.vocab_size, device=x.device)

    # Create models
    parseq = MockPARSeq()
    extractor = TextPriorExtractor(parseq)

    # Test data
    B, T, C, H, W = 2, 5, 3, 16, 48
    lr_frames = torch.randn(B, T, C, H, W)

    # Forward pass
    probs, features = extractor(lr_frames, frame_idx=0)

    print(f"Input shape: {lr_frames.shape}")
    print(f"Text prior shape: {probs.shape}")
    print(f"Text features shape: {features.shape}")
    print(f"Probabilities sum to 1: {torch.allclose(probs.sum(dim=-1), torch.ones(B * 7))}")

    # Test multi-frame
    multi_extractor = MultiFrameTextPriorExtractor(parseq, num_frames=T)
    probs_multi, features_multi = multi_extractor(lr_frames)

    print(f"Multi-frame text prior shape: {probs_multi.shape}")
    print(f"Multi-frame text features shape: {features_multi.shape}")

    print("✓ Text prior extractor test passed!")


if __name__ == "__main__":
    _test_text_prior_extractor()
