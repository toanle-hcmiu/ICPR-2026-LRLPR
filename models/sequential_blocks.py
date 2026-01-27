"""
Sequential Residual Blocks for License Plate Super-Resolution.

This module implements Sequential Residual Blocks (SRB) from TSRN, which use
Bi-directional LSTM for sequential modeling of characters. This is better suited
for text than SwinIR's window-based attention.

Reference: TSRN (ECCV 2020) - "Scene Text Image Super-resolution in the Wild"

Author: New Architecture Implementation (2026-01-27)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SequentialResidualBlock(nn.Module):
    """
    Sequential Residual Block (SRB) from TSRN.

    Uses Bi-directional LSTM for sequential modeling of characters.
    Better for text than SwinIR's window-based attention because:
    - Captures horizontal sequential dependencies (character order)
    - Captures vertical sequential dependencies (stroke structure)
    - No window size constraints

    Args:
        channels: Number of input/output channels
        lstm_hidden: Hidden size for LSTM (default: 32, output is 64 due to bidirectional)
        use_dropout: Whether to use dropout
    """

    def __init__(
        self,
        channels: int = 64,
        lstm_hidden: int = 32,
        use_dropout: bool = True
    ):
        super().__init__()

        self.channels = channels
        self.lstm_hidden = lstm_hidden
        self.use_dropout = use_dropout

        # CNN feature extraction
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Dropout for regularization
        if use_dropout:
            self.dropout = nn.Dropout2d(0.1)

        # Bi-directional LSTM for horizontal sequential modeling
        # Processes each row as a sequence of columns
        self.blstm_h = nn.LSTM(
            input_size=channels,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )

        # Bi-directional LSTM for vertical sequential modeling
        # Processes each column as a sequence of rows
        self.blstm_v = nn.LSTM(
            input_size=channels,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )

        # Projection to match channels (bidirectional LSTM outputs 2*hidden)
        self.proj_h = nn.Conv2d(2 * lstm_hidden, channels, 1)
        self.proj_v = nn.Conv2d(2 * lstm_hidden, channels, 1)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        text_prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply sequential residual block.

        Args:
            x: Input tensor of shape (B, C, H, W)
            text_prior: Optional text prior of shape (B, 7, vocab_size)
                         (not used in basic SRB, but kept for compatibility)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        identity = x
        B, C, H, W = x.shape

        # CNN feature extraction
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        if self.use_dropout:
            out = self.dropout(out)

        # Horizontal sequential modeling (BiLSTM across width)
        # Reshape: (B, C, H, W) -> (B*H, W, C) - each row is a sequence
        out_h = out.permute(0, 2, 3, 1).reshape(B * H, W, C)
        out_h, _ = self.blstm_h(out_h)  # (B*H, W, 2*hidden)
        # Reshape back: (B*H, W, 2*hidden) -> (B, 2*hidden, H, W) -> (B, C, H, W)
        out_h = out_h.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        out_h = self.proj_h(out_h)

        # Vertical sequential modeling (BiLSTM across height)
        # Reshape: (B, C, H, W) -> (B*W, H, C) - each column is a sequence
        out_v = out.permute(0, 3, 2, 1).reshape(B * W, H, C)
        out_v, _ = self.blstm_v(out_v)  # (B*W, H, 2*hidden)
        # Reshape back: (B*W, H, 2*hidden) -> (B, 2*hidden, H, W)
        out_v = out_v.reshape(B, W, H, -1).permute(0, 3, 2, 1)
        out_v = self.proj_v(out_v)

        # Fuse original, horizontal, and vertical features
        out = torch.cat([out, out_h, out_v], dim=1)
        out = self.fusion(out)

        # Residual connection
        out = out + identity

        return out


class TextGuidedSRB(nn.Module):
    """
    Text-Guided Sequential Residual Block.

    Enhanced version of SRB that incorporates text prior information.
    Uses text priors to guide the sequential modeling.

    Args:
        channels: Number of input/output channels
        text_dim: Dimension of text prior features
        lstm_hidden: Hidden size for LSTM
    """

    def __init__(
        self,
        channels: int = 64,
        text_dim: int = 256,
        lstm_hidden: int = 32
    ):
        super().__init__()

        self.channels = channels
        self.text_dim = text_dim

        # Base SRB
        self.srb = SequentialResidualBlock(channels, lstm_hidden, use_dropout=False)

        # Text-image fusion
        self.text_fusion = nn.Sequential(
            nn.Conv2d(text_dim, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

        # Gating mechanism for text guidance
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        text_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply text-guided sequential residual block.

        Args:
            x: Input tensor of shape (B, C, H, W)
            text_features: Text features of shape (B, text_dim, 1, 7) or None

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        identity = x

        # Apply base SRB
        srb_out = self.srb(x)

        if text_features is not None:
            # Upsample text features to match spatial size
            # (B, text_dim, 1, 7) -> (B, text_dim, H, W)
            B, text_dim, _, num_chars = text_features.shape
            text_feat_up = F.interpolate(
                text_features,
                size=(x.shape[2], x.shape[3]),
                mode='bilinear',
                align_corners=False
            )

            # Fuse text features
            text_feat = self.text_fusion(text_feat_up)

            # Gated combination
            gate = self.gate(torch.cat([srb_out, text_feat], dim=1))
            out = gate * srb_out + (1 - gate) * text_feat
        else:
            out = srb_out

        # Residual
        out = out + identity

        return out


class SequentialResidualBlocks(nn.Module):
    """
    Stack of Sequential Residual Blocks.

    Args:
        channels: Number of input/output channels
        num_blocks: Number of SRB blocks
        lstm_hidden: Hidden size for LSTM
        use_text_guidance: Whether to use text guidance
        text_dim: Dimension of text prior features (if use_text_guidance)
    """

    def __init__(
        self,
        channels: int = 64,
        num_blocks: int = 4,
        lstm_hidden: int = 32,
        use_text_guidance: bool = False,
        text_dim: int = 256
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.use_text_guidance = use_text_guidance

        if use_text_guidance:
            blocks = [
                TextGuidedSRB(channels, text_dim, lstm_hidden)
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                SequentialResidualBlock(channels, lstm_hidden)
                for _ in range(num_blocks)
            ]

        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        x: torch.Tensor,
        text_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply stack of SRB blocks.

        Args:
            x: Input tensor of shape (B, C, H, W)
            text_features: Optional text features

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        out = x
        for block in self.blocks:
            if isinstance(block, TextGuidedSRB):
                out = block(out, text_features)
            else:
                out = block(out)

        return out


def create_srb(
    channels: int = 64,
    num_blocks: int = 4,
    lstm_hidden: int = 32,
    use_text_guidance: bool = False,
    text_dim: int = 256
) -> nn.Module:
    """
    Factory function to create SRB blocks.

    Args:
        channels: Number of input/output channels
        num_blocks: Number of SRB blocks
        lstm_hidden: Hidden size for LSTM
        use_text_guidance: Whether to use text guidance
        text_dim: Dimension of text prior features

    Returns:
        SRB module
    """
    if use_text_guidance:
        return SequentialResidualBlocks(
            channels, num_blocks, lstm_hidden, use_text_guidance, text_dim
        )
    else:
        return SequentialResidualBlocks(
            channels, num_blocks, lstm_hidden, use_text_guidance=False
        )


# Test function
def _test_sequential_blocks():
    """Quick test to verify SRB works correctly."""
    import torch

    batch_size = 2
    channels = 64
    h, w = 16, 48

    # Create test input
    x = torch.randn(batch_size, channels, h, w)

    # Test basic SRB
    srb = SequentialResidualBlock(channels)
    output = srb(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output requires grad: {output.requires_grad}")

    # Test SRB with text guidance
    text_feat = torch.randn(batch_size, 256, 1, 7)
    srb_tg = TextGuidedSRB(channels, 256)
    output_tg = srb_tg(x, text_feat)
    print(f"Text-guided output shape: {output_tg.shape}")

    # Test stack of SRBs
    srbs = SequentialResidualBlocks(channels, num_blocks=4)
    output_multi = srbs(x)
    print(f"Multi-block output shape: {output_multi.shape}")

    print("✓ Sequential blocks test passed!")


if __name__ == "__main__":
    _test_sequential_blocks()
