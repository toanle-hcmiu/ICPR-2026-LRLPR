"""
Character Refinement Network for License Plate Super-Resolution.

This module provides a lightweight refinement network that takes Stage 2 generator output
and refines characters to improve OCR accuracy without causing mode collapse.

The key insight: Training the main generator with OCR loss causes mode collapse.
Solution: Keep generator fixed (Stage 2), train a separate lightweight refiner with OCR loss.

Author: Fix for Stage 3 mode collapse (2026-01-27)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RefinerResBlock(nn.Module):
    """
    Lightweight residual block for character refinement.

    Uses simpler architecture than the main encoder's ResidualBlock:
    - No BatchNorm (uses instance norm for stability)
    - Fewer parameters
    - Focus on local character refinement
    """

    def __init__(self, channels: int = 3, use_dropout: bool = False):
        super().__init__()

        self.use_dropout = use_dropout

        # Two conv layers for local refinement
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)

        # Dropout for regularization
        if use_dropout:
            self.dropout = nn.Dropout2d(0.1)

        # Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Refined output with same shape as input
        """
        identity = x

        # First conv + norm + relu
        out = self.relu(self.norm1(self.conv1(x)))

        # Optional dropout
        if self.use_dropout:
            out = self.dropout(out)

        # Second conv + norm
        out = self.norm2(self.conv2(out))

        # Residual connection
        out = out + identity

        # Final activation
        out = self.relu(out)

        return out


class CharacterRefiner(nn.Module):
    """
    Lightweight character refinement network.

    Takes Stage 2 generator output (which has varied but somewhat blurry characters)
    and refines it to produce sharper, more OCR-accurate characters.

    Key design principles:
    1. Lightweight: Only 3-5 residual blocks (not full U-Net)
    2. Residual: Output = Input + Refinement (preserves Stage 2 structure)
    3. Focused: Only refines character regions (not full image reconstruction)

    Architecture:
        Stage2 HR (64x192) -> [RefBlock x3] -> Residual -> Refined HR

    Training:
        - Stage 2 generator is FROZEN (no gradients)
        - Refiner trained ONLY with OCR loss (cross-entropy)
        - No pixel loss to refiner (prevents blurring)
    """

    def __init__(
        self,
        num_blocks: int = 3,
        channels: int = 3,
        use_dropout: bool = True,
        use_attention: bool = False
    ):
        """
        Initialize character refiner.

        Args:
            num_blocks: Number of residual blocks (default: 3)
            channels: Number of input/output channels (default: 3 for RGB)
            use_dropout: Whether to use dropout for regularization
            use_attention: Whether to add lightweight attention (experimental)
        """
        super().__init__()

        self.num_blocks = num_blocks
        self.channels = channels

        # Build residual blocks
        blocks = []
        for i in range(num_blocks):
            blocks.append(RefinerResBlock(channels, use_dropout=use_dropout))
        self.blocks = nn.ModuleList(blocks)

        # Optional: Lightweight attention for character focus
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(channels, 1, kernel_size=1),
                nn.Sigmoid()
            )

        # Final projection (keeps output in valid range)
        self.final_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Initialize final conv to near-zero (small residual updates)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(
        self,
        stage2_output: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Refine Stage 2 output to improve character clarity.

        Args:
            stage2_output: Stage 2 generator output (B, 3, 64, 192)
            return_attention: Whether to return attention map for visualization

        Returns:
            Refined output (B, 3, 64, 192)
            Optional: Attention map (B, 1, 64, 192) if return_attention=True
        """
        x = stage2_output

        # Apply residual blocks
        for block in self.blocks:
            x = block(x)

        # Optional attention for character focus
        attention_map = None
        if hasattr(self, 'attention') and return_attention:
            attention_map = self.attention(x)
            x = x * attention_map
        elif hasattr(self, 'attention'):
            x = x * self.attention(x)

        # Final projection
        refinement = self.final_conv(x)

        # Residual connection: output = input + small_refinement
        # This preserves Stage 2 structure while adding character details
        output = stage2_output + refinement

        # Clamp to valid range (should be [-1, 1] after tanh in generator)
        output = torch.clamp(output, -1.0, 1.0)

        if return_attention:
            return output, attention_map
        return output

    def get_trainable_params(self) -> list:
        """
        Get trainable parameters for optimizer.

        Returns:
            List of parameter groups for optimizer
        """
        return [
            {'params': self.parameters(), 'name': 'refiner'}
        ]


class RefinerWithCheckpoint(nn.Module):
    """
    Wrapper for CharacterRefiner that supports gradient checkpointing
    to save memory during training.
    """

    def __init__(self, refiner: CharacterRefiner):
        super().__init__()
        self.refiner = refiner

    def forward(self, stage2_output: torch.Tensor) -> torch.Tensor:
        # Use gradient checkpointing if enabled
        if self.training:
            # Checkpoint each residual block separately
            x = stage2_output
            for i, block in enumerate(self.refiner.blocks):
                # Only checkpoint during training
                x = torch.utils.checkpoint.checkpoint(block, x)
            stage2_output_refined = x
        else:
            stage2_output_refined = self.refiner(stage2_output)
        return stage2_output_refined


def create_refiner(
    num_blocks: int = 3,
    channels: int = 3,
    use_dropout: bool = True,
    use_attention: bool = False,
    use_checkpointing: bool = False
) -> nn.Module:
    """
    Factory function to create a character refiner.

    Args:
        num_blocks: Number of residual blocks
        channels: Number of input/output channels
        use_dropout: Whether to use dropout
        use_attention: Whether to use attention mechanism
        use_checkpointing: Whether to use gradient checkpointing

    Returns:
        CharacterRefiner or RefinerWithCheckpoint instance
    """
    refiner = CharacterRefiner(
        num_blocks=num_blocks,
        channels=channels,
        use_dropout=use_dropout,
        use_attention=use_attention
    )

    if use_checkpointing:
        refiner = RefinerWithCheckpoint(refiner)

    return refiner


# Test function
def _test_refiner():
    """Quick test to verify refiner works correctly."""
    import torch

    # Create refiner
    refiner = CharacterRefiner(num_blocks=3, channels=3)

    # Test input (simulating Stage 2 output)
    batch_size = 2
    h, w = 64, 192
    x = torch.randn(batch_size, 3, h, w)

    # Forward pass
    with torch.no_grad():
        output = refiner(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Mean difference: {(output - x).abs().mean():.6f}")

    # Check residual connection works
    assert output.shape == x.shape, "Output shape should match input shape"
    assert output.min() >= -1.0 and output.max() <= 1.0, "Output should be in [-1, 1]"

    print("✓ Refiner test passed!")


if __name__ == "__main__":
    _test_refiner()
