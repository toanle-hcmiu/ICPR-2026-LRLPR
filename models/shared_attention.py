"""
Shared Attention Module (PLTFAM-style) for License Plate Super-Resolution.

Implements a Pixel-Level Three-Fold Attention Module (PLTFAM) with shared weights
across all residual blocks, as described in "Enhancing License Plate Super-Resolution:
A Layout-Aware and Character-Driven Approach" (Nascimento et al.).

Key Features:
1. **Shared Weights**: Single attention module used across all blocks for consistent feature emphasis
2. **Three-Fold Attention**: Channel, Positional, and Geometrical Perception units
3. **Deformable Convolutions**: Used in positional and channel units for adaptive sampling
4. **Sub-pixel Convolution**: PixelShuffle for efficient upsampling

Reference:
- LCOFL Paper: Nascimento et al. (2024)
- Original PLTFAM: Nascimento et al. (2023) "Super-resolution of license plate images 
  using attention modules and sub-pixel convolution layers"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .deformable_conv import DeformableConv2dSimple, create_deformable_conv


class ChannelAttentionUnit(nn.Module):
    """
    Channel Attention Unit using PixelShuffle capabilities.
    
    Exploits inter-channel feature relationships to emphasize informative channels.
    Uses deformable convolutions for adaptive spatial sampling.
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        use_deformable: bool = True
    ):
        """
        Initialize channel attention unit.
        
        Args:
            in_channels: Number of input channels.
            reduction_ratio: Channel reduction ratio for bottleneck.
            use_deformable: Whether to use deformable convolutions.
        """
        super().__init__()
        
        reduced_channels = max(in_channels // reduction_ratio, 8)
        
        # Global context aggregation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Channel attention with optional deformable conv
        if use_deformable:
            # Use 1x1 conv equivalent (deformable doesn't help for 1x1)
            self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1, bias=False)
            self.fc2 = nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        else:
            self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1, bias=False)
            self.fc2 = nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.
        
        Args:
            x: Input tensor (B, C, H, W).
            
        Returns:
            Channel-attended tensor (B, C, H, W).
        """
        # Global context
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        
        # Combine and apply attention
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class PositionalAttentionUnit(nn.Module):
    """
    Positional Attention Unit for spatial localization.
    
    Integrates positional information to help the network focus on
    character-relevant regions. Uses deformable convolutions for
    adaptive spatial sampling.
    """
    
    def __init__(
        self,
        in_channels: int,
        use_deformable: bool = True
    ):
        """
        Initialize positional attention unit.
        
        Args:
            in_channels: Number of input channels.
            use_deformable: Whether to use deformable convolutions.
        """
        super().__init__()
        
        # Spatial attention generation
        if use_deformable:
            self.conv1 = create_deformable_conv(
                in_channels, in_channels, kernel_size=3, padding=1,
                use_simple=True
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional attention.
        
        Args:
            x: Input tensor (B, C, H, W).
            
        Returns:
            Position-attended tensor (B, C, H, W).
        """
        # Generate spatial attention map
        attention = self.relu(self.conv1(x))
        attention = self.sigmoid(self.conv2(attention))
        
        return x * attention


class GeometricalPerceptionUnit(nn.Module):
    """
    Geometrical Perception Unit for texture and structure enhancement.
    
    Improves texture and reconstruction quality by learning geometric
    transformations that preserve character details.
    """
    
    def __init__(
        self,
        in_channels: int,
        use_deformable: bool = True
    ):
        """
        Initialize geometrical perception unit.
        
        Args:
            in_channels: Number of input channels.
            use_deformable: Whether to use deformable convolutions.
        """
        super().__init__()
        
        # Multi-scale feature extraction
        if use_deformable:
            self.conv3x3 = create_deformable_conv(
                in_channels, in_channels // 2, kernel_size=3, padding=1,
                use_simple=True
            )
            self.conv5x5 = create_deformable_conv(
                in_channels, in_channels // 2, kernel_size=3, padding=2, dilation=2,
                use_simple=True
            )
        else:
            self.conv3x3 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
            self.conv5x5 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=2, dilation=2)
        
        # Feature fusion
        self.fusion = nn.Conv2d(in_channels, in_channels, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply geometrical perception.
        
        Args:
            x: Input tensor (B, C, H, W).
            
        Returns:
            Geometry-enhanced tensor (B, C, H, W).
        """
        # Multi-scale features
        feat_3x3 = self.relu(self.conv3x3(x))
        feat_5x5 = self.relu(self.conv5x5(x))
        
        # Concatenate and fuse
        fused = torch.cat([feat_3x3, feat_5x5], dim=1)
        out = self.fusion(fused)
        
        return out + x  # Residual connection


class SharedAttentionModule(nn.Module):
    """
    Shared Attention Module (PLTFAM-style) with shared weights.
    
    Combines three attention units (Channel, Positional, Geometrical)
    into a single module that can be shared across all residual blocks.
    This enables consistent feature emphasis from early to late layers.
    
    Key Benefits:
    1. Consistent attention across all network depths
    2. Parameter efficiency through weight sharing
    3. Better feature extraction for character recognition
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        use_deformable: bool = True
    ):
        """
        Initialize shared attention module.
        
        Args:
            in_channels: Number of input channels.
            reduction_ratio: Reduction ratio for channel attention.
            use_deformable: Whether to use deformable convolutions.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.use_deformable = use_deformable
        
        # Three-fold attention units
        self.channel_unit = ChannelAttentionUnit(
            in_channels, reduction_ratio, use_deformable
        )
        self.positional_unit = PositionalAttentionUnit(
            in_channels, use_deformable
        )
        self.geometrical_unit = GeometricalPerceptionUnit(
            in_channels, use_deformable
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply shared attention.
        
        Args:
            x: Input tensor (B, C, H, W).
            
        Returns:
            Attention-enhanced tensor (B, C, H, W).
        """
        # Apply three attention units in sequence
        out = self.channel_unit(x)
        out = self.positional_unit(out)
        out = self.geometrical_unit(out)
        
        # Final fusion with residual
        out = self.fusion(out)
        out = out + x
        
        return out


class PixelLevelAttentionBlock(nn.Module):
    """
    Complete Pixel-Level Attention Block combining convolutions with shared attention.
    
    This block can be used as a drop-in replacement for standard residual blocks,
    providing enhanced feature extraction through attention mechanisms.
    """
    
    def __init__(
        self,
        in_channels: int,
        shared_attention: Optional[SharedAttentionModule] = None,
        use_deformable: bool = True
    ):
        """
        Initialize pixel-level attention block.
        
        Args:
            in_channels: Number of input channels.
            shared_attention: Optional shared attention module. If None, creates own.
            use_deformable: Whether to use deformable convolutions.
        """
        super().__init__()
        
        # Main convolution path
        if use_deformable:
            self.conv1 = create_deformable_conv(
                in_channels, in_channels, kernel_size=3, padding=1, use_simple=True
            )
            self.conv2 = create_deformable_conv(
                in_channels, in_channels, kernel_size=3, padding=1, use_simple=True
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Use shared attention or create own
        if shared_attention is not None:
            self.attention = shared_attention
        else:
            self.attention = SharedAttentionModule(
                in_channels, use_deformable=use_deformable
            )
        
        self._shared_attention = shared_attention is not None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W).
            
        Returns:
            Output tensor (B, C, H, W).
        """
        # Convolution path
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        
        # Apply attention
        out = self.attention(out)
        
        # Residual connection
        out = out + x
        
        return out


class SubPixelConvolutionBlock(nn.Module):
    """
    Sub-Pixel Convolution Block for upsampling.
    
    Uses PixelShuffle for efficient learnable upsampling,
    as used in the original PLTFAM paper.
    """
    
    def __init__(
        self,
        in_channels: int,
        upscale_factor: int = 2
    ):
        """
        Initialize sub-pixel convolution block.
        
        Args:
            in_channels: Number of input channels.
            upscale_factor: Upscaling factor.
        """
        super().__init__()
        
        out_channels = in_channels * (upscale_factor ** 2)
        
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W).
            
        Returns:
            Upsampled tensor (B, C, H*scale, W*scale).
        """
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.relu(out)
        return out


def create_shared_attention_network(
    in_channels: int,
    num_blocks: int = 4,
    use_deformable: bool = True
) -> Tuple[nn.ModuleList, SharedAttentionModule]:
    """
    Create a network with shared attention across multiple blocks.
    
    Args:
        in_channels: Number of input channels.
        num_blocks: Number of attention blocks.
        use_deformable: Whether to use deformable convolutions.
        
    Returns:
        Tuple of (blocks, shared_attention_module).
    """
    # Create shared attention module
    shared_attention = SharedAttentionModule(
        in_channels, use_deformable=use_deformable
    )
    
    # Create blocks that share the attention module
    blocks = nn.ModuleList([
        PixelLevelAttentionBlock(
            in_channels, shared_attention=shared_attention,
            use_deformable=use_deformable
        )
        for _ in range(num_blocks)
    ])
    
    return blocks, shared_attention
