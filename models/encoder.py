"""
Shared CNN Encoder for Multi-Frame Feature Extraction.

This module implements the shared CNN encoder that processes multiple
low-resolution frames to extract features for the STN and downstream modules.

Now with anti-aliased downsampling to prevent aliasing artifacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .blur_pool import BlurPool2d


class ConvBlock(nn.Module):
    """
    Convolutional block with BatchNorm and ReLU.
    
    Optionally uses anti-aliased downsampling (BlurPool) to prevent
    aliasing artifacts when stride > 1.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        activation: str = 'relu',
        use_blur_pool: bool = False  # Anti-aliased downsampling
    ):
        super().__init__()
        
        self.use_blur_pool = use_blur_pool and stride > 1
        
        # If using blur pool, conv has stride=1, then blur pool downsamples
        actual_stride = 1 if self.use_blur_pool else stride
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, actual_stride, padding, bias=not use_bn)
        ]
        
        # Anti-aliased downsampling: blur then subsample
        if self.use_blur_pool:
            layers.append(BlurPool2d(out_channels, stride=stride))
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'none':
            pass
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SharedCNNEncoder(nn.Module):
    """
    Shared CNN Encoder for multi-frame feature extraction.
    
    This encoder processes multiple LR frames using shared weights,
    producing feature maps for each frame that can be used by the STN
    and other downstream modules.
    
    Architecture:
        - 4 convolutional blocks with increasing channels
        - Each block: Conv -> BatchNorm -> ReLU
        - Downsampling via anti-aliased stride-2 convolutions (BlurPool)
    
    Anti-aliasing prevents high-frequency content from causing aliasing
    artifacts (wavy distortions) during downsampling.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 4,
        channel_multipliers: Optional[List[int]] = None,
        use_blur_pool: bool = True  # Enable anti-aliased downsampling
    ):
        """
        Initialize the shared CNN encoder.
        
        Args:
            in_channels: Number of input channels (3 for RGB).
            base_channels: Number of channels in the first block.
            num_blocks: Number of convolutional blocks.
            channel_multipliers: Channel multipliers for each block.
            use_blur_pool: Whether to use anti-aliased downsampling (recommended: True).
        """
        super().__init__()
        
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 8]
        
        assert len(channel_multipliers) == num_blocks, \
            f"channel_multipliers must have {num_blocks} elements"
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.use_blur_pool = use_blur_pool
        
        # Build encoder blocks
        blocks = []
        prev_channels = in_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_channels = base_channels * mult
            stride = 2 if i > 0 else 1  # First block keeps resolution
            
            # Use anti-aliased downsampling for stride-2 convolutions
            blocks.append(ConvBlock(
                prev_channels, out_channels, 
                kernel_size=3, stride=stride, padding=1,
                use_blur_pool=use_blur_pool  # Anti-aliased downsampling
            ))
            
            # Add a second conv in each block for more capacity
            blocks.append(ConvBlock(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=1
            ))
            
            prev_channels = out_channels
        
        self.encoder = nn.Sequential(*blocks)
        self.out_channels = prev_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for single frame.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Feature map of shape (B, out_channels, H', W').
        """
        return self.encoder(x)
    
    def forward_multi_frame(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multiple frames with shared weights.
        
        Args:
            x: Input tensor of shape (B, num_frames, C, H, W).
            
        Returns:
            Feature maps of shape (B, num_frames, out_channels, H', W').
        """
        B, T, C, H, W = x.shape
        
        # Reshape to process all frames together
        x = x.view(B * T, C, H, W)
        
        # Apply shared encoder
        features = self.encoder(x)
        
        # Reshape back to include frame dimension
        _, F_out, H_out, W_out = features.shape
        features = features.view(B, T, F_out, H_out, W_out)
        
        return features
    
    def get_output_size(self, input_height: int, input_width: int) -> tuple:
        """
        Calculate output feature map size given input size.
        
        Args:
            input_height: Input image height.
            input_width: Input image width.
            
        Returns:
            Tuple of (out_channels, out_height, out_width).
        """
        # 3 stride-2 convolutions reduce spatial size by 8
        out_height = input_height // 8
        out_width = input_width // 8
        return self.out_channels, out_height, out_width


class ResidualBlock(nn.Module):
    """Residual block for deeper encoders."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResidualCNNEncoder(nn.Module):
    """
    Deeper CNN encoder with residual connections.
    
    Alternative to SharedCNNEncoder with more capacity for complex tasks.
    Now with anti-aliased downsampling to prevent aliasing artifacts.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_res_blocks: int = 4,
        use_blur_pool: bool = True  # Enable anti-aliased downsampling
    ):
        super().__init__()
        
        self.use_blur_pool = use_blur_pool
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, 1, 3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling stages with optional anti-aliasing
        # Anti-aliased: Conv(stride=1) -> BlurPool(stride=2)
        # Standard: Conv(stride=2)
        if use_blur_pool:
            self.down1 = nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 3, 1, 1, bias=False),
                BlurPool2d(base_channels * 2, stride=2),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True)
            )
            
            self.down2 = nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, 1, 1, bias=False),
                BlurPool2d(base_channels * 4, stride=2),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True)
            )
            
            self.down3 = nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 8, 3, 1, 1, bias=False),
                BlurPool2d(base_channels * 8, stride=2),
                nn.BatchNorm2d(base_channels * 8),
                nn.ReLU(inplace=True)
            )
        else:
            self.down1 = nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True)
            )
            
            self.down2 = nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True)
            )
            
            self.down3 = nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 8, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 8),
                nn.ReLU(inplace=True)
            )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels * 8) for _ in range(num_res_blocks)]
        )
        
        self.out_channels = base_channels * 8
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.res_blocks(x)
        return x
    
    def forward_multi_frame(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.forward(x)
        _, F_out, H_out, W_out = features.shape
        features = features.view(B, T, F_out, H_out, W_out)
        return features
