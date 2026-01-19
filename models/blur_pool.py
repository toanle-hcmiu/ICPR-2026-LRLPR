"""
Anti-Aliased Downsampling (BlurPool).

Implements low-pass filtering before decimation to prevent aliasing artifacts
that manifest as "wavy" distortions and texture sticking in generated images.

Reference: Richard Zhang, "Making Convolutional Networks Shift-Invariant Again" (ICML 2019)
https://github.com/adobe/antialiased-cnns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BlurPool2d(nn.Module):
    """
    Anti-aliased downsampling layer.
    
    Applies a low-pass Gaussian blur before strided pooling to prevent aliasing.
    This addresses the Nyquist-Shannon sampling theorem violation that occurs
    when high-frequency content is present during downsampling.
    
    The blur kernel removes frequencies above the Nyquist limit of the
    downsampled grid, preventing them from "folding over" into lower frequencies
    and creating wavy artifacts.
    
    Args:
        channels: Number of input channels.
        stride: Downsampling stride (typically 2).
        filter_size: Size of the blur kernel (3 or 5 recommended).
        pad_type: Padding mode ('reflect' or 'replicate').
    """
    
    def __init__(
        self, 
        channels: int, 
        stride: int = 2, 
        filter_size: int = 3,
        pad_type: str = 'reflect'
    ):
        super().__init__()
        
        self.channels = channels
        self.stride = stride
        self.pad_type = pad_type
        
        # Create low-pass filter kernel
        # Using binomial filter coefficients for smooth anti-aliasing
        if filter_size == 3:
            # [1, 2, 1] / 4 - good balance between blur and sharpness
            blur_1d = torch.tensor([1., 2., 1.]) / 4.0
        elif filter_size == 5:
            # [1, 4, 6, 4, 1] / 16 - stronger anti-aliasing
            blur_1d = torch.tensor([1., 4., 6., 4., 1.]) / 16.0
        elif filter_size == 7:
            # [1, 6, 15, 20, 15, 6, 1] / 64 - very strong anti-aliasing
            blur_1d = torch.tensor([1., 6., 15., 20., 15., 6., 1.]) / 64.0
        else:
            raise ValueError(f"Unsupported filter_size: {filter_size}. Use 3, 5, or 7.")
        
        # Create 2D kernel from outer product
        blur_kernel = blur_1d[:, None] * blur_1d[None, :]
        
        # Expand to (channels, 1, H, W) for depthwise convolution
        blur_kernel = blur_kernel.view(1, 1, filter_size, filter_size)
        blur_kernel = blur_kernel.repeat(channels, 1, 1, 1)
        
        self.register_buffer('blur_kernel', blur_kernel)
        self.pad = (filter_size - 1) // 2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply anti-aliased downsampling.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Downsampled tensor of shape (B, C, H//stride, W//stride).
        """
        # Pad input to handle boundary conditions
        if self.pad > 0:
            if self.pad_type == 'reflect':
                x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
            else:
                x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='replicate')
        
        # Apply blur using depthwise convolution, then downsample
        x = F.conv2d(x, self.blur_kernel, stride=self.stride, groups=self.channels)
        return x


class MaxBlurPool2d(nn.Module):
    """
    Anti-aliased max pooling.
    
    Performs max pooling followed by blur pooling to prevent aliasing.
    This is preferable to standard MaxPool2d for encoder networks.
    
    Reference: Zhang (2019) - MaxPool creates binary decisions that generate
    high-frequency artifacts when pooled without anti-aliasing.
    
    Args:
        channels: Number of input channels.
        kernel_size: Max pooling kernel size.
        stride: Downsampling stride.
        filter_size: Size of the blur kernel.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        filter_size: int = 3
    ):
        super().__init__()
        
        # Max pooling with stride=1 (no downsampling yet)
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
        # Blur and downsample
        self.blur_pool = BlurPool2d(channels, stride=stride, filter_size=filter_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply anti-aliased max pooling.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Pooled tensor of shape (B, C, H//stride, W//stride).
        """
        x = self.max_pool(x)
        x = self.blur_pool(x)
        return x


class AntiAliasedConv2d(nn.Module):
    """
    Strided convolution with anti-aliasing.
    
    Performs convolution with stride=1, then applies blur pooling for
    downsampling. This prevents aliasing that occurs in standard
    strided convolutions.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Effective stride (applied via blur pooling).
        padding: Convolution padding.
        bias: Whether to include bias.
        filter_size: Size of the anti-aliasing blur kernel.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        bias: bool = True,
        filter_size: int = 3
    ):
        super().__init__()
        
        self.stride = stride
        
        # Convolution with stride=1
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding, 
            bias=bias
        )
        
        # Anti-aliased downsampling (only if stride > 1)
        if stride > 1:
            self.blur_pool = BlurPool2d(out_channels, stride=stride, filter_size=filter_size)
        else:
            self.blur_pool = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply anti-aliased strided convolution.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Output tensor of shape (B, out_channels, H//stride, W//stride).
        """
        x = self.conv(x)
        if self.blur_pool is not None:
            x = self.blur_pool(x)
        return x
