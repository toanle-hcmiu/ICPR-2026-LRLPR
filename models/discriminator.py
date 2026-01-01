"""
PatchGAN Discriminator for Adversarial Training.

This module implements a PatchGAN discriminator that distinguishes between
real and generated high-resolution license plate images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SpectralNorm(nn.Module):
    """Wrapper for spectral normalization."""
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = nn.utils.spectral_norm(module)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class DiscriminatorBlock(nn.Module):
    """
    Basic discriminator block with optional spectral normalization.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_spectral_norm: bool = True,
        use_bn: bool = True,
        activation: str = 'leaky_relu'
    ):
        super().__init__()
        
        # Convolution
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        
        layers = [conv]
        
        # Batch normalization
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation
        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for adversarial training.
    
    Outputs a grid of predictions, where each cell represents
    whether a corresponding patch of the input is real or fake.
    This provides more detailed feedback than a single scalar.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 3,
        use_spectral_norm: bool = True
    ):
        """
        Initialize the PatchGAN discriminator.
        
        Args:
            in_channels: Number of input channels.
            base_channels: Number of channels in first layer.
            num_layers: Number of discriminator layers.
            use_spectral_norm: Whether to use spectral normalization.
        """
        super().__init__()
        
        self.use_spectral_norm = use_spectral_norm
        
        # First layer (no batch norm)
        layers = [
            DiscriminatorBlock(
                in_channels, base_channels,
                kernel_size=4, stride=2, padding=1,
                use_spectral_norm=use_spectral_norm,
                use_bn=False
            )
        ]
        
        # Intermediate layers
        channels = base_channels
        for i in range(1, num_layers):
            out_channels = min(channels * 2, 512)
            layers.append(
                DiscriminatorBlock(
                    channels, out_channels,
                    kernel_size=4, stride=2, padding=1,
                    use_spectral_norm=use_spectral_norm,
                    use_bn=True
                )
            )
            channels = out_channels
        
        # Penultimate layer (stride 1)
        out_channels = min(channels * 2, 512)
        layers.append(
            DiscriminatorBlock(
                channels, out_channels,
                kernel_size=4, stride=1, padding=1,
                use_spectral_norm=use_spectral_norm,
                use_bn=True
            )
        )
        channels = out_channels
        
        # Final layer (output single channel)
        final_conv = nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1)
        if use_spectral_norm:
            final_conv = nn.utils.spectral_norm(final_conv)
        layers.append(final_conv)
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute patch-wise discrimination scores.
        
        Args:
            x: Input image of shape (B, C, H, W).
            
        Returns:
            Discrimination scores of shape (B, 1, H', W').
        """
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for more robust adversarial training.
    
    Uses multiple discriminators at different scales to provide
    richer feedback to the generator.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 3,
        num_scales: int = 3,
        use_spectral_norm: bool = True
    ):
        """
        Initialize multi-scale discriminator.
        
        Args:
            in_channels: Number of input channels.
            base_channels: Base channels for each discriminator.
            num_layers: Number of layers per discriminator.
            num_scales: Number of scales.
            use_spectral_norm: Whether to use spectral normalization.
        """
        super().__init__()
        
        self.num_scales = num_scales
        
        # Create discriminators for each scale
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(
                in_channels, base_channels, num_layers, use_spectral_norm
            )
            for _ in range(num_scales)
        ])
        
        # Downsampler for multi-scale input
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute discrimination scores at multiple scales.
        
        Args:
            x: Input image of shape (B, C, H, W).
            
        Returns:
            List of discrimination scores at different scales.
        """
        outputs = []
        
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x))
            if i < self.num_scales - 1:
                x = self.downsample(x)
        
        return outputs


class UNetDiscriminator(nn.Module):
    """
    U-Net based discriminator for pixel-level discrimination.
    
    Provides both global and local feedback by using skip connections.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        skip_connections: bool = True
    ):
        super().__init__()
        
        self.skip_connections = skip_connections
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 2, 4, 2, 1),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder
        dec_in = num_features * 8
        if skip_connections:
            self.dec4 = nn.Sequential(
                nn.ConvTranspose2d(num_features * 8, num_features * 4, 4, 2, 1),
                nn.BatchNorm2d(num_features * 4),
                nn.ReLU(inplace=True)
            )
            self.dec3 = nn.Sequential(
                nn.ConvTranspose2d(num_features * 8, num_features * 2, 4, 2, 1),
                nn.BatchNorm2d(num_features * 2),
                nn.ReLU(inplace=True)
            )
            self.dec2 = nn.Sequential(
                nn.ConvTranspose2d(num_features * 4, num_features, 4, 2, 1),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True)
            )
            self.dec1 = nn.Sequential(
                nn.ConvTranspose2d(num_features * 2, 1, 4, 2, 1)
            )
        else:
            self.dec4 = nn.Sequential(
                nn.ConvTranspose2d(num_features * 8, num_features * 4, 4, 2, 1),
                nn.BatchNorm2d(num_features * 4),
                nn.ReLU(inplace=True)
            )
            self.dec3 = nn.Sequential(
                nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1),
                nn.BatchNorm2d(num_features * 2),
                nn.ReLU(inplace=True)
            )
            self.dec2 = nn.Sequential(
                nn.ConvTranspose2d(num_features * 2, num_features, 4, 2, 1),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True)
            )
            self.dec1 = nn.Sequential(
                nn.ConvTranspose2d(num_features, 1, 4, 2, 1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pixel-level discrimination scores.
        
        Args:
            x: Input image of shape (B, C, H, W).
            
        Returns:
            Discrimination map of shape (B, 1, H, W).
        """
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Decode with skip connections
        if self.skip_connections:
            d4 = self.dec4(e4)
            d3 = self.dec3(torch.cat([d4, e3], dim=1))
            d2 = self.dec2(torch.cat([d3, e2], dim=1))
            d1 = self.dec1(torch.cat([d2, e1], dim=1))
        else:
            d4 = self.dec4(e4)
            d3 = self.dec3(d4)
            d2 = self.dec2(d3)
            d1 = self.dec1(d2)
        
        return d1
