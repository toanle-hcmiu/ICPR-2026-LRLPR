"""
Deformable Convolution v2 for License Plate Recognition.

Implements Deformable Convolution as proposed in:
"Deformable ConvNets V2: More Deformable, Better Results" (Zhu et al., CVPR 2019)

Deformable convolutions dynamically adjust their receptive fields based on input
features, allowing for more adaptable and precise modeling of spatial dependencies.
This is particularly useful for license plate super-resolution where characters
may have varying fonts, sizes, and perspective distortions.

Reference: 
- Paper: https://arxiv.org/abs/1811.11168
- Used in LCOFL paper (Nascimento et al.) for attention module enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class DeformableConv2d(nn.Module):
    """
    Deformable Convolution v2 with learnable offsets and modulation.
    
    Unlike standard convolutions with fixed geometric transformations,
    deformable convolutions learn spatial offsets for each sampling location,
    enabling adaptive receptive fields.
    
    Key components:
    1. Offset convolution: Predicts 2D offsets for each sampling location
    2. Modulation convolution: Predicts importance weights for each sample
    3. Deformable sampling: Applies bilinear interpolation at offset locations
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        use_modulation: bool = True
    ):
        """
        Initialize deformable convolution.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel.
            stride: Stride of the convolution.
            padding: Padding added to input.
            dilation: Dilation rate.
            groups: Number of groups for grouped convolution.
            bias: Whether to use bias.
            use_modulation: Whether to use modulation (DCNv2).
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.use_modulation = use_modulation
        
        # Number of sampling points
        self.num_points = self.kernel_size[0] * self.kernel_size[1]
        
        # Offset prediction: 2 (x, y) per sampling point
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * self.num_points,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True
        )
        
        # Modulation prediction: 1 weight per sampling point (DCNv2)
        if use_modulation:
            self.modulation_conv = nn.Conv2d(
                in_channels,
                self.num_points,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                bias=True
            )
        
        # Main convolution weight
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize offset conv to output zeros (start with regular convolution)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        
        if self.use_modulation:
            # Initialize modulation to output ones (equal weights)
            nn.init.zeros_(self.modulation_conv.weight)
            nn.init.zeros_(self.modulation_conv.bias)
        
        # Initialize main conv weight
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def _get_grid(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create regular grid of sampling locations.
        
        Args:
            batch_size: Batch size.
            height: Output height.
            width: Output width.
            device: Device to create tensor on.
            
        Returns:
            Grid tensor of shape (B, H*W, num_points, 2).
        """
        # Create base grid
        y_range = torch.arange(height, device=device, dtype=torch.float32)
        x_range = torch.arange(width, device=device, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
        
        # Base coordinates (center of each output location)
        base_coords = torch.stack([x_grid, y_grid], dim=-1)  # (H, W, 2)
        base_coords = base_coords.view(1, height * width, 1, 2)
        
        # Kernel offsets (relative positions of sampling points)
        kh, kw = self.kernel_size
        dh, dw = self.dilation
        
        kernel_offsets = []
        for dy in range(kh):
            for dx in range(kw):
                offset_y = (dy - kh // 2) * dh
                offset_x = (dx - kw // 2) * dw
                kernel_offsets.append([offset_x, offset_y])
        
        kernel_offsets = torch.tensor(kernel_offsets, device=device, dtype=torch.float32)
        kernel_offsets = kernel_offsets.view(1, 1, self.num_points, 2)
        
        # Combine base coordinates with kernel offsets
        grid = base_coords + kernel_offsets  # (1, H*W, num_points, 2)
        grid = grid.expand(batch_size, -1, -1, -1)
        
        return grid
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with deformable convolution.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Output tensor of shape (B, out_channels, H', W').
        """
        B, C, H, W = x.shape
        
        # Predict offsets
        offsets = self.offset_conv(x)  # (B, 2*num_points, H', W')
        
        # Output spatial dimensions
        out_H = offsets.shape[2]
        out_W = offsets.shape[3]
        
        # Reshape offsets: (B, H'*W', num_points, 2)
        offsets = offsets.view(B, 2, self.num_points, out_H, out_W)
        offsets = offsets.permute(0, 3, 4, 2, 1)  # (B, H', W', num_points, 2)
        offsets = offsets.reshape(B, out_H * out_W, self.num_points, 2)
        
        # Get base grid and add offsets
        grid = self._get_grid(B, out_H, out_W, x.device)
        sampling_locations = grid + offsets  # (B, H'*W', num_points, 2)
        
        # Normalize to [-1, 1] for grid_sample
        sampling_locations_norm = sampling_locations.clone()
        sampling_locations_norm[..., 0] = 2.0 * sampling_locations[..., 0] / max(W - 1, 1) - 1.0
        sampling_locations_norm[..., 1] = 2.0 * sampling_locations[..., 1] / max(H - 1, 1) - 1.0
        
        # Reshape for grid_sample: (B, H'*W'*num_points, 1, 2)
        sampling_locations_flat = sampling_locations_norm.view(B, -1, 1, 2)
        
        # Sample input at offset locations
        sampled = F.grid_sample(
            x,
            sampling_locations_flat,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # (B, C, H'*W'*num_points, 1)
        
        # Reshape sampled values: (B, C, H'*W', num_points)
        sampled = sampled.view(B, C, out_H * out_W, self.num_points)
        
        # Apply modulation if enabled
        if self.use_modulation:
            modulation = self.modulation_conv(x)  # (B, num_points, H', W')
            modulation = torch.sigmoid(modulation)
            modulation = modulation.view(B, self.num_points, out_H * out_W)
            modulation = modulation.permute(0, 2, 1)  # (B, H'*W', num_points)
            sampled = sampled * modulation.unsqueeze(1)
        
        # Reshape for grouped convolution
        # sampled: (B, C, H'*W', num_points) -> (B*H'*W', C, kh, kw)
        sampled = sampled.permute(0, 2, 1, 3)  # (B, H'*W', C, num_points)
        sampled = sampled.reshape(B * out_H * out_W, C, *self.kernel_size)
        
        # Apply convolution weights
        # weight: (out_C, C/groups, kh, kw)
        output = F.conv2d(sampled, self.weight, groups=self.groups)
        # output: (B*H'*W', out_C, 1, 1)
        
        # Reshape to final output
        output = output.view(B, out_H, out_W, self.out_channels)
        output = output.permute(0, 3, 1, 2)  # (B, out_C, H', W')
        
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output


class DeformableConv2dSimple(nn.Module):
    """
    Simplified Deformable Convolution using torchvision if available.
    
    Falls back to standard convolution if torchvision deform_conv2d is not available.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Check if torchvision deformable conv is available
        self._use_torchvision = False
        try:
            from torchvision.ops import deform_conv2d
            self._use_torchvision = True
            self._deform_conv2d = deform_conv2d
        except ImportError:
            pass
        
        if self._use_torchvision:
            # Offset prediction
            self.offset_conv = nn.Conv2d(
                in_channels,
                2 * kernel_size * kernel_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True
            )
            
            # Mask prediction (for DCNv2)
            self.mask_conv = nn.Conv2d(
                in_channels,
                kernel_size * kernel_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True
            )
            
            # Main weight
            self.weight = nn.Parameter(
                torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
            )
            
            if bias:
                self.bias = nn.Parameter(torch.empty(out_channels))
            else:
                self.register_parameter('bias', None)
            
            self._init_weights()
        else:
            # Fallback to regular convolution
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=bias
            )
    
    def _init_weights(self):
        """Initialize weights."""
        if self._use_torchvision:
            nn.init.zeros_(self.offset_conv.weight)
            nn.init.zeros_(self.offset_conv.bias)
            nn.init.zeros_(self.mask_conv.weight)
            nn.init.zeros_(self.mask_conv.bias)
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self._use_torchvision:
            offset = self.offset_conv(x)
            mask = torch.sigmoid(self.mask_conv(x))
            
            return self._deform_conv2d(
                x, offset, self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                mask=mask
            )
        else:
            return self.conv(x)


def create_deformable_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    use_simple: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create deformable convolution.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.
        use_simple: Whether to use simplified version with torchvision fallback.
        **kwargs: Additional arguments.
        
    Returns:
        Deformable convolution module.
    """
    if use_simple:
        return DeformableConv2dSimple(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, **kwargs
        )
    else:
        return DeformableConv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, **kwargs
        )
