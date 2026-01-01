"""
SwinIR Generator for Image Super-Resolution.

This module implements the full SwinIR architecture for license plate
super-resolution, based on the paper "SwinIR: Image Restoration Using Swin Transformer".
It uses Swin Transformer blocks with shifted window attention to restore 
high-resolution details from low-resolution inputs.

Reference: https://github.com/JingyunLiang/SwinIR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math
from einops import rearrange


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition input into non-overlapping windows.
    
    Args:
        x: Input tensor of shape (B, H, W, C).
        window_size: Window size.
        
    Returns:
        Windows of shape (num_windows*B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition.
    
    Args:
        windows: Windows of shape (num_windows*B, window_size, window_size, C).
        window_size: Window size.
        H, W: Original height and width.
        
    Returns:
        Tensor of shape (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self attention with relative position bias.
    """
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # Get pair-wise relative position index
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (num_windows*B, N, C).
            mask: Attention mask.
            
        Returns:
            Output tensor of shape (num_windows*B, N, C).
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with shifted window attention.
    """
    
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size),
            num_heads=num_heads, qkv_bias=True,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # Create attention mask for shifted window
        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class RSTB(nn.Module):
    """
    Residual Swin Transformer Block (RSTB).
    
    Contains multiple Swin Transformer blocks with residual connection.
    """
    
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            )
            for i in range(depth)
        ])
        
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        
        shortcut = x
        
        for block in self.blocks:
            x = block(x)
        
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).view(B, L, C)
        
        return x + shortcut


class Upsample(nn.Module):
    """
    Upsampling module using PixelShuffle.
    """
    
    def __init__(self, in_channels: int, scale: int):
        super().__init__()
        
        self.scale = scale
        self.conv = nn.Conv2d(in_channels, in_channels * scale * scale, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class SwinIRGenerator(nn.Module):
    """
    SwinIR-based generator for license plate super-resolution.
    
    Uses Swin Transformer blocks with shifted window attention to
    generate high-resolution, deblurred license plate images from
    low-resolution inputs.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 96,
        depths: List[int] = None,
        num_heads: List[int] = None,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        upscale: int = 4,
        img_size: Tuple[int, int] = (16, 48),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1
    ):
        """
        Initialize SwinIR Generator.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            embed_dim: Embedding dimension.
            depths: Number of blocks in each RSTB.
            num_heads: Number of attention heads in each RSTB.
            window_size: Window size for attention.
            mlp_ratio: MLP expansion ratio.
            upscale: Upscaling factor.
            img_size: Input image size (H, W).
            drop_rate: Dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
        """
        super().__init__()
        
        if depths is None:
            depths = [6, 6, 6, 6]
        if num_heads is None:
            num_heads = [6, 6, 6, 6]
        
        self.window_size = window_size
        self.upscale = upscale
        num_layers = len(depths)
        
        # Pad image size to be divisible by window size
        self.img_size = img_size
        self.padded_size = (
            math.ceil(img_size[0] / window_size) * window_size,
            math.ceil(img_size[1] / window_size) * window_size
        )
        
        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
        
        # Deep feature extraction (Swin Transformer blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.layers = nn.ModuleList()
        for i_layer in range(num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=self.padded_size,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            )
            self.layers.append(layer)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        
        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        ) if upscale == 4 else Upsample(embed_dim, upscale)
        
        # Final convolution
        self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate high-resolution image from low-resolution input.
        
        Args:
            x: Low-resolution input of shape (B, C, H, W).
            
        Returns:
            High-resolution output of shape (B, C, H*upscale, W*upscale).
        """
        B, C, H, W = x.shape
        
        # Pad to window size
        pad_h = self.padded_size[0] - H
        pad_w = self.padded_size[1] - W
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Shallow features
        x = self.conv_first(x)
        x_first = x
        
        # Convert to sequence format for transformer
        B, C_emb, H_pad, W_pad = x.shape
        x = x.permute(0, 2, 3, 1).view(B, H_pad * W_pad, C_emb)
        
        # Deep feature extraction
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        x = x.view(B, H_pad, W_pad, C_emb).permute(0, 3, 1, 2)
        
        # Residual connection
        x = self.conv_after_body(x) + x_first
        
        # Upsample
        x = self.upsample(x)
        x = self.conv_last(x)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H * self.upscale, :W * self.upscale]
        
        return x


class LightweightSRGenerator(nn.Module):
    """
    Lightweight super-resolution generator for faster inference.
    
    Uses residual blocks instead of Swin Transformer for efficiency.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 16,
        upscale: int = 4
    ):
        super().__init__()
        
        # Head
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        
        # Body (residual blocks)
        body = []
        for _ in range(num_blocks):
            body.append(self._make_res_block(num_features))
        self.body = nn.Sequential(*body)
        
        self.conv_body = nn.Conv2d(num_features, num_features, 3, 1, 1)
        
        # Upsampling
        upsample = []
        if upscale == 4:
            upsample.extend([
                nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ])
        elif upscale == 2:
            upsample.extend([
                nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ])
        self.upsample = nn.Sequential(*upsample)
        
        # Tail
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)
    
    def _make_res_block(self, features: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(features, features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, 1, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_first(x)
        res = self.conv_body(self.body(x))
        x = x + res
        x = self.upsample(x)
        x = self.conv_last(x)
        return x
