"""
Spatial Transformer Network (STN) for Geometry-Constrained Alignment.

This module implements the STN that rectifies license plate images by
learning to predict affine transformation parameters. The STN is supervised
using corner loss for accurate geometric alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class LocalizationNetwork(nn.Module):
    """
    Localization network that predicts affine transformation parameters.
    
    Architecture:
        - Small CNN for feature extraction
        - Fully connected layers for parameter regression
        - Outputs 6 parameters (2x3 affine matrix)
    """
    
    def __init__(
        self,
        in_channels: int,
        feature_height: int,
        feature_width: int,
        hidden_channels: list = None,
        fc_dims: list = None
    ):
        """
        Initialize the localization network.
        
        Args:
            in_channels: Number of input feature channels.
            feature_height: Height of input feature map.
            feature_width: Width of input feature map.
            hidden_channels: Channel sizes for conv layers.
            fc_dims: Dimensions for fully connected layers.
        """
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        if fc_dims is None:
            fc_dims = [256, 128]
        
        # Convolutional layers
        conv_layers = []
        prev_channels = in_channels
        
        for out_channels in hidden_channels:
            conv_layers.extend([
                nn.Conv2d(prev_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            prev_channels = out_channels
        
        self.conv = nn.Sequential(*conv_layers)
        
        # Calculate flattened size after convolutions
        num_pools = len(hidden_channels)
        final_h = feature_height // (2 ** num_pools)
        final_w = feature_width // (2 ** num_pools)
        flatten_size = hidden_channels[-1] * max(1, final_h) * max(1, final_w)
        
        # Fully connected layers
        fc_layers = []
        prev_dim = flatten_size
        
        for dim in fc_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = dim
        
        # Output layer for 6 affine parameters
        fc_layers.append(nn.Linear(prev_dim, 6))
        
        self.fc = nn.Sequential(*fc_layers)
        
        # Initialize to identity transformation
        self._init_identity()
    
    def _init_identity(self):
        """Initialize the final layer to output identity transformation."""
        # Identity affine: [[1, 0, 0], [0, 1, 0]]
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict affine transformation parameters.
        
        Args:
            x: Input feature map of shape (B, C, H, W).
            
        Returns:
            Affine parameters of shape (B, 2, 3).
        """
        batch_size = x.size(0)
        
        # Apply convolutions
        x = self.conv(x)
        
        # Flatten and apply FC layers
        x = x.view(batch_size, -1)
        theta = self.fc(x)
        
        # Reshape to 2x3 matrix
        theta = theta.view(batch_size, 2, 3)
        
        return theta


class CornerPredictor(nn.Module):
    """
    Auxiliary head that predicts license plate corner coordinates.
    
    Used for direct supervision of geometric alignment via corner loss.
    """
    
    def __init__(
        self,
        in_channels: int,
        feature_height: int,
        feature_width: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # Global average pooling + FC for corner prediction
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 8)  # 4 corners * 2 coordinates
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict corner coordinates.
        
        Args:
            features: Feature map of shape (B, C, H, W).
            
        Returns:
            Corner coordinates of shape (B, 4, 2) normalized to [-1, 1].
        """
        batch_size = features.size(0)
        
        x = self.gap(features).view(batch_size, -1)
        corners = self.fc(x)
        corners = corners.view(batch_size, 4, 2)
        
        # Apply tanh to constrain to [-1, 1]
        corners = torch.tanh(corners)
        
        return corners


class SpatialTransformerNetwork(nn.Module):
    """
    Spatial Transformer Network for license plate rectification.
    
    This module learns to perform affine transformations to correct
    perspective distortion, rotation, and skew in license plate images.
    It can be supervised using corner annotations for precise alignment.
    
    Components:
        - Localization Network: Predicts affine parameters
        - Grid Generator: Creates sampling grid from affine matrix
        - Sampler: Bilinear interpolation for differentiable warping
        - Corner Predictor: Auxiliary head for corner supervision
    """
    
    def __init__(
        self,
        in_channels: int,
        feature_height: int,
        feature_width: int,
        output_size: Tuple[int, int] = None,
        hidden_channels: list = None,
        fc_dims: list = None,
        use_corner_predictor: bool = True
    ):
        """
        Initialize the Spatial Transformer Network.
        
        Args:
            in_channels: Number of input feature channels.
            feature_height: Height of input feature map.
            feature_width: Width of input feature map.
            output_size: Size of output rectified image (H, W).
            hidden_channels: Channel sizes for localization conv layers.
            fc_dims: Dimensions for localization FC layers.
            use_corner_predictor: Whether to use corner prediction head.
        """
        super().__init__()
        
        self.output_size = output_size or (feature_height, feature_width)
        self.use_corner_predictor = use_corner_predictor
        
        # Localization network
        self.localization = LocalizationNetwork(
            in_channels=in_channels,
            feature_height=feature_height,
            feature_width=feature_width,
            hidden_channels=hidden_channels,
            fc_dims=fc_dims
        )
        
        # Optional corner predictor for supervision
        if use_corner_predictor:
            self.corner_predictor = CornerPredictor(
                in_channels=in_channels,
                feature_height=feature_height,
                feature_width=feature_width
            )
    
    def forward(
        self, 
        x: torch.Tensor,
        return_theta: bool = False,
        return_corners: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Apply spatial transformation to input features.
        
        Args:
            x: Input feature map of shape (B, C, H, W).
            return_theta: Whether to return the affine parameters.
            return_corners: Whether to return predicted corners.
            
        Returns:
            Tuple containing:
                - Rectified feature map of shape (B, C, H', W')
                - (optional) Affine parameters of shape (B, 2, 3)
                - (optional) Corner predictions of shape (B, 4, 2)
        """
        # Get affine transformation parameters
        theta = self.localization(x)
        
        # Generate sampling grid
        grid = F.affine_grid(
            theta, 
            [x.size(0), x.size(1), self.output_size[0], self.output_size[1]],
            align_corners=False
        )
        
        # Sample from input using grid (bilinear interpolation)
        rectified = F.grid_sample(
            x, grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=False
        )
        
        # Prepare outputs
        outputs = [rectified]
        
        if return_theta:
            outputs.append(theta)
        
        if return_corners and self.use_corner_predictor:
            corners = self.corner_predictor(x)
            outputs.append(corners)
        
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
    
    def transform_corners(
        self, 
        theta: torch.Tensor, 
        corners: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform corner coordinates using affine matrix.
        
        Args:
            theta: Affine parameters of shape (B, 2, 3).
            corners: Corner coordinates of shape (B, 4, 2).
            
        Returns:
            Transformed corners of shape (B, 4, 2).
        """
        batch_size = corners.size(0)
        
        # Add homogeneous coordinate
        ones = torch.ones(batch_size, 4, 1, device=corners.device)
        corners_h = torch.cat([corners, ones], dim=-1)  # (B, 4, 3)
        
        # Transform: theta @ corners^T -> (B, 2, 4) -> (B, 4, 2)
        transformed = torch.bmm(theta, corners_h.transpose(1, 2))
        transformed = transformed.transpose(1, 2)
        
        return transformed
    
    def get_corners_from_theta(
        self, 
        theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the corner positions of the transformed region.
        
        Given the affine transformation, computes where the corners
        of the canonical rectangle [-1, 1] x [-1, 1] are mapped.
        
        Args:
            theta: Affine parameters of shape (B, 2, 3).
            
        Returns:
            Corner coordinates of shape (B, 4, 2).
        """
        batch_size = theta.size(0)
        device = theta.device
        
        # Canonical corners (top-left, top-right, bottom-right, bottom-left)
        canonical_corners = torch.tensor([
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1]
        ], dtype=torch.float, device=device)
        
        # Expand for batch
        canonical_corners = canonical_corners.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Transform corners
        return self.transform_corners(theta, canonical_corners)


class MultiFrameSTN(nn.Module):
    """
    Spatial Transformer Network for multiple frames.
    
    Processes multiple frames independently using shared STN weights.
    """
    
    def __init__(self, stn: SpatialTransformerNetwork):
        super().__init__()
        self.stn = stn
    
    def forward(
        self, 
        x: torch.Tensor,
        return_theta: bool = False,
        return_corners: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Apply STN to multiple frames.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W).
            return_theta: Whether to return affine parameters.
            return_corners: Whether to return corner predictions.
            
        Returns:
            Rectified features and optional additional outputs.
        """
        B, T, C, H, W = x.shape
        
        # Reshape to process all frames
        x = x.view(B * T, C, H, W)
        
        # Apply STN
        result = self.stn(x, return_theta=return_theta, return_corners=return_corners)
        
        # Unpack results
        if isinstance(result, tuple):
            rectified = result[0]
            other_outputs = list(result[1:])
        else:
            rectified = result
            other_outputs = []
        
        # Reshape rectified features
        _, C_out, H_out, W_out = rectified.shape
        rectified = rectified.view(B, T, C_out, H_out, W_out)
        
        # Reshape other outputs
        reshaped_outputs = [rectified]
        for output in other_outputs:
            if len(output.shape) == 3:
                # Affine parameters (B*T, 2, 3) -> (B, T, 2, 3)
                output = output.view(B, T, output.size(1), output.size(2))
            elif len(output.shape) == 3:
                # Corners (B*T, 4, 2) -> (B, T, 4, 2)
                output = output.view(B, T, output.size(1), output.size(2))
            reshaped_outputs.append(output)
        
        if len(reshaped_outputs) == 1:
            return reshaped_outputs[0]
        return tuple(reshaped_outputs)
