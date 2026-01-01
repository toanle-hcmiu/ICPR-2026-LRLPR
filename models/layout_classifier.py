"""
Layout Classifier for Brazilian vs Mercosul Plate Detection.

This module implements the Layout Switch branch that classifies license plates
as either old Brazilian format (LLL-NNNN) or Mercosul format (LLLNLNN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayoutClassifier(nn.Module):
    """
    Layout Switch branch for plate type classification.
    
    This lightweight classifier determines whether a license plate follows
    the old Brazilian format or the newer Mercosul format. The classification
    result is used to select the appropriate syntax mask for recognition.
    
    Architecture:
        - Tiny CNN for feature extraction
        - Global Average Pooling
        - Fully Connected layer
        - Sigmoid activation for binary classification
    
    Visual differences detected:
        - Color scheme (grey vs white with blue band)
        - Character layout (LLL-NNNN vs LLLNLNN)
        - Presence of QR code and emblems (Mercosul only)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list = None,
        fc_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        Initialize the Layout Classifier.
        
        Args:
            in_channels: Number of input feature channels.
            hidden_channels: Channel sizes for CNN layers.
            fc_dim: Dimension of the fully connected layer.
            dropout: Dropout probability.
        """
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        
        # CNN for feature extraction
        layers = []
        prev_channels = in_channels
        
        for out_channels in hidden_channels:
            layers.extend([
                nn.Conv2d(prev_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            prev_channels = out_channels
        
        self.cnn = nn.Sequential(*layers)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels[-1], fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1)  # Binary classification
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify the layout of the license plate.
        
        Args:
            x: Input feature map of shape (B, C, H, W).
            
        Returns:
            Logits of shape (B, 1) for binary classification.
            Positive values indicate Mercosul, negative indicate Brazilian.
        """
        batch_size = x.size(0)
        
        # Extract features
        features = self.cnn(x)
        
        # Global pooling
        pooled = self.gap(features).view(batch_size, -1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict layout type with hard decision.
        
        Args:
            x: Input feature map of shape (B, C, H, W).
            threshold: Decision threshold for sigmoid output.
            
        Returns:
            Binary predictions of shape (B,). 
            0 = Brazilian, 1 = Mercosul.
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        predictions = (probs >= threshold).long().squeeze(-1)
        return predictions
    
    def forward_multi_frame(
        self, 
        x: torch.Tensor,
        aggregation: str = 'mean'
    ) -> torch.Tensor:
        """
        Classify layout from multiple frames.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W).
            aggregation: How to aggregate multi-frame predictions.
                         Options: 'mean', 'max', 'first'.
            
        Returns:
            Logits of shape (B, 1).
        """
        B, T, C, H, W = x.shape
        
        # Process each frame
        x = x.view(B * T, C, H, W)
        logits = self.forward(x)  # (B*T, 1)
        logits = logits.view(B, T, 1)
        
        # Aggregate across frames
        if aggregation == 'mean':
            logits = logits.mean(dim=1)
        elif aggregation == 'max':
            logits = logits.max(dim=1)[0]
        elif aggregation == 'first':
            logits = logits[:, 0, :]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        return logits


class LayoutClassifierWithAttention(nn.Module):
    """
    Enhanced layout classifier with attention mechanism.
    
    Uses self-attention to focus on the most discriminative regions
    of the license plate (e.g., the blue band on Mercosul plates).
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Project to hidden dimension
        self.project = nn.Conv2d(in_channels, hidden_dim, 1)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify with attention.
        
        Args:
            x: Input feature map of shape (B, C, H, W).
            
        Returns:
            Logits of shape (B, 1).
        """
        B, C, H, W = x.shape
        
        # Project and reshape for attention
        x = self.project(x)  # (B, hidden_dim, H, W)
        x = x.view(B, -1, H * W).transpose(1, 2)  # (B, H*W, hidden_dim)
        
        # Apply self-attention
        attended, _ = self.attention(x, x, x)
        x = self.norm(x + attended)
        
        # Global pooling over spatial positions
        x = x.mean(dim=1)  # (B, hidden_dim)
        
        # Classify
        logits = self.classifier(x)
        
        return logits
