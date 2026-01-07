"""
PARSeq Recognizer for License Plate Character Recognition.

This module provides two options for PARSeq-based recognition:
1. Pretrained PARSeq from the official repository (recommended)
2. Full custom implementation (fallback)

The pretrained model can be loaded from:
- GitHub: https://github.com/baudm/parseq
- torch.hub: torch.hub.load('baudm/parseq', 'parseq')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
import math
import warnings

from config import VOCAB_SIZE, PLATE_LENGTH, PAD_IDX, BOS_IDX, EOS_IDX, CHAR_START_IDX, CHARSET


# =============================================================================
# Pretrained PARSeq Wrapper (Recommended)
# =============================================================================

class PretrainedPARSeq(nn.Module):
    """
    Wrapper for the official pretrained PARSeq model.
    
    This uses the model from https://github.com/baudm/parseq which
    was trained on a large-scale scene text recognition dataset.
    
    Benefits of using pretrained:
    - Much better OCR accuracy out-of-the-box
    - Trained on millions of images
    - State-of-the-art performance on standard benchmarks
    
    The output is adapted to match our plate recognition format.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        model_name: str = 'parseq',  # Options: 'parseq', 'parseq_tiny'
        freeze_backbone: bool = False,
        output_raw_logits: bool = True,
        img_size: tuple = (64, 192)  # For fallback custom implementation
    ):
        """
        Initialize pretrained PARSeq wrapper.
        
        Args:
            pretrained: Whether to load pretrained weights.
            model_name: Model variant ('parseq' or 'parseq_tiny').
            freeze_backbone: Whether to freeze the backbone encoder.
            output_raw_logits: Whether to output raw logits for syntax masking.
            img_size: Image size for fallback custom implementation.
        """
        super().__init__()
        
        self.output_raw_logits = output_raw_logits
        self.model_name = model_name
        self._model = None
        self._fallback_model = None  # Custom implementation fallback
        self._charset_adapter = None
        self._img_size = img_size
        
        # Lazy loading to avoid import errors if torch.hub fails
        self._pretrained = pretrained
        self._freeze_backbone = freeze_backbone
        self._loaded = False
        self._use_fallback = False
    
    def _load_model(self):
        """Lazy load the pretrained model."""
        if self._loaded:
            return
        
        try:
            # Load from torch.hub
            self._model = torch.hub.load(
                'baudm/parseq', 
                self.model_name,
                pretrained=self._pretrained,
                trust_repo=True
            )
            
            # Freeze backbone if requested
            if self._freeze_backbone:
                for name, param in self._model.named_parameters():
                    if 'encoder' in name or 'embed' in name:
                        param.requires_grad = False
            
            # Create charset adapter (pretrained uses different charset)
            self._create_charset_adapter()
            
            self._loaded = True
            self._use_fallback = False
            print(f"✓ Loaded pretrained PARSeq ({self.model_name})")
            
        except Exception as e:
            warnings.warn(
                f"Failed to load pretrained PARSeq: {e}. "
                f"Using custom implementation instead."
            )
            # Create fallback custom implementation
            self._fallback_model = PARSeqRecognizer(
                img_size=self._img_size,
                embed_dim=384,
                num_heads=6,
                encoder_depth=12
            )
            # Move to same device as self
            if hasattr(self, '_device'):
                self._fallback_model = self._fallback_model.to(self._device)
            
            self._model = None
            self._use_fallback = True
            self._loaded = True
            print(f"✓ Initialized custom PARSeq fallback implementation")
    
    def _create_charset_adapter(self):
        """
        Create mapping between pretrained charset and our plate charset.
        
        The pretrained model uses a different character set, so we need
        to map its output indices to our CHARSET indices.
        """
        if self._model is None:
            return
        
        # Get pretrained model's charset
        pretrained_charset = self._model.tokenizer.charset
        
        # Create mapping from pretrained indices to our indices
        pretrained_to_ours = {}
        for i, char in enumerate(pretrained_charset):
            if char.upper() in CHARSET:
                our_idx = CHAR_START_IDX + CHARSET.index(char.upper())
                pretrained_to_ours[i] = our_idx
        
        self._pretrained_to_ours = pretrained_to_ours
        self._pretrained_charset_size = len(pretrained_charset)
    
    def _adapt_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Adapt pretrained model's logits to our vocabulary.
        
        Args:
            logits: Logits from pretrained model (B, T, pretrained_vocab_size).
            
        Returns:
            Adapted logits (B, PLATE_LENGTH, VOCAB_SIZE).
        """
        B, T, V_pretrained = logits.shape
        device = logits.device
        
        # Create output logits initialized to very negative values
        adapted = torch.full(
            (B, PLATE_LENGTH, VOCAB_SIZE), 
            fill_value=-100.0, 
            device=device
        )
        
        # Map pretrained indices to our indices
        for pretrained_idx, our_idx in self._pretrained_to_ours.items():
            if pretrained_idx < V_pretrained:
                # Take first PLATE_LENGTH positions
                adapted[:, :, our_idx] = logits[:, :PLATE_LENGTH, pretrained_idx]
        
        return adapted
    
    def forward(
        self,
        x: torch.Tensor,
        tgt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass using pretrained model or fallback.
        
        Args:
            x: Input image (B, C, H, W).
            tgt: Target tokens (ignored, for API compatibility).
            
        Returns:
            Logits of shape (B, PLATE_LENGTH, VOCAB_SIZE).
        """
        # Store device for fallback model creation
        self._device = x.device
        self._load_model()
        
        # Use fallback custom implementation if pretrained failed
        if self._use_fallback:
            # Move fallback model to correct device if needed
            if self._fallback_model is not None:
                if next(self._fallback_model.parameters()).device != x.device:
                    self._fallback_model = self._fallback_model.to(x.device)
                return self._fallback_model(x)
            else:
                raise RuntimeError("Neither pretrained nor fallback model available.")
        
        if self._model is None:
            raise RuntimeError("PARSeq model not loaded. Check your internet connection.")
        
        # Pretrained model expects images in [0, 1] range
        # Our images are in [-1, 1] range
        x_normalized = (x + 1) / 2
        
        # Get predictions from pretrained model
        with torch.cuda.amp.autocast(enabled=False):
            logits = self._model(x_normalized)
        
        # Adapt to our charset
        adapted_logits = self._adapt_logits(logits)
        
        return adapted_logits
    
    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel forward pass (same as forward for pretrained)."""
        return self.forward(x)
    
    def parameters(self, recurse: bool = True):
        """Return parameters from the active model."""
        self._load_model()
        if self._use_fallback and self._fallback_model is not None:
            return self._fallback_model.parameters(recurse)
        elif self._model is not None:
            return self._model.parameters(recurse)
        else:
            return iter([])  # Empty iterator
    
    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        if self._use_fallback and self._fallback_model is not None:
            self._fallback_model.train(mode)
        elif self._model is not None:
            self._model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)


def load_pretrained_parseq(
    model_name: str = 'parseq',
    freeze_backbone: bool = False,
    device: torch.device = None
) -> PretrainedPARSeq:
    """
    Convenience function to load pretrained PARSeq.
    
    Args:
        model_name: 'parseq' (full) or 'parseq_tiny' (lightweight).
        freeze_backbone: Whether to freeze encoder weights.
        device: Device to load model on.
        
    Returns:
        PretrainedPARSeq model.
    """
    model = PretrainedPARSeq(
        pretrained=True,
        model_name=model_name,
        freeze_backbone=freeze_backbone
    )
    
    if device is not None:
        model = model.to(device)
    
    return model


# =============================================================================
# Custom PARSeq Implementation (Fallback)
# =============================================================================


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding using convolutions.
    
    Converts input image into a sequence of patch embeddings.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 192),
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patch embeddings.
        
        Args:
            x: Input image of shape (B, C, H, W).
            
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim).
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # Self-attention
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention
        tgt2 = self.cross_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class PARSeqRecognizer(nn.Module):
    """
    PARSeq-style recognizer for license plate character recognition.
    
    Uses a Vision Transformer encoder to extract visual features and
    a Transformer decoder for autoregressive sequence prediction.
    The model produces raw logits that can be constrained by syntax masks.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 192),
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
        num_heads: int = 6,
        encoder_depth: int = 12,
        decoder_depth: int = 1,
        mlp_ratio: float = 4.0,
        num_classes: int = VOCAB_SIZE,
        max_length: int = PLATE_LENGTH + 2,  # +2 for BOS and EOS
        dropout: float = 0.1
    ):
        """
        Initialize PARSeq recognizer.
        
        Args:
            img_size: Input image size (H, W).
            patch_size: Patch size for ViT.
            in_channels: Number of input channels.
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            encoder_depth: Number of encoder layers.
            decoder_depth: Number of decoder layers.
            mlp_ratio: MLP expansion ratio.
            num_classes: Number of output classes (vocabulary size).
            max_length: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.max_length = max_length
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Encoder
        dim_feedforward = int(embed_dim * mlp_ratio)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # Token embedding for decoder
        self.token_embed = nn.Embedding(num_classes, embed_dim)
        self.token_pos_embed = PositionalEncoding(embed_dim, max_length, dropout)
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
    
    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input image into visual features.
        
        Args:
            x: Input image of shape (B, C, H, W).
            
        Returns:
            Visual features of shape (B, num_patches+1, embed_dim).
        """
        B = x.size(0)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = self.encoder_norm(x)
        
        return x
    
    def decode(
        self,
        memory: torch.Tensor,
        tgt: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode visual features into character logits.
        
        Args:
            memory: Encoder output of shape (B, N, embed_dim).
            tgt: Target tokens of shape (B, T).
            
        Returns:
            Logits of shape (B, T, num_classes).
        """
        # Token embedding
        tgt = self.token_embed(tgt)  # (B, T, embed_dim)
        tgt = self.token_pos_embed(tgt)
        
        # Causal mask
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)
        
        # Transformer decoder
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, tgt_mask)
        
        tgt = self.decoder_norm(tgt)
        
        # Project to vocabulary
        logits = self.head(tgt)
        
        return logits
    
    def forward(
        self,
        x: torch.Tensor,
        tgt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training or inference.
        
        Args:
            x: Input image of shape (B, C, H, W).
            tgt: Target tokens for training (B, T), or None for inference.
            
        Returns:
            If tgt is provided: Logits of shape (B, T, num_classes).
            If tgt is None: Raw logits for all positions (B, max_length-1, num_classes).
        """
        # Encode image
        memory = self.encode(x)
        
        if tgt is not None:
            # Training mode: teacher forcing
            logits = self.decode(memory, tgt)
        else:
            # Inference mode: autoregressive decoding
            B = x.size(0)
            device = x.device
            
            # Start with BOS token
            tgt = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
            
            # Full autoregressive decoding: loop and append
            all_logits = []
            for _ in range(self.max_length - 1):
                logits = self.decode(memory, tgt)
                next_logits = logits[:, -1:, :]  # (B, 1, num_classes)
                all_logits.append(next_logits)
                
                # Greedy selection for next token
                next_token = next_logits.argmax(dim=-1)  # (B, 1)
                tgt = torch.cat([tgt, next_token], dim=1)
            
            logits = torch.cat(all_logits, dim=1)  # (B, max_length-1, num_classes)
        
        return logits
    
    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parallel forward pass for getting raw logits.
        
        This is used during training with syntax masks, where we want
        logits for all positions simultaneously.
        
        Args:
            x: Input image of shape (B, C, H, W).
            
        Returns:
            Raw logits of shape (B, PLATE_LENGTH, num_classes).
        """
        B = x.size(0)
        device = x.device
        
        # Encode image
        memory = self.encode(x)
        
        # Create position queries for all plate positions
        positions = torch.arange(PLATE_LENGTH, device=device).unsqueeze(0).expand(B, -1)
        
        # Simple approach: use position embeddings as queries
        query_embed = self.token_embed.weight[CHAR_START_IDX:CHAR_START_IDX + PLATE_LENGTH]
        query_embed = query_embed.unsqueeze(0).expand(B, -1, -1)
        
        # Cross-attention only (no self-attention mask needed)
        for layer in self.decoder_layers:
            query_embed = layer(query_embed, memory)
        
        query_embed = self.decoder_norm(query_embed)
        logits = self.head(query_embed)
        
        return logits


class CTCRecognizer(nn.Module):
    """
    Alternative CTC-based recognizer for license plate recognition.
    
    Uses CTC loss which doesn't require explicit alignment between
    input and output sequences.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 192),
        in_channels: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = VOCAB_SIZE
    ):
        super().__init__()
        
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Calculate feature width
        self.feature_height = img_size[0] // 8
        self.feature_width = img_size[1]  # No width pooling in last layers
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            hidden_dim * self.feature_height,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image of shape (B, C, H, W).
            
        Returns:
            Logits of shape (B, T, num_classes) where T is sequence length.
        """
        # CNN features
        features = self.cnn(x)  # (B, C, H', W')
        
        B, C, H, W = features.shape
        
        # Reshape for RNN: (B, W, C*H)
        features = features.permute(0, 3, 1, 2).contiguous()
        features = features.view(B, W, -1)
        
        # RNN
        rnn_out, _ = self.rnn(features)  # (B, W, hidden*2)
        
        # Output projection
        logits = self.fc(rnn_out)  # (B, W, num_classes)
        
        return logits
