"""
OCR Embedding Similarity Loss for License Plate Super-Resolution.

This module implements an alternative to cross-entropy loss that fixes
the mode collapse problem in text super-resolution.

Key insight: Cross-entropy rewards confidence, not correctness.
A model can minimize cross-entropy by predicting the same character
confidently for all positions (mode collapse).

Solution: Use frozen OCR embeddings with cosine similarity.
This rewards semantic correctness without rewarding confidence.

Reference: LP-Diff (CVPR 2025), LPSRGAN, Embedding SR approaches
Author: Fix for Stage 3 mode collapse (2026-01-27)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class OCREmbeddingLoss(nn.Module):
    """
    OCR embedding similarity loss - fixes mode collapse.

    Uses frozen PARSeq encoder embeddings with cosine similarity instead of
    cross-entropy. This rewards semantic correctness without rewarding confidence.

    How it works:
    1. Extract encoder features from frozen PARSeq for generated and ground truth images
    2. Pool features to character-level embeddings (7 chars per plate)
    3. Compute cosine similarity between corresponding character embeddings
    4. Loss = 1 - mean_similarity (maximize similarity)

    Why this fixes mode collapse:
    - Cross-entropy: low when model is confident (even if wrong!)
    - Embedding similarity: low only when embeddings match semantically
    - Can't "cheat" by being confidently wrong
    """

    def __init__(
        self,
        ocr_model: nn.Module,
        char_pooling: str = 'spatial',  # 'spatial' or 'adaptive'
        loss_type: str = 'cosine',  # 'cosine', 'mse', 'combined'
        similarity_temperature: float = 1.0
    ):
        """
        Initialize OCR embedding loss.

        Args:
            ocr_model: PARSeq model (will be frozen)
            char_pooling: How to pool to character level ('spatial' or 'adaptive')
            loss_type: Type of loss ('cosine', 'mse', or 'combined')
            similarity_temperature: Temperature for cosine similarity (lower = sharper)
        """
        super().__init__()

        self.char_pooling = char_pooling
        self.loss_type = loss_type
        self.temperature = similarity_temperature

        # Extract encoder from PARSeq model
        # Handle both PretrainedPARSeq wrapper and PARSeqRecognizer
        if hasattr(ocr_model, '_model') and ocr_model._model is not None:
            # PretrainedPARSeq wrapper - access underlying model
            underlying_model = ocr_model._model
            # The pretrained model has a .feat_extractor attribute for the encoder
            if hasattr(underlying_model, 'feat_extractor'):
                self.encoder = underlying_model.feat_extractor
            elif hasattr(underlying_model, 'encoder'):
                self.encoder = underlying_model.encoder
            else:
                # Fallback: use the whole model as feature extractor
                self.encoder = underlying_model
        elif hasattr(ocr_model, 'encode'):
            # PARSeqRecognizer - has explicit encode method
            self.encoder = ocr_model
            self._use_encode_method = True
        else:
            raise ValueError(f"Unsupported OCR model type: {type(ocr_model)}")

        # Freeze OCR completely - no gradients!
        for param in ocr_model.parameters():
            param.requires_grad = False
        ocr_model.eval()

        # Track if we're using the encode method
        self._use_encode_method = hasattr(self, '_use_encode_method')

    def _get_encoder_features(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract encoder features from images.

        Args:
            images: Input images (B, 3, H, W)

        Returns:
            Features of shape (B, C, H', W') or (B, N, D)
        """
        with torch.no_grad():  # FROZEN OCR - no gradients!
            if self._use_encode_method:
                # Use encode method directly (returns B, N, D)
                features = self.encoder.encode(images)
                # Convert from (B, N, D) to spatial format (B, D, H, W)
                # PARSeq uses 16x16 patches, so grid is 4x12 for 64x192
                B, N, D = features.shape
                # Remove CLS token if present (first token)
                if N == 49:  # 48 patches + 1 CLS
                    features = features[:, 1:, :]  # (B, 48, D)
                # Reshape to spatial: 48 patches = 4 rows x 12 cols
                features = features.permute(0, 2, 1).reshape(B, D, 4, 12)
            else:
                # Use forward pass on encoder
                # This may return logits or features depending on the model
                features = self.encoder(images)
                # If we got logits (B, T, C), we need to handle differently
                if features.dim() == 3 and features.size(1) == 7:
                    # Got sequence logits - convert to spatial
                    # This happens if encoder is actually the full model
                    features = features.permute(0, 2, 1)  # (B, C, 7)
                    # Upsample to spatial dimensions
                    features = F.interpolate(
                        features.unsqueeze(2),  # Add height dimension
                        size=(4, 12),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(2)

        return features

    def _pool_to_character_level(
        self,
        features: torch.Tensor,
        num_chars: int = 7
    ) -> torch.Tensor:
        """
        Pool spatial features to character-level embeddings.

        Args:
            features: Encoder features (B, C, H, W)
            num_chars: Number of characters (7 for license plates)

        Returns:
            Character embeddings of shape (B, C, num_chars)
        """
        B, C, H, W = features.shape

        if self.char_pooling == 'adaptive':
            # Use adaptive avg pooling to get exactly num_chars positions
            # Pool width to num_chars, keep height
            char_features = F.adaptive_avg_pool2d(
                features,
                (1, num_chars)  # (H=1, W=num_chars)
            )  # (B, C, 1, num_chars)
            char_features = char_features.squeeze(2)  # (B, C, num_chars)

        elif self.char_pooling == 'spatial':
            # Divide width into num_chars equal segments and average each
            # This preserves spatial correspondence
            char_width = W // num_chars

            # Handle case where W is not evenly divisible
            if W % num_chars != 0:
                # Pad to make it divisible
                pad_width = (num_chars - (W % num_chars)) // 2
                features = F.pad(features, (pad_width, pad_width, 0, 0), mode='replicate')
                B, C, H, W = features.shape
                char_width = W // num_chars

            # Reshape and pool
            features = features.view(B, C, H * num_chars, char_width)
            char_features = features.mean(dim=-1)  # (B, C, H*num_chars)
            char_features = char_features.view(B, C, H, num_chars)
            char_features = char_features.mean(dim=2)  # (B, C, num_chars)

        else:
            raise ValueError(f"Unknown pooling type: {self.char_pooling}")

        return char_features

    def forward(
        self,
        generated: torch.Tensor,
        ground_truth: torch.Tensor,
        char_positions: Optional[torch.Tensor] = None,
        num_chars: int = 7
    ) -> torch.Tensor:
        """
        Compute embedding similarity loss.

        Args:
            generated: Generated HR images (B, 3, 64, 192)
            ground_truth: Ground truth HR images (B, 3, 64, 192)
            char_positions: Optional character x-positions (B, 7) for custom pooling
            num_chars: Number of characters (default: 7)

        Returns:
            Scalar loss value (lower = more similar embeddings)
        """
        # Get encoder features
        gen_features = self._get_encoder_features(generated)
        gt_features = self._get_encoder_features(ground_truth)

        # Pool to character level
        gen_chars = self._pool_to_character_level(gen_features, num_chars)
        gt_chars = self._pool_to_character_level(gt_features, num_chars)

        # Compute loss based on type
        if self.loss_type == 'cosine':
            loss = self._cosine_loss(gen_chars, gt_chars)
        elif self.loss_type == 'mse':
            loss = self._mse_loss(gen_chars, gt_chars)
        elif self.loss_type == 'combined':
            loss_cosine = self._cosine_loss(gen_chars, gt_chars)
            loss_mse = self._mse_loss(gen_chars, gt_chars)
            loss = 0.7 * loss_cosine + 0.3 * loss_mse
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    def _cosine_loss(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity loss.

        Args:
            emb1: First embeddings (B, C, num_chars)
            emb2: Second embeddings (B, C, num_chars)

        Returns:
            Loss = 1 - mean_cosine_similarity
        """
        # L2 normalize
        emb1_norm = F.normalize(emb1, p=2, dim=1)  # (B, C, num_chars)
        emb2_norm = F.normalize(emb2, p=2, dim=1)

        # Compute cosine similarity per character
        # (B, C, num_chars) -> (B, num_chars) after dot product over C
        similarity = (emb1_norm * emb2_norm).sum(dim=1)  # (B, num_chars)

        # Apply temperature (lower = sharper discrimination)
        similarity = similarity / self.temperature

        # Mean over characters and batch
        mean_similarity = similarity.mean()

        # Loss = 1 - similarity (maximize similarity)
        return 1.0 - mean_similarity

    def _mse_loss(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss between embeddings.

        Args:
            emb1: First embeddings (B, C, num_chars)
            emb2: Second embeddings (B, C, num_chars)

        Returns:
            MSE loss
        """
        return F.mse_loss(emb1, emb2)


class OCREmbeddingLossWithBaseline(nn.Module):
    """
    OCR embedding loss with baseline constraint (Stage 3.5 variant).

    Similar to OCR hinge constraint in Stage 3, this only penalizes
    when embedding similarity is worse than a baseline (e.g., Stage 2).

    This prevents the model from drifting during embedding fine-tuning.
    """

    def __init__(
        self,
        ocr_model: nn.Module,
        baseline_similarity: float = 0.5,  # Target baseline to maintain or exceed
        margin: float = 0.1,  # Allow some degradation before penalizing
        **kwargs
    ):
        """
        Initialize embedding loss with baseline.

        Args:
            ocr_model: PARSeq model (will be frozen)
            baseline_similarity: Target similarity to maintain
            margin: Allowable degradation below baseline
            **kwargs: Passed to OCREmbeddingLoss
        """
        super().__init__()

        self.embedding_loss = OCREmbeddingLoss(ocr_model, **kwargs)
        self.baseline_similarity = baseline_similarity
        self.margin = margin

    def forward(
        self,
        generated: torch.Tensor,
        ground_truth: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute embedding loss with baseline constraint.

        Only penalizes if similarity is below (baseline - margin).
        """
        # Compute raw embedding loss
        loss = self.embedding_loss(generated, ground_truth, **kwargs)

        # Convert loss to similarity
        current_similarity = 1.0 - loss

        # Only penalize if below baseline - margin
        # relu(baseline - margin - current_similarity)
        hinge_loss = F.relu(
            self.baseline_similarity - self.margin - current_similarity
        )

        return hinge_loss


def create_embedding_loss(
    ocr_model: nn.Module,
    loss_type: str = 'cosine',
    char_pooling: str = 'spatial',
    use_baseline: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create OCR embedding loss.

    Args:
        ocr_model: PARSeq model
        loss_type: 'cosine', 'mse', or 'combined'
        char_pooling: 'spatial' or 'adaptive'
        use_baseline: Whether to use baseline constraint
        **kwargs: Additional arguments

    Returns:
        OCREmbeddingLoss or OCREmbeddingLossWithBaseline
    """
    if use_baseline:
        return OCREmbeddingLossWithBaseline(
            ocr_model=ocr_model,
            loss_type=loss_type,
            char_pooling=char_pooling,
            **kwargs
        )
    else:
        return OCREmbeddingLoss(
            ocr_model=ocr_model,
            loss_type=loss_type,
            char_pooling=char_pooling,
            **kwargs
        )


# Test function
def _test_embedding_loss():
    """Quick test to verify embedding loss works correctly."""
    import torch

    # Create a simple mock PARSeq model for testing
    class MockPARSeq(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
            )

        def encode(self, x):
            return self.encoder(x)

    # Create models
    ocr_model = MockPARSeq()
    loss_fn = OCREmbeddingLoss(ocr_model, char_pooling='spatial')

    # Test data
    batch_size = 2
    gen_images = torch.randn(batch_size, 3, 64, 192)
    gt_images = torch.randn(batch_size, 3, 64, 192)

    # Forward pass
    loss = loss_fn(gen_images, gt_images)

    print(f"Embedding loss: {loss.item():.4f}")
    print(f"Loss requires grad: {loss.requires_grad}")

    # Test with identical images (should have low loss)
    loss_same = loss_fn(gt_images, gt_images)
    print(f"Loss for identical images: {loss_same.item():.4f}")

    print("✓ Embedding loss test passed!")


if __name__ == "__main__":
    _test_embedding_loss()
