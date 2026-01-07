"""
Dynamic Syntax Mask Layer for Constrained Recognition.

This module implements the core neuro-symbolic component that enforces
Brazilian license plate syntax rules during character recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import (
    LETTERS, DIGITS, CHARSET, VOCAB_SIZE, PLATE_LENGTH,
    PAD_IDX, BOS_IDX, EOS_IDX, CHAR_START_IDX,
    get_position_constraints
)


def create_letter_mask(vocab_size: int, char_start_idx: int) -> torch.Tensor:
    """
    Create a mask that allows only letters.
    
    Args:
        vocab_size: Total vocabulary size.
        char_start_idx: Index where characters start in vocabulary.
        
    Returns:
        Mask tensor where 0 allows the character and -inf forbids it.
    """
    mask = torch.full((vocab_size,), float('-inf'))
    
    # Allow letters (first 26 characters after special tokens)
    for i, char in enumerate(CHARSET):
        if char in LETTERS:
            mask[char_start_idx + i] = 0.0
    
    return mask


def create_digit_mask(vocab_size: int, char_start_idx: int) -> torch.Tensor:
    """
    Create a mask that allows only digits.
    
    Args:
        vocab_size: Total vocabulary size.
        char_start_idx: Index where characters start in vocabulary.
        
    Returns:
        Mask tensor where 0 allows the character and -inf forbids it.
    """
    mask = torch.full((vocab_size,), float('-inf'))
    
    # Allow digits (characters after letters in charset)
    for i, char in enumerate(CHARSET):
        if char in DIGITS:
            mask[char_start_idx + i] = 0.0
    
    return mask


class SyntaxMaskLayer(nn.Module):
    """
    Dynamic Syntax Mask Layer for enforcing license plate format.
    
    This is the core neuro-symbolic component that constrains the
    recognizer's output to only produce syntactically valid license
    plate strings based on the detected layout type.
    
    Brazilian format: LLL-NNNN (positions 0-2 letters, 3-6 numbers)
    Mercosul format: LLLNLNN (positions 0-2,4 letters, 3,5,6 numbers)
    
    The mask is applied by adding it to the raw logits:
        - 0 for allowed characters (no effect)
        - -inf (or large negative) for forbidden characters (probability → 0)
    
    Soft inference mode:
        For robustness to damaged/non-standard plates, soft_inference mode
        uses a less aggressive penalty during inference, allowing the model
        to predict "invalid" characters if visual evidence is overwhelming.
    """
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        plate_length: int = PLATE_LENGTH,
        char_start_idx: int = CHAR_START_IDX,
        soft_mask_value: float = -100.0,
        soft_inference: bool = False,
        soft_inference_value: float = -50.0
    ):
        """
        Initialize the syntax mask layer.
        
        Args:
            vocab_size: Size of the character vocabulary.
            plate_length: Number of characters in the license plate.
            char_start_idx: Index where regular characters start.
            soft_mask_value: Value for soft masking during training.
            soft_inference: Whether to use soft constraints during inference.
            soft_inference_value: Penalty value for soft inference mode.
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.plate_length = plate_length
        self.char_start_idx = char_start_idx
        self.soft_mask_value = soft_mask_value
        self.soft_inference = soft_inference
        self.soft_inference_value = soft_inference_value
        
        # Pre-compute masks for letters and digits
        letter_mask = create_letter_mask(vocab_size, char_start_idx)
        digit_mask = create_digit_mask(vocab_size, char_start_idx)
        
        self.register_buffer('letter_mask', letter_mask)
        self.register_buffer('digit_mask', digit_mask)
        
        # Pre-compute full masks for both layouts
        brazilian_mask = self._create_layout_mask(is_mercosul=False)
        mercosul_mask = self._create_layout_mask(is_mercosul=True)
        
        self.register_buffer('brazilian_mask', brazilian_mask)
        self.register_buffer('mercosul_mask', mercosul_mask)
    
    def _create_layout_mask(self, is_mercosul: bool) -> torch.Tensor:
        """
        Create the full mask for a specific layout.
        
        Args:
            is_mercosul: Whether to create mask for Mercosul format.
            
        Returns:
            Mask tensor of shape (plate_length, vocab_size).
        """
        constraints = get_position_constraints(is_mercosul)
        mask = torch.zeros(self.plate_length, self.vocab_size)
        
        for pos, constraint in enumerate(constraints):
            if constraint == 'L':
                mask[pos] = self.letter_mask.clone()
            else:  # 'N'
                mask[pos] = self.digit_mask.clone()
        
        return mask
    
    def get_mask(
        self,
        is_mercosul: torch.Tensor,
        use_soft_mask: bool = True
    ) -> torch.Tensor:
        """
        Get the syntax mask based on layout classification.
        
        Args:
            is_mercosul: Binary tensor of shape (B,) or (B, 1).
            use_soft_mask: Whether to use soft mask for training.
            
        Returns:
            Mask tensor of shape (B, plate_length, vocab_size).
        """
        if is_mercosul.dim() == 2:
            is_mercosul = is_mercosul.squeeze(-1)
        
        batch_size = is_mercosul.size(0)
        device = is_mercosul.device
        
        # Convert probabilities to hard decisions for mask selection
        is_mercosul_hard = (is_mercosul > 0.5).float()
        
        # Get masks for each sample
        # Use broadcasting: (B, 1, 1) * (plate_length, vocab_size)
        mercosul_weight = is_mercosul_hard.view(batch_size, 1, 1)
        brazilian_weight = 1.0 - mercosul_weight
        
        mask = (
            brazilian_weight * self.brazilian_mask.unsqueeze(0) +
            mercosul_weight * self.mercosul_mask.unsqueeze(0)
        )
        
        # Use soft mask value during training
        if use_soft_mask:
            mask = mask.masked_fill(mask == float('-inf'), self.soft_mask_value)
        
        return mask
    
    def forward(
        self,
        logits: torch.Tensor,
        is_mercosul: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        Apply syntax mask to raw logits.
        
        Args:
            logits: Raw logits of shape (B, plate_length, vocab_size).
            is_mercosul: Binary tensor of shape (B,) or (B, 1).
            training: Whether in training mode (uses soft mask).
            
        Returns:
            Masked logits of shape (B, plate_length, vocab_size).
        """
        # Use soft mask if:
        # 1. Training (always soft for gradient flow)
        # 2. Inference with soft_inference=True (for robustness to anomalies)
        use_soft = training or self.soft_inference
        mask = self.get_mask(is_mercosul, use_soft_mask=use_soft)
        
        # Use different soft values for training vs soft inference
        if use_soft and not training:
            # In soft inference mode, use less aggressive penalty
            mask = mask.masked_fill(
                mask == self.soft_mask_value,
                self.soft_inference_value
            )
        
        # Add mask to logits
        masked_logits = logits + mask
        
        return masked_logits
    
    def decode(
        self,
        logits: torch.Tensor,
        is_mercosul: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode logits to character indices with hard constraints.
        
        Args:
            logits: Raw logits of shape (B, plate_length, vocab_size).
            is_mercosul: Binary tensor of shape (B,) or (B, 1).
            
        Returns:
            Character indices of shape (B, plate_length).
        """
        # Apply hard mask
        masked_logits = self.forward(logits, is_mercosul, training=False)
        
        # Greedy decoding
        indices = masked_logits.argmax(dim=-1)
        
        return indices
    
    def decode_to_text(
        self,
        logits: torch.Tensor,
        is_mercosul: torch.Tensor
    ) -> list:
        """
        Decode logits to text strings with hard constraints.
        
        Args:
            logits: Raw logits of shape (B, plate_length, vocab_size).
            is_mercosul: Binary tensor of shape (B,) or (B, 1).
            
        Returns:
            List of decoded plate strings.
        """
        indices = self.decode(logits, is_mercosul)
        
        texts = []
        for batch_idx in range(indices.size(0)):
            chars = []
            for pos_idx in range(indices.size(1)):
                char_idx = indices[batch_idx, pos_idx].item() - CHAR_START_IDX
                if 0 <= char_idx < len(CHARSET):
                    chars.append(CHARSET[char_idx])
                else:
                    chars.append('?')  # Unknown character
            texts.append(''.join(chars))
        
        return texts


class DifferentiableSyntaxMask(nn.Module):
    """
    Differentiable version of syntax mask using Gumbel-Softmax.
    
    This version allows gradients to flow through the layout decision,
    enabling joint training of the layout classifier and recognizer.
    """
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        plate_length: int = PLATE_LENGTH,
        char_start_idx: int = CHAR_START_IDX,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.plate_length = plate_length
        self.char_start_idx = char_start_idx
        self.temperature = temperature
        
        # Create mask layers for each position
        self.position_masks = nn.ModuleList()
        
        for pos in range(plate_length):
            # Position-specific mask selection layer
            self.position_masks.append(
                nn.Embedding(2, vocab_size)  # 2 options: letter or digit
            )
        
        # Initialize embeddings with mask values
        self._init_masks()
    
    def _init_masks(self):
        """Initialize mask embeddings."""
        letter_mask = create_letter_mask(self.vocab_size, self.char_start_idx)
        digit_mask = create_digit_mask(self.vocab_size, self.char_start_idx)
        
        # HARDCODED: Brazilian format constraints LLLNNNN
        # 0=letter, 1=digit → [L, L, L, N, N, N, N]
        brazilian_constraints = [0, 0, 0, 1, 1, 1, 1]
        
        # HARDCODED: Mercosul format constraints LLLNLNN
        # 0=letter, 1=digit → [L, L, L, N, L, N, N]
        mercosul_constraints = [0, 0, 0, 1, 0, 1, 1]
        
        for pos, mask_layer in enumerate(self.position_masks):
            # Set letter mask (index 0)
            mask_layer.weight.data[0] = letter_mask
            # Set digit mask (index 1)
            mask_layer.weight.data[1] = digit_mask
    
    def forward(
        self,
        logits: torch.Tensor,
        layout_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply differentiable syntax mask.
        
        Args:
            logits: Raw logits of shape (B, plate_length, vocab_size).
            layout_logits: Layout classification logits (B, 1).
            
        Returns:
            Masked logits of shape (B, plate_length, vocab_size).
        """
        batch_size = logits.size(0)
        
        # Get layout probabilities
        layout_prob = torch.sigmoid(layout_logits)  # P(mercosul)
        
        # HARDCODED: Brazilian format constraints LLLNNNN
        # 0=letter, 1=digit → [L, L, L, N, N, N, N]
        brazilian_constraints = torch.tensor([0, 0, 0, 1, 1, 1, 1], device=logits.device)
        
        # HARDCODED: Mercosul format constraints LLLNLNN
        # 0=letter, 1=digit → [L, L, L, N, L, N, N]
        mercosul_constraints = torch.tensor([0, 0, 0, 1, 0, 1, 1], device=logits.device)
        
        # Interpolate constraints based on layout probability
        # This allows gradients to flow through
        masked_logits = logits.clone()
        
        for pos in range(self.plate_length):
            # Get mask for this position based on interpolated constraint
            brazilian_mask_idx = brazilian_constraints[pos]
            mercosul_mask_idx = mercosul_constraints[pos]
            
            brazilian_mask = self.position_masks[pos].weight[brazilian_mask_idx]
            mercosul_mask = self.position_masks[pos].weight[mercosul_mask_idx]
            
            # Soft interpolation
            pos_mask = (
                (1 - layout_prob) * brazilian_mask.unsqueeze(0) +
                layout_prob * mercosul_mask.unsqueeze(0)
            )
            
            masked_logits[:, pos, :] = logits[:, pos, :] + pos_mask.squeeze(1)
        
        return masked_logits


def validate_plate_output(
    predicted_text: str,
    is_mercosul: bool
) -> bool:
    """
    Validate that predicted text matches expected format.
    
    Args:
        predicted_text: Predicted license plate text.
        is_mercosul: Whether the plate is Mercosul format.
        
    Returns:
        True if the text is valid for the given format.
    """
    if len(predicted_text) != PLATE_LENGTH:
        return False
    
    constraints = get_position_constraints(is_mercosul)
    
    for pos, (char, constraint) in enumerate(zip(predicted_text, constraints)):
        if constraint == 'L' and char not in LETTERS:
            return False
        if constraint == 'N' and char not in DIGITS:
            return False
    
    return True
