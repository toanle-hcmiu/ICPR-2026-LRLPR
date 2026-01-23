"""
Layout and Character Oriented Focal Loss (LCOFL) for License Plate Super-Resolution.

Implements the loss function from "Enhancing License Plate Super-Resolution:
A Layout-Aware and Character-Driven Approach" (Nascimento et al.).

The LCOFL loss combines four key components:
1. Classification Loss (LC): Weighted cross-entropy with dynamic character confusion penalties
2. Layout Penalty (LP): Penalizes digit/letter misplacements based on plate format
3. Dissimilarity Loss (LS): SSIM-based structural similarity loss
4. Confusion Matrix Tracking: Updates weights based on frequently confused character pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np

from config import (
    CHARSET, LETTERS, DIGITS, PLATE_LENGTH, VOCAB_SIZE,
    CHAR_START_IDX, get_position_constraints
)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) based loss.
    
    Computes (1 - SSIM) / 2 to normalize to [0, 1] range where:
    - 0 means highly similar images
    - 1 means highly dissimilar images
    """
    
    def __init__(
        self,
        window_size: int = 11,
        channel: int = 3,
        size_average: bool = True
    ):
        """
        Initialize SSIM loss.
        
        Args:
            window_size: Size of the Gaussian window.
            channel: Number of image channels.
            size_average: Whether to average the loss.
        """
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        
        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, channel))
    
    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian distribution."""
        gauss = torch.tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create 2D Gaussian window."""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        window_size: int,
        channel: int,
        size_average: bool = True
    ) -> torch.Tensor:
        """Compute SSIM between two images."""
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss.
        
        Args:
            img1: First image (B, C, H, W).
            img2: Second image (B, C, H, W).
            
        Returns:
            SSIM loss value in [0, 1] range.
        """
        # Ensure window is on the same device
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        channel = img1.size(1)
        
        if channel != self.channel:
            window = self._create_window(self.window_size, channel).to(img1.device)
        else:
            window = self.window
        
        ssim_value = self._ssim(img1, img2, window, self.window_size, channel, self.size_average)
        
        # Transform SSIM to loss: (1 - SSIM) / 2
        # SSIM range: [-1, 1] -> Loss range: [0, 1]
        loss = (1 - ssim_value) / 2
        
        return loss


class ConfusionMatrixTracker:
    """
    Tracks character confusion between predictions and ground truth.
    
    After each validation epoch, identifies frequently confused character pairs
    and updates penalty weights for the classification loss.
    """
    
    def __init__(self, charset_size: int = len(CHARSET), alpha: float = 1.0):
        """
        Initialize confusion matrix tracker.
        
        Args:
            charset_size: Number of characters in the charset.
            alpha: Penalty increment for confused character pairs.
        """
        self.charset_size = charset_size
        self.alpha = alpha
        
        # Confusion matrix: rows = GT, cols = predictions
        self.confusion_matrix = np.zeros((charset_size, charset_size), dtype=np.float32)
        
        # Character weights for classification loss
        self.char_weights = np.ones(charset_size, dtype=np.float32)
        
        # Common confused pairs to watch for
        self.known_confused_pairs = [
            ('B', '8'), ('8', 'B'),
            ('G', '6'), ('6', 'G'),
            ('O', '0'), ('0', 'O'),
            ('I', '1'), ('1', 'I'),
            ('S', '5'), ('5', 'S'),
            ('Z', '2'), ('2', 'Z'),
            ('T', '7'), ('7', 'T'),
            ('D', '0'), ('0', 'D'),
            ('Q', '0'), ('0', 'Q'),
        ]
    
    def reset(self):
        """Reset confusion matrix for new epoch."""
        self.confusion_matrix.fill(0)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion matrix with batch predictions.
        
        Args:
            predictions: Predicted character indices (B, L) where L is plate length.
            targets: Ground truth character indices (B, L).
        """
        # Convert to numpy
        preds = predictions.detach().cpu().numpy().flatten()
        tgts = targets.detach().cpu().numpy().flatten()
        
        # Update confusion matrix
        for pred, gt in zip(preds, tgts):
            # Convert from vocab index to charset index
            pred_idx = pred - CHAR_START_IDX
            gt_idx = gt - CHAR_START_IDX
            
            if 0 <= pred_idx < self.charset_size and 0 <= gt_idx < self.charset_size:
                self.confusion_matrix[gt_idx, pred_idx] += 1
    
    def get_confused_pairs(self, threshold: float = 0.1) -> List[Tuple[str, str, float]]:
        """
        Get frequently confused character pairs.
        
        Args:
            threshold: Minimum confusion ratio to report.
            
        Returns:
            List of (char1, char2, confusion_rate) tuples.
        """
        confused_pairs = []
        
        for gt_idx in range(self.charset_size):
            total = self.confusion_matrix[gt_idx].sum()
            if total == 0:
                continue
            
            for pred_idx in range(self.charset_size):
                if gt_idx == pred_idx:
                    continue
                
                confusion_rate = self.confusion_matrix[gt_idx, pred_idx] / total
                if confusion_rate >= threshold:
                    gt_char = CHARSET[gt_idx]
                    pred_char = CHARSET[pred_idx]
                    confused_pairs.append((gt_char, pred_char, confusion_rate))
        
        # Sort by confusion rate
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        return confused_pairs
    
    def update_weights(self, threshold: float = 0.05):
        """
        Update character weights based on confusion matrix.
        
        Increases weight for characters that are frequently confused.
        
        Args:
            threshold: Minimum confusion ratio to trigger weight increase.
        """
        for gt_idx in range(self.charset_size):
            total = self.confusion_matrix[gt_idx].sum()
            if total == 0:
                continue
            
            # Check for confusions with other characters
            for pred_idx in range(self.charset_size):
                if gt_idx == pred_idx:
                    continue
                
                confusion_rate = self.confusion_matrix[gt_idx, pred_idx] / total
                if confusion_rate >= threshold:
                    # Increase weight for both the GT and confused character
                    self.char_weights[gt_idx] += self.alpha * confusion_rate
                    self.char_weights[pred_idx] += self.alpha * confusion_rate * 0.5
        
        # Normalize weights to prevent explosion
        self.char_weights = np.clip(self.char_weights, 1.0, 10.0)
    
    def get_weight_tensor(self, device: torch.device) -> torch.Tensor:
        """Get character weights as a tensor."""
        return torch.from_numpy(self.char_weights).to(device)
    
    def get_confusion_matrix_image(self) -> Optional[torch.Tensor]:
        """
        Generate a confusion matrix heatmap image for TensorBoard visualization.
        
        Returns:
            Tensor of shape (3, H, W) suitable for TensorBoard add_image, or None if empty.
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Check if we have any data
        if self.confusion_matrix.sum() == 0:
            return None
        
        # Normalize each row to show confusion rates (not counts)
        row_sums = self.confusion_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized = self.confusion_matrix / row_sums
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        im = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Confusion Rate', rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(self.charset_size))
        ax.set_yticks(np.arange(self.charset_size))
        ax.set_xticklabels(list(CHARSET), fontsize=8)
        ax.set_yticklabels(list(CHARSET), fontsize=8)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Labels
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
        ax.set_title('Character Confusion Matrix (Row-Normalized)')
        
        # Highlight off-diagonal confusions > 5%
        for i in range(self.charset_size):
            for j in range(self.charset_size):
                if i != j and normalized[i, j] > 0.05:
                    text = ax.text(j, i, f'{normalized[i, j]:.0%}',
                                   ha="center", va="center", color="red", fontsize=6)
        
        fig.tight_layout()
        
        # Convert to tensor
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        # Convert to CHW format for TensorBoard (0-1 range)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        return img_tensor


class LayoutPenalty(nn.Module):
    """
    Layout Penalty Loss for enforcing digit/letter constraints.
    
    Penalizes when a digit is predicted at a letter position or vice versa,
    based on the license plate format (Brazilian or Mercosul).
    """
    
    def __init__(self, beta: float = 2.0):
        """
        Initialize layout penalty.
        
        Args:
            beta: Penalty value for each layout violation.
        """
        super().__init__()
        self.beta = beta
        
        # Create letter and digit index sets
        self.letter_indices = torch.tensor([
            CHAR_START_IDX + CHARSET.index(c) for c in LETTERS if c in CHARSET
        ])
        self.digit_indices = torch.tensor([
            CHAR_START_IDX + CHARSET.index(c) for c in DIGITS if c in CHARSET
        ])
    
    def forward(
        self,
        logits: torch.Tensor,
        is_mercosul: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute layout penalty.
        
        Args:
            logits: Predicted logits (B, L, V) where L is plate length, V is vocab size.
            is_mercosul: Layout indicator (B,) - 0 for Brazilian, 1 for Mercosul.
            
        Returns:
            Layout penalty value.
        """
        B, L, V = logits.shape
        device = logits.device
        
        # Move indices to device
        letter_indices = self.letter_indices.to(device)
        digit_indices = self.digit_indices.to(device)
        
        # Get predicted classes
        predictions = logits.argmax(dim=-1)  # (B, L)
        
        total_penalty = torch.tensor(0.0, device=device)
        
        for b in range(B):
            is_merc = is_mercosul[b].item() > 0.5 if isinstance(is_mercosul[b], torch.Tensor) else is_mercosul[b] > 0.5
            constraints = get_position_constraints(is_merc)
            
            for pos, constraint in enumerate(constraints):
                pred = predictions[b, pos]
                
                if constraint == 'L':
                    # Position should be a letter, check if digit was predicted
                    is_digit = (pred.unsqueeze(0) == digit_indices).any()
                    if is_digit:
                        total_penalty = total_penalty + self.beta
                else:  # constraint == 'N'
                    # Position should be a digit, check if letter was predicted
                    is_letter = (pred.unsqueeze(0) == letter_indices).any()
                    if is_letter:
                        total_penalty = total_penalty + self.beta
        
        # Average over batch
        return total_penalty / B


class LCOFLLoss(nn.Module):
    """
    Layout and Character Oriented Focal Loss (LCOFL).
    
    Combines four loss components to enhance LP super-resolution:
    1. Classification Loss (LC): Weighted cross-entropy
    2. Layout Penalty (LP): Digit/letter position enforcement
    3. Dissimilarity Loss (LS): SSIM-based structural similarity
    4. Dynamic weight updates based on confusion matrix
    
    Reference: Nascimento et al., "Enhancing License Plate Super-Resolution:
               A Layout-Aware and Character-Driven Approach"
    """
    
    def __init__(
        self,
        weight_classification: float = 1.0,
        weight_layout: float = 1.0,
        weight_ssim: float = 0.3,
        beta: float = 2.0,
        alpha: float = 1.0,
        label_smoothing: float = 0.0
    ):
        """
        Initialize LCOFL loss.
        
        Args:
            weight_classification: Weight for classification loss (LC).
            weight_layout: Weight for layout penalty (LP).
            weight_ssim: Weight for SSIM dissimilarity loss (LS).
            beta: Layout violation penalty value.
            alpha: Penalty increment for confused characters.
            label_smoothing: Label smoothing for cross-entropy.
        """
        super().__init__()
        
        self.weight_classification = weight_classification
        self.weight_layout = weight_layout
        self.weight_ssim = weight_ssim
        
        # Component losses
        self.ssim_loss = SSIMLoss()
        self.layout_penalty = LayoutPenalty(beta=beta)
        
        # Cross-entropy with ignore index for padding
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
            reduction='none'  # We'll apply custom weights
        )
        
        # Confusion matrix tracker
        self.confusion_tracker = ConfusionMatrixTracker(alpha=alpha)
        
        # Character weights (will be updated based on confusion matrix)
        self.register_buffer(
            'char_weights',
            torch.ones(len(CHARSET))
        )
    
    def update_char_weights(self, device: torch.device):
        """Update character weights from confusion tracker."""
        self.char_weights = self.confusion_tracker.get_weight_tensor(device)
    
    def reset_confusion_matrix(self):
        """Reset confusion matrix for new epoch."""
        self.confusion_tracker.reset()
    
    def update_confusion_matrix(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update confusion matrix with batch results."""
        self.confusion_tracker.update(predictions, targets)
    
    def finalize_epoch_weights(self, threshold: float = 0.05):
        """
        Finalize epoch by updating weights based on confusion matrix.
        
        Call this at the end of each validation epoch.
        
        Args:
            threshold: Minimum confusion ratio to trigger weight increase.
        """
        self.confusion_tracker.update_weights(threshold)
    
    def get_confused_pairs(self, threshold: float = 0.1) -> List[Tuple[str, str, float]]:
        """Get frequently confused character pairs for logging."""
        return self.confusion_tracker.get_confused_pairs(threshold)
    
    def get_confusion_matrix_image(self) -> Optional[torch.Tensor]:
        """Get confusion matrix heatmap for TensorBoard visualization."""
        return self.confusion_tracker.get_confusion_matrix_image()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        is_mercosul: torch.Tensor,
        generated_hr: Optional[torch.Tensor] = None,
        gt_hr: Optional[torch.Tensor] = None,
        compute_ssim: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute LCOFL loss.
        
        Args:
            logits: Predicted logits (B, L, V) where L is plate length, V is vocab size.
            targets: Ground truth character indices (B, L).
            is_mercosul: Layout indicator (B,) - 0 for Brazilian, 1 for Mercosul.
            generated_hr: Generated HR image (B, C, H, W) for SSIM loss.
            gt_hr: Ground truth HR image (B, C, H, W) for SSIM loss.
            compute_ssim: Whether to compute SSIM loss.
            
        Returns:
            Tuple of (total_loss, loss_dict).
        """
        B, L, V = logits.shape
        device = logits.device
        
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # Ensure char_weights is on correct device
        if self.char_weights.device != device:
            self.char_weights = self.char_weights.to(device)
        
        # 1. Classification Loss (LC) with dynamic weights
        # Reshape for cross-entropy: (B*L, V) and (B*L,)
        logits_flat = logits.view(-1, V)
        targets_flat = targets.reshape(-1)
        
        # Compute per-element cross-entropy
        ce_per_element = self.ce_loss(logits_flat, targets_flat)  # (B*L,)
        
        # Apply character-specific weights
        weights = torch.ones_like(ce_per_element)
        for i, target in enumerate(targets_flat):
            char_idx = target.item() - CHAR_START_IDX
            if 0 <= char_idx < len(self.char_weights):
                weights[i] = self.char_weights[char_idx]
        
        # Weighted mean
        lc = (ce_per_element * weights).mean()
        loss_dict['classification'] = lc.item()
        total_loss = total_loss + self.weight_classification * lc
        
        # 2. Layout Penalty (LP)
        lp = self.layout_penalty(logits, is_mercosul)
        loss_dict['layout_penalty'] = lp.item()
        total_loss = total_loss + self.weight_layout * lp
        
        # 3. SSIM Dissimilarity Loss (LS)
        if compute_ssim and generated_hr is not None and gt_hr is not None:
            ls = self.ssim_loss(generated_hr, gt_hr)
            loss_dict['ssim'] = ls.item()
            total_loss = total_loss + self.weight_ssim * ls
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


class LCOFLWithOCR(nn.Module):
    """
    LCOFL loss integrated with OCR-based training.
    
    This version is designed for GAN-style training where an OCR model
    acts as part of the discriminator, providing recognition-based feedback.
    """
    
    def __init__(
        self,
        ocr_model: Optional[nn.Module] = None,
        freeze_ocr: bool = True,
        **lcofl_kwargs
    ):
        """
        Initialize LCOFL with OCR.
        
        Args:
            ocr_model: Pre-trained OCR model for generating predictions.
            freeze_ocr: Whether to freeze OCR model weights.
            **lcofl_kwargs: Arguments for LCOFLLoss.
        """
        super().__init__()
        
        self.lcofl = LCOFLLoss(**lcofl_kwargs)
        self.ocr_model = ocr_model
        
        if ocr_model is not None and freeze_ocr:
            for param in ocr_model.parameters():
                param.requires_grad = False
            ocr_model.eval()
    
    def forward(
        self,
        generated_hr: torch.Tensor,
        gt_hr: torch.Tensor,
        targets: torch.Tensor,
        is_mercosul: torch.Tensor,
        logits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute LCOFL loss with optional OCR inference.
        
        Args:
            generated_hr: Generated HR image (B, C, H, W).
            gt_hr: Ground truth HR image (B, C, H, W).
            targets: Ground truth character indices (B, L).
            is_mercosul: Layout indicator (B,).
            logits: Pre-computed logits (B, L, V). If None, OCR model is used.
            
        Returns:
            Tuple of (total_loss, loss_dict).
        """
        if logits is None and self.ocr_model is not None:
            # Get predictions from OCR model
            with torch.set_grad_enabled(True):
                logits = self.ocr_model(generated_hr)
        
        if logits is None:
            raise ValueError("Either logits or ocr_model must be provided")
        
        return self.lcofl(
            logits=logits,
            targets=targets,
            is_mercosul=is_mercosul,
            generated_hr=generated_hr,
            gt_hr=gt_hr,
            compute_ssim=True
        )
