"""
Visualization Utilities for Neuro-Symbolic LPR System.

This module provides visualization functions for debugging and
demonstrating the system's components.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import torch


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize tensor from [-1, 1] to [0, 1].
    
    Args:
        tensor: Input tensor.
        
    Returns:
        Denormalized numpy array.
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    return (tensor + 1) / 2


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to displayable image.
    
    Args:
        tensor: Image tensor of shape (C, H, W) or (H, W, C).
        
    Returns:
        Image as numpy array (H, W, C) in [0, 1].
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    if tensor.shape[0] in [1, 3]:  # CHW format
        tensor = np.transpose(tensor, (1, 2, 0))
    
    tensor = denormalize(tensor)
    
    if tensor.shape[-1] == 1:
        tensor = np.repeat(tensor, 3, axis=-1)
    
    return np.clip(tensor, 0, 1)


def visualize_stn(
    original: torch.Tensor,
    rectified: torch.Tensor,
    corners_pred: Optional[torch.Tensor] = None,
    corners_gt: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
):
    """
    Visualize STN rectification results.
    
    Args:
        original: Original image tensor (C, H, W).
        rectified: Rectified image tensor (C, H, W).
        corners_pred: Predicted corners (4, 2) normalized to [-1, 1].
        corners_gt: Ground-truth corners (4, 2) normalized to [-1, 1].
        save_path: Path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(tensor_to_image(original))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Draw corners on original
    if corners_pred is not None or corners_gt is not None:
        h, w = original.shape[1:] if original.shape[0] == 3 else original.shape[:2]
        
        if corners_gt is not None:
            gt = corners_gt.cpu().numpy()
            gt_pixel = (gt + 1) / 2 * np.array([w, h])
            gt_pixel = np.vstack([gt_pixel, gt_pixel[0]])  # Close polygon
            axes[0].plot(gt_pixel[:, 0], gt_pixel[:, 1], 'g-', linewidth=2, label='GT')
            axes[0].scatter(gt_pixel[:-1, 0], gt_pixel[:-1, 1], c='g', s=50)
        
        if corners_pred is not None:
            pred = corners_pred.cpu().numpy()
            pred_pixel = (pred + 1) / 2 * np.array([w, h])
            pred_pixel = np.vstack([pred_pixel, pred_pixel[0]])
            axes[0].plot(pred_pixel[:, 0], pred_pixel[:, 1], 'r--', linewidth=2, label='Pred')
            axes[0].scatter(pred_pixel[:-1, 0], pred_pixel[:-1, 1], c='r', s=50)
        
        axes[0].legend()
    
    # Rectified image
    axes[1].imshow(tensor_to_image(rectified))
    axes[1].set_title('Rectified')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_sr(
    lr_image: torch.Tensor,
    sr_image: torch.Tensor,
    hr_image: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
):
    """
    Visualize super-resolution results.
    
    Args:
        lr_image: Low-resolution image tensor.
        sr_image: Super-resolved image tensor.
        hr_image: Ground-truth high-resolution image (optional).
        save_path: Path to save the figure.
    """
    num_cols = 3 if hr_image is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    
    # LR image
    lr_display = tensor_to_image(lr_image)
    axes[0].imshow(lr_display)
    axes[0].set_title('Low Resolution')
    axes[0].axis('off')
    
    # SR image
    sr_display = tensor_to_image(sr_image)
    axes[1].imshow(sr_display)
    axes[1].set_title('Super-Resolved')
    axes[1].axis('off')
    
    # HR image (if available)
    if hr_image is not None:
        hr_display = tensor_to_image(hr_image)
        axes[2].imshow(hr_display)
        axes[2].set_title('Ground Truth HR')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_predictions(
    image: torch.Tensor,
    predicted_text: str,
    ground_truth: Optional[str] = None,
    is_mercosul: bool = False,
    confidence: float = 1.0,
    save_path: Optional[str] = None
):
    """
    Visualize recognition predictions.
    
    Args:
        image: Input image tensor.
        predicted_text: Predicted license plate text.
        ground_truth: Ground-truth text (optional).
        is_mercosul: Whether plate is Mercosul format.
        confidence: Prediction confidence.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    ax.imshow(tensor_to_image(image))
    ax.axis('off')
    
    # Create title
    layout = "Mercosul" if is_mercosul else "Brazilian"
    title = f"Predicted: {predicted_text} ({layout})\nConfidence: {confidence:.2%}"
    
    if ground_truth is not None:
        correct = predicted_text == ground_truth
        color = 'green' if correct else 'red'
        title += f"\nGround Truth: {ground_truth} ({'✓' if correct else '✗'})"
    else:
        color = 'black'
    
    ax.set_title(title, fontsize=14, color=color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_multi_frame(
    frames: torch.Tensor,
    rectified_frames: Optional[torch.Tensor] = None,
    quality_scores: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None
):
    """
    Visualize multi-frame input and processing.
    
    Args:
        frames: Input frames tensor (T, C, H, W) or (B, T, C, H, W).
        rectified_frames: Rectified frames (optional).
        quality_scores: Quality scores per frame (optional).
        save_path: Path to save the figure.
    """
    if frames.dim() == 5:
        frames = frames[0]  # Take first batch
    if rectified_frames is not None and rectified_frames.dim() == 5:
        rectified_frames = rectified_frames[0]
    
    num_frames = frames.shape[0]
    num_rows = 2 if rectified_frames is not None else 1
    
    fig, axes = plt.subplots(num_rows, num_frames, figsize=(3 * num_frames, 3 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_frames):
        # Original frame
        axes[0, i].imshow(tensor_to_image(frames[i]))
        title = f'Frame {i}'
        if quality_scores is not None:
            title += f'\nQ: {quality_scores[i].item():.2f}'
        axes[0, i].set_title(title)
        axes[0, i].axis('off')
        
        # Rectified frame
        if rectified_frames is not None:
            axes[1, i].imshow(tensor_to_image(rectified_frames[i]))
            axes[1, i].set_title(f'Rectified {i}')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_attention(
    image: torch.Tensor,
    attention_map: torch.Tensor,
    save_path: Optional[str] = None
):
    """
    Visualize attention maps overlaid on image.
    
    Args:
        image: Input image tensor (C, H, W).
        attention_map: Attention weights (H', W') or (num_heads, H', W').
        save_path: Path to save the figure.
    """
    import matplotlib.cm as cm
    
    if attention_map.dim() == 3:
        # Average over heads
        attention_map = attention_map.mean(dim=0)
    
    # Resize attention to image size
    h, w = image.shape[1:]
    attention = torch.nn.functional.interpolate(
        attention_map.unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
    
    # Normalize
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img = tensor_to_image(image)
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Attention map
    axes[1].imshow(attention, cmap='hot')
    axes[1].set_title('Attention')
    axes[1].axis('off')
    
    # Overlay
    heatmap = cm.jet(attention)[:, :, :3]
    overlay = 0.6 * img + 0.4 * heatmap
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_comparison_grid(
    images: List[torch.Tensor],
    titles: List[str],
    ncols: int = 4,
    save_path: Optional[str] = None
):
    """
    Create a grid of comparison images.
    
    Args:
        images: List of image tensors.
        titles: List of titles for each image.
        ncols: Number of columns.
        save_path: Path to save the figure.
    """
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(tensor_to_image(img))
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
