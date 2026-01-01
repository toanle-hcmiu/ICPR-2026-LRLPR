"""
Dataset Classes for License Plate Recognition.

This module implements dataset classes for loading and processing
license plate images for training the Neuro-Symbolic LPR system.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHARSET, CHAR_START_IDX, PLATE_LENGTH, PAD_IDX, BOS_IDX, EOS_IDX,
    infer_layout_from_text, VOCABULARY
)


def text_to_indices(text: str, max_length: int = PLATE_LENGTH) -> torch.Tensor:
    """
    Convert plate text to token indices.
    
    Args:
        text: License plate text (without dash).
        max_length: Maximum sequence length.
        
    Returns:
        Tensor of token indices.
    """
    text = text.upper().replace('-', '').replace(' ', '')
    
    indices = []
    for char in text[:max_length]:
        if char in CHARSET:
            idx = CHAR_START_IDX + CHARSET.index(char)
            indices.append(idx)
        else:
            indices.append(PAD_IDX)  # Unknown character
    
    # Pad if necessary
    while len(indices) < max_length:
        indices.append(PAD_IDX)
    
    return torch.tensor(indices, dtype=torch.long)


def indices_to_text(indices: torch.Tensor) -> str:
    """
    Convert token indices back to text.
    
    Args:
        indices: Tensor of token indices.
        
    Returns:
        Decoded text string.
    """
    chars = []
    for idx in indices:
        idx = idx.item()
        if idx >= CHAR_START_IDX and idx < CHAR_START_IDX + len(CHARSET):
            chars.append(CHARSET[idx - CHAR_START_IDX])
        elif idx == EOS_IDX:
            break
    return ''.join(chars)


class LPRDataset(Dataset):
    """
    Base dataset class for license plate recognition.
    
    Supports loading multi-frame sequences with corresponding
    annotations including text, corners, and layout.
    """
    
    def __init__(
        self,
        data_dir: str,
        num_frames: int = 5,
        lr_size: Tuple[int, int] = (16, 48),
        hr_size: Tuple[int, int] = (64, 192),
        transform: Optional[Callable] = None,
        return_path: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the data.
            num_frames: Number of frames per sample.
            lr_size: Low-resolution image size (H, W).
            hr_size: High-resolution image size (H, W).
            transform: Optional transform to apply.
            return_path: Whether to return file paths.
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.transform = transform
        self.return_path = return_path
        
        # Load annotations
        self.samples = self._load_annotations()
    
    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations. Override in subclasses."""
        raise NotImplementedError
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load an image from path."""
        img = Image.open(path).convert('RGB')
        return np.array(img)
    
    def _preprocess_image(
        self, 
        image: np.ndarray, 
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Preprocess image to target size and normalize.
        
        Args:
            image: Input image as numpy array.
            target_size: Target size (H, W).
            
        Returns:
            Preprocessed image tensor.
        """
        # Resize
        img = Image.fromarray(image)
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        
        # Convert to tensor and normalize to [-1, 1]
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = img_array * 2 - 1
        
        # Convert to CHW format
        if img_array.ndim == 3:
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        else:
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        return img_tensor
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class RodoSolDataset(LPRDataset):
    """
    Dataset class for RodoSol-ALPR dataset.
    
    The RodoSol-ALPR dataset contains 20,000 images of Brazilian
    license plates captured from real-world toll cameras.
    
    Expected directory structure:
        data_dir/
            images/
                img_0001.jpg
                img_0002.jpg
                ...
            annotations.json
    
    Annotations format:
        {
            "img_0001.jpg": {
                "text": "ABC1234",
                "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                "layout": "brazilian"  # or "mercosul"
            },
            ...
        }
    """
    
    def _load_annotations(self) -> List[Dict]:
        """Load RodoSol annotations."""
        samples = []
        
        # Try to load annotations file
        ann_file = self.data_dir / 'annotations.json'
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                annotations = json.load(f)
            
            for img_name, ann in annotations.items():
                samples.append({
                    'image_path': str(self.data_dir / 'images' / img_name),
                    'text': ann['text'],
                    'corners': ann.get('corners', None),
                    'layout': ann.get('layout', None)
                })
        else:
            # Fallback: scan directory for images
            images_dir = self.data_dir / 'images'
            if not images_dir.exists():
                images_dir = self.data_dir
            
            for img_path in images_dir.glob('*.jpg'):
                # Try to extract text from filename
                # Assume format: plate_ABC1234.jpg or ABC1234.jpg
                name = img_path.stem
                text = name.split('_')[-1] if '_' in name else name
                
                samples.append({
                    'image_path': str(img_path),
                    'text': text,
                    'corners': None,
                    'layout': None
                })
        
        return samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing:
                - 'lr_frames': LR frames tensor (num_frames, C, H, W)
                - 'hr_image': HR image tensor (C, H, W)
                - 'text_indices': Character indices (PLATE_LENGTH,)
                - 'text': Original text string
                - 'layout': Layout label (0=Brazilian, 1=Mercosul)
                - 'corners': Corner coordinates (4, 2) if available
        """
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image_path'])
        
        # Apply transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        
        # Create HR image
        hr_image = self._preprocess_image(image, self.hr_size)
        
        # Create LR frames (simulate multiple frames from single image)
        lr_frames = []
        for _ in range(self.num_frames):
            # Add slight variations for multi-frame simulation
            if self.num_frames > 1:
                # Small random shift
                h, w = image.shape[:2]
                shift_x = random.randint(-2, 2)
                shift_y = random.randint(-2, 2)
                
                shifted = np.roll(np.roll(image, shift_x, axis=1), shift_y, axis=0)
                lr_frame = self._preprocess_image(shifted, self.lr_size)
            else:
                lr_frame = self._preprocess_image(image, self.lr_size)
            
            lr_frames.append(lr_frame)
        
        lr_frames = torch.stack(lr_frames, dim=0)  # (T, C, H, W)
        
        # Process text
        text = sample['text']
        text_indices = text_to_indices(text)
        
        # Determine layout
        if sample['layout'] is not None:
            layout = 1 if sample['layout'] == 'mercosul' else 0
        else:
            layout = infer_layout_from_text(text)
            layout = max(0, layout)  # Handle invalid texts
        
        result = {
            'lr_frames': lr_frames,
            'hr_image': hr_image,
            'text_indices': text_indices,
            'text': text,
            'layout': torch.tensor(layout, dtype=torch.long)
        }
        
        # Add corners if available
        if sample['corners'] is not None:
            corners = torch.tensor(sample['corners'], dtype=torch.float32)
            # Normalize corners to [-1, 1]
            h, w = image.shape[:2]
            corners[:, 0] = corners[:, 0] / w * 2 - 1
            corners[:, 1] = corners[:, 1] / h * 2 - 1
            result['corners'] = corners
        
        if self.return_path:
            result['path'] = sample['image_path']
        
        return result


class SyntheticLPRDataset(Dataset):
    """
    Synthetic license plate dataset for pre-training.
    
    Generates synthetic license plate images on-the-fly with
    random text, fonts, and degradations.
    """
    
    def __init__(
        self,
        num_samples: int = 100000,
        lr_size: Tuple[int, int] = (16, 48),
        hr_size: Tuple[int, int] = (64, 192),
        num_frames: int = 5,
        fonts: List[str] = None,
        include_mercosul: bool = True
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            num_samples: Number of samples to generate.
            lr_size: Low-resolution size.
            hr_size: High-resolution size.
            num_frames: Number of frames per sample.
            fonts: List of font names to use.
            include_mercosul: Whether to include Mercosul plates.
        """
        super().__init__()
        
        self.num_samples = num_samples
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.num_frames = num_frames
        self.fonts = fonts or ['arial']
        self.include_mercosul = include_mercosul
        
        # Pre-generate plate texts for consistency
        self.plate_texts = self._generate_plate_texts()
    
    def _generate_plate_texts(self) -> List[Tuple[str, int]]:
        """Generate random plate texts with layouts."""
        texts = []
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        digits = '0123456789'
        
        for _ in range(self.num_samples):
            if self.include_mercosul and random.random() > 0.5:
                # Mercosul format: LLLNLNN
                text = (
                    random.choice(letters) +
                    random.choice(letters) +
                    random.choice(letters) +
                    random.choice(digits) +
                    random.choice(letters) +
                    random.choice(digits) +
                    random.choice(digits)
                )
                layout = 1
            else:
                # Brazilian format: LLLNNNN
                text = (
                    random.choice(letters) +
                    random.choice(letters) +
                    random.choice(letters) +
                    random.choice(digits) +
                    random.choice(digits) +
                    random.choice(digits) +
                    random.choice(digits)
                )
                layout = 0
            
            texts.append((text, layout))
        
        return texts
    
    def _render_plate(
        self, 
        text: str, 
        layout: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render a synthetic license plate image.
        
        Args:
            text: Plate text.
            layout: Layout type (0=Brazilian, 1=Mercosul).
            
        Returns:
            Tuple of (hr_image, corners).
        """
        # Create a simple synthetic plate
        # This is a basic implementation - can be enhanced with proper fonts
        
        from PIL import Image, ImageDraw, ImageFont
        
        # Plate dimensions
        width, height = self.hr_size[1], self.hr_size[0]
        
        # Create plate background
        if layout == 1:  # Mercosul
            # White background with blue band
            img = Image.new('RGB', (width, height), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # Blue band at top
            band_height = height // 6
            draw.rectangle([0, 0, width, band_height], fill=(0, 51, 102))
        else:
            # Grey background
            img = Image.new('RGB', (width, height), color=(192, 192, 192))
            draw = ImageDraw.Draw(img)
        
        # Try to use a font, fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", height // 2)
        except:
            font = ImageFont.load_default()
        
        # Draw text
        text_color = (0, 0, 0)
        
        # Calculate text position (centered)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            text_width = len(text) * (height // 4)
            text_height = height // 2
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), text, fill=text_color, font=font)
        
        # Convert to numpy
        hr_image = np.array(img)
        
        # Corners (normalized to [-1, 1])
        corners = np.array([
            [-1, -1],  # top-left
            [1, -1],   # top-right
            [1, 1],    # bottom-right
            [-1, 1]    # bottom-left
        ], dtype=np.float32)
        
        return hr_image, corners
    
    def _apply_degradation(self, image: np.ndarray) -> np.ndarray:
        """Apply random degradations to simulate LR image."""
        from PIL import Image, ImageFilter
        
        img = Image.fromarray(image)
        
        # Random blur
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        
        # Random noise
        if random.random() > 0.5:
            img_array = np.array(img).astype(np.float32)
            noise = np.random.randn(*img_array.shape) * random.uniform(5, 20)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        return np.array(img)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a synthetic sample."""
        text, layout = self.plate_texts[idx]
        
        # Render plate
        hr_image, corners = self._render_plate(text, layout)
        
        # Apply degradations and create LR frames
        lr_frames = []
        for _ in range(self.num_frames):
            degraded = self._apply_degradation(hr_image)
            
            # Resize to LR
            lr_img = Image.fromarray(degraded)
            lr_img = lr_img.resize((self.lr_size[1], self.lr_size[0]), Image.BILINEAR)
            lr_array = np.array(lr_img).astype(np.float32) / 255.0 * 2 - 1
            lr_tensor = torch.from_numpy(lr_array).permute(2, 0, 1)
            lr_frames.append(lr_tensor)
        
        lr_frames = torch.stack(lr_frames, dim=0)
        
        # Process HR image
        hr_array = hr_image.astype(np.float32) / 255.0 * 2 - 1
        hr_tensor = torch.from_numpy(hr_array).permute(2, 0, 1)
        
        return {
            'lr_frames': lr_frames,
            'hr_image': hr_tensor,
            'text_indices': text_to_indices(text),
            'text': text,
            'layout': torch.tensor(layout, dtype=torch.long),
            'corners': torch.from_numpy(corners)
        }


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    num_frames: int = 5
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_dir: Training data directory.
        val_dir: Validation data directory.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        num_frames: Number of frames per sample.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = RodoSolDataset(train_dir, num_frames=num_frames)
    val_dataset = RodoSolDataset(val_dir, num_frames=num_frames)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
