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


def text_to_indices(
    text: str, 
    max_length: int = PLATE_LENGTH,
    add_bos: bool = True,
    add_eos: bool = True
) -> torch.Tensor:
    """
    Convert plate text to token indices with BOS/EOS tokens.
    
    The output sequence format is: [BOS, char1, char2, ..., charN, EOS, PAD, PAD, ...]
    
    Args:
        text: License plate text (without dash).
        max_length: Maximum number of characters (not including BOS/EOS).
        add_bos: Whether to prepend BOS token.
        add_eos: Whether to append EOS token.
        
    Returns:
        Tensor of token indices with shape (max_length + 2,) if both BOS/EOS added.
    """
    text = text.upper().replace('-', '').replace(' ', '')
    
    indices = []
    
    # Add BOS token
    if add_bos:
        indices.append(BOS_IDX)
    
    # Add character tokens
    for char in text[:max_length]:
        if char in CHARSET:
            idx = CHAR_START_IDX + CHARSET.index(char)
            indices.append(idx)
        else:
            # Unknown character - skip or use special handling
            # For robustness, we skip unknown characters
            pass
    
    # Add EOS token
    if add_eos:
        indices.append(EOS_IDX)
    
    # Calculate total sequence length
    total_length = max_length
    if add_bos:
        total_length += 1
    if add_eos:
        total_length += 1
    
    # Pad to total length
    while len(indices) < total_length:
        indices.append(PAD_IDX)
    
    # Truncate if somehow longer (shouldn't happen with proper max_length)
    indices = indices[:total_length]
    
    return torch.tensor(indices, dtype=torch.long)


def indices_to_text(indices: torch.Tensor, skip_special_tokens: bool = True) -> str:
    """
    Convert token indices back to text.
    
    Args:
        indices: Tensor of token indices (may include BOS, EOS, PAD tokens).
        skip_special_tokens: Whether to skip BOS/EOS/PAD tokens in output.
        
    Returns:
        Decoded text string.
    """
    chars = []
    for idx in indices:
        idx = idx.item() if hasattr(idx, 'item') else idx
        
        if idx >= CHAR_START_IDX and idx < CHAR_START_IDX + len(CHARSET):
            chars.append(CHARSET[idx - CHAR_START_IDX])
        elif idx == EOS_IDX:
            if skip_special_tokens:
                break  # Stop at EOS
            else:
                chars.append('<EOS>')
        elif idx == BOS_IDX:
            if not skip_special_tokens:
                chars.append('<BOS>')
            # Otherwise skip BOS
        elif idx == PAD_IDX:
            if not skip_special_tokens:
                chars.append('<PAD>')
            # Otherwise skip PAD
    
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
    
    The RodoSol-ALPR dataset contains multi-frame sequences of Brazilian
    license plates captured from real-world toll cameras.
    
    Expected directory structure:
        data_dir/
            Scenario-A/
                Brazilian/
                    track_00001/
                        annotations.json
                        lr-001.jpg, lr-002.jpg, ...
                        hr-001.jpg, hr-002.jpg, ...
                    track_00002/
                        ...
                Mercosur/
                    track_00001/
                        ...
            Scenario-B/
                ...
    
    Annotations format:
        {
            "plate_layout": "Brazilian" or "Mercosur",
            "plate_text": "ABC1234",
            "corners": {
                "lr-001.png": {
                    "top-left": [x, y],
                    "top-right": [x, y],
                    "bottom-right": [x, y],
                    "bottom-left": [x, y]
                },
                ...
            }
        }
    """
    
    @staticmethod
    def _parse_frame_corners(corner_dict: Dict) -> Optional[np.ndarray]:
        """
        Parse corners from per-frame dictionary format.
        
        Args:
            corner_dict: Dictionary with 'top-left', 'top-right', 
                        'bottom-right', 'bottom-left' keys.
                        
        Returns:
            Corners as numpy array of shape (4, 2) in order:
            [top-left, top-right, bottom-right, bottom-left]
            or None if invalid.
        """
        if not isinstance(corner_dict, dict):
            return None
        
        required_keys = ['top-left', 'top-right', 'bottom-right', 'bottom-left']
        if not all(k in corner_dict for k in required_keys):
            return None
        
        try:
            corners = np.array([
                corner_dict['top-left'],
                corner_dict['top-right'],
                corner_dict['bottom-right'],
                corner_dict['bottom-left']
            ], dtype=np.float32)
            
            if corners.shape != (4, 2):
                return None
            return corners
        except (ValueError, TypeError):
            return None
    
    def _load_annotations(self) -> List[Dict]:
        """Load RodoSol annotations from track directories."""
        samples = []
        
        # Check if this is a track-based structure (Scenario-A/B, Brazilian/Mercosur)
        scenario_dirs = list(self.data_dir.glob('Scenario-*'))
        if scenario_dirs:
            # Track-based structure: scan all tracks
            for scenario_dir in scenario_dirs:
                for layout_dir in scenario_dir.iterdir():
                    if not layout_dir.is_dir():
                        continue
                    
                    # Find all track directories
                    for track_dir in layout_dir.iterdir():
                        if not track_dir.is_dir() or not track_dir.name.startswith('track_'):
                            continue
                        
                        ann_file = track_dir / 'annotations.json'
                        if not ann_file.exists():
                            continue
                        
                        try:
                            with open(ann_file, 'r') as f:
                                ann = json.load(f)
                            
                            # Get layout type
                            layout_str = ann.get('plate_layout', '').lower()
                            if 'mercosur' in layout_str or 'mercosul' in layout_str:
                                layout = 'mercosul'
                            else:
                                layout = 'brazilian'
                            
                            # Get plate text
                            text = ann.get('plate_text', '')
                            if not text:
                                continue
                            
                            # Get corners if available
                            corners = ann.get('corners', None)
                            if isinstance(corners, dict) and len(corners) == 0:
                                corners = None
                            
                            # Check for LR and HR frames
                            lr_files = sorted(track_dir.glob('lr-*.jpg')) + sorted(track_dir.glob('lr-*.png'))
                            hr_files = sorted(track_dir.glob('hr-*.jpg')) + sorted(track_dir.glob('hr-*.png'))
                            
                            if len(lr_files) > 0 and len(hr_files) > 0:
                                samples.append({
                                    'track_dir': str(track_dir),
                                    'lr_files': [str(f) for f in lr_files],
                                    'hr_files': [str(f) for f in hr_files],
                                    'text': text,
                                    'corners': corners,
                                    'layout': layout
                                })
                        except Exception as e:
                            # Skip tracks with invalid annotations
                            continue
        else:
            # Try to load single annotations file (old format)
            ann_file = self.data_dir / 'annotations.json'
            if ann_file.exists():
                with open(ann_file, 'r') as f:
                    annotations = json.load(f)
                
                for img_name, ann in annotations.items():
                    # Handle corners - convert empty dict to None
                    corners = ann.get('corners', None)
                    if isinstance(corners, dict) and len(corners) == 0:
                        corners = None
                    
                    samples.append({
                        'image_path': str(self.data_dir / 'images' / img_name),
                        'text': ann.get('text', ann.get('plate_text', '')),
                        'corners': corners,
                        'layout': ann.get('layout', None)
                    })
            else:
                # Fallback: scan directory for images
                images_dir = self.data_dir / 'images'
                if not images_dir.exists():
                    images_dir = self.data_dir
                
                for img_path in list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')):
                    # Try to extract text from filename
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
                - 'text_indices': Character indices (PLATE_LENGTH + 2,) with BOS/EOS
                - 'text': Original text string
                - 'layout': Layout label (0=Brazilian, 1=Mercosul), -1 for invalid
                - 'corners': Corner coordinates (4, 2) if available, transformed to match augmented image
                - 'plate_style': Plate style string ('brazilian' or 'mercosul') for style-aware processing
        """
        sample = self.samples[idx]
        
        # Determine layout early for style-aware processing
        # Note: layout = -1 indicates invalid plate format (neither Brazilian nor Mercosul)
        # This is preserved to allow filtering during collation or training
        if sample['layout'] is not None:
            layout = 1 if sample['layout'] == 'mercosul' else 0
            plate_style = sample['layout']  # 'brazilian' or 'mercosul'
        else:
            text = sample['text']
            layout = infer_layout_from_text(text)
            # IMPORTANT: Do NOT force invalid layout to 0
            # Invalid plates (layout=-1) should be:
            # 1. Filtered out during collation, OR
            # 2. Excluded from layout loss computation
            if layout == 1:
                plate_style = 'mercosul'
            elif layout == 0:
                plate_style = 'brazilian'
            else:
                # Invalid layout - default to brazilian for augmentation purposes
                # but preserve layout=-1 for proper handling
                plate_style = 'brazilian'
        
        # Store original image dimensions for corner transformation
        original_hr_size = None  # (width, height)
        hr_transform_matrix = None  # Transform matrix applied to HR image
        
        # Check if this is a track-based sample
        if 'track_dir' in sample:
            # Load multi-frame sequence from track
            lr_files = sample['lr_files']
            hr_files = sample['hr_files']
            
            # Select frames (use all available or sample if more than needed)
            num_available = min(len(lr_files), len(hr_files))
            if num_available >= self.num_frames:
                # Sample evenly spaced frames
                indices = np.linspace(0, num_available - 1, self.num_frames, dtype=int)
            else:
                # Repeat frames if not enough available
                indices = list(range(num_available)) * (self.num_frames // num_available + 1)
                indices = indices[:self.num_frames]
            
            # Load raw LR frames (no augmentation yet)
            lr_frames_raw = []
            for i in indices:
                lr_path = lr_files[min(i, len(lr_files) - 1)]
                lr_image = self._load_image(lr_path)
                lr_frames_raw.append(lr_image)
            
            # Load HR image (use middle frame or first available)
            hr_idx = len(hr_files) // 2 if hr_files else 0
            hr_path = hr_files[min(hr_idx, len(hr_files) - 1)]
            hr_image_raw = self._load_image(hr_path)
            
            # Store original HR image size before transform
            original_hr_size = (hr_image_raw.shape[1], hr_image_raw.shape[0])  # (width, height)
            
            # Get corners for this sample if available (for paired augmentation)
            corners_for_aug = self._extract_corners_for_sample(sample, hr_path)
            
            # Apply PAIRED augmentation (LR and HR stay aligned)
            if self.transform is not None and hasattr(self.transform, 'augment_pair'):
                # Set plate style for style-aware augmentation
                if hasattr(self.transform, 'set_plate_style'):
                    self.transform.set_plate_style(plate_style)
                
                # Apply paired augmentation: same geometric/photometric to all,
                # degradation only to LR frames
                lr_frames_aug, hr_image_aug, corners_aug, hr_transform_matrix = \
                    self.transform.augment_pair(lr_frames_raw, hr_image_raw, corners_for_aug)
                
                # Preprocess augmented images
                lr_frames = []
                for lr_img in lr_frames_aug:
                    lr_frame = self._preprocess_image(lr_img, self.lr_size)
                    lr_frames.append(lr_frame)
                lr_frames = torch.stack(lr_frames, dim=0)
                
                hr_image = self._preprocess_image(hr_image_aug, self.hr_size)
                hr_image_shape = hr_image_aug.shape[:2]  # (H, W)
                
                # Use augmented corners if available
                if corners_aug is not None:
                    corners_for_aug = corners_aug
            elif self.transform is not None:
                # Fallback: old behavior (independent augmentation - NOT RECOMMENDED)
                # This path is kept for backward compatibility but should be avoided
                lr_frames = []
                for lr_img in lr_frames_raw:
                    if hasattr(self.transform, 'set_plate_style'):
                        self.transform.set_plate_style(plate_style)
                    lr_img_aug = self.transform(lr_img)
                    lr_frame = self._preprocess_image(lr_img_aug, self.lr_size)
                    lr_frames.append(lr_frame)
                lr_frames = torch.stack(lr_frames, dim=0)
                
                if hasattr(self.transform, 'set_plate_style'):
                    self.transform.set_plate_style(plate_style)
                hr_image_aug = self.transform(hr_image_raw)
                if hasattr(self.transform, 'get_last_transform_matrix'):
                    hr_transform_matrix = self.transform.get_last_transform_matrix()
                hr_image = self._preprocess_image(hr_image_aug, self.hr_size)
                hr_image_shape = hr_image_aug.shape[:2]
            else:
                # No augmentation
                lr_frames = []
                for lr_img in lr_frames_raw:
                    lr_frame = self._preprocess_image(lr_img, self.lr_size)
                    lr_frames.append(lr_frame)
                lr_frames = torch.stack(lr_frames, dim=0)
                
                hr_image = self._preprocess_image(hr_image_raw, self.hr_size)
                hr_image_shape = hr_image_raw.shape[:2]
        else:
            # Single image sample (old format)
            image = self._load_image(sample['image_path'])
            
            # Store original image size before transform
            original_hr_size = (image.shape[1], image.shape[0])  # (width, height)
            
            # Get corners for this sample
            corners_for_aug = self._extract_corners_for_sample(sample, None)
            
            # Apply transforms if provided
            if self.transform is not None:
                # Pass plate style to transform for style-aware augmentation
                if hasattr(self.transform, 'set_plate_style'):
                    self.transform.set_plate_style(plate_style)
                image = self.transform(image)
                
                # Get the transform matrix for corner transformation
                if hasattr(self.transform, 'get_last_transform_matrix'):
                    hr_transform_matrix = self.transform.get_last_transform_matrix()
                    # Transform corners if we have them
                    if corners_for_aug is not None and hasattr(self.transform, 'transform_corners'):
                        corners_for_aug = self.transform.transform_corners(
                            corners_for_aug, original_hr_size
                        )
            
            # Create HR image
            hr_image = self._preprocess_image(image, self.hr_size)
            hr_image_shape = image.shape[:2]  # (H, W)
            
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
        
        result = {
            'lr_frames': lr_frames,
            'hr_image': hr_image,
            'text_indices': text_indices,
            'text': text,
            'layout': torch.tensor(layout, dtype=torch.long),
            'plate_style': plate_style  # Add plate style for potential style-aware processing
        }
        
        # Add corners if available (corners_for_aug was already extracted and transformed)
        if corners_for_aug is not None:
            corners_array = np.array(corners_for_aug, dtype=np.float32)
            if corners_array.shape == (4, 2):  # Valid corner format
                corners = torch.from_numpy(corners_array)
                # Normalize corners to [-1, 1] using transformed image dimensions
                h, w = hr_image_shape
                corners[:, 0] = corners[:, 0] / w * 2 - 1
                corners[:, 1] = corners[:, 1] / h * 2 - 1
                
                # Clamp to valid range (in case transform pushed corners outside)
                corners = torch.clamp(corners, -1.0, 1.0)
                result['corners'] = corners
        
        if self.return_path:
            if 'track_dir' in sample:
                result['path'] = sample['track_dir']
            else:
                result['path'] = sample.get('image_path', '')
        
        return result
    
    def _extract_corners_for_sample(
        self,
        sample: Dict,
        hr_path: Optional[str],
    ) -> Optional[np.ndarray]:
        """
        Extract corner coordinates from a sample's annotation.
        
        Args:
            sample: Sample dictionary from _load_annotations.
            hr_path: Path to the HR image being used (for per-frame corner lookup).
            
        Returns:
            Corners as numpy array of shape (4, 2), or None if not available.
        """
        corners_data = sample.get('corners', None)
        if corners_data is None:
            return None
        
        # Handle different corner formats
        if isinstance(corners_data, dict):
            # Empty dict - skip
            if len(corners_data) == 0:
                return None
            
            # Per-frame corner format: {"lr-001.png": {"top-left": [x, y], ...}, ...}
            if hr_path is not None:
                # Try to extract corners from the HR image we're using
                hr_filename = Path(hr_path).name
                if hr_filename in corners_data:
                    parsed = self._parse_frame_corners(corners_data[hr_filename])
                    if parsed is not None:
                        return parsed
                
                # Try to find any HR corner data
                for key in corners_data:
                    if key.startswith('hr-'):
                        parsed = self._parse_frame_corners(corners_data[key])
                        if parsed is not None:
                            return parsed
                
                # No HR corners found, try LR corners as fallback
                for key in corners_data:
                    parsed = self._parse_frame_corners(corners_data[key])
                    if parsed is not None:
                        return parsed
            else:
                # Single image format - try any available corners
                for key in corners_data:
                    parsed = self._parse_frame_corners(corners_data[key])
                    if parsed is not None:
                        return parsed
            
            return None
        
        # Direct array format
        if isinstance(corners_data, (list, tuple, np.ndarray)):
            corners_array = np.array(corners_data, dtype=np.float32)
            if corners_array.shape == (4, 2):
                return corners_array
        
        return None


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


def lpr_collate_fn(
    batch: List[Dict], 
    filter_invalid_layout: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for LPR dataset that handles optional fields.
    
    Args:
        batch: List of sample dictionaries from dataset.
        filter_invalid_layout: If True, samples with layout=-1 (invalid format)
            are excluded from the batch. If False, they are kept but should be
            handled appropriately in the loss computation.
        
    Returns:
        Batched dictionary with stacked tensors.
        
    Note:
        Samples with invalid layout (layout=-1) represent plates that don't match
        either Brazilian or Mercosul format. These can contaminate training if
        included, so by default they are filtered out.
    """
    # Filter out invalid layout samples if requested
    if filter_invalid_layout:
        valid_batch = [s for s in batch if s['layout'].item() >= 0]
        if len(valid_batch) == 0:
            # All samples were invalid - fall back to using original batch
            # but log a warning (caller should handle this case)
            valid_batch = batch
        batch = valid_batch
    
    # Required fields that are always present
    result = {
        'lr_frames': torch.stack([s['lr_frames'] for s in batch]),
        'hr_image': torch.stack([s['hr_image'] for s in batch]),
        'text_indices': torch.stack([s['text_indices'] for s in batch]),
        'layout': torch.stack([s['layout'] for s in batch]),
    }
    
    # Text (list of strings)
    result['text'] = [s['text'] for s in batch]
    
    # Plate style (list of strings)
    if 'plate_style' in batch[0]:
        result['plate_style'] = [s['plate_style'] for s in batch]
    
    # Optional corners - only include if ALL samples have corners
    if all('corners' in s for s in batch):
        result['corners'] = torch.stack([s['corners'] for s in batch])
    
    # Optional path
    if 'path' in batch[0]:
        result['path'] = [s['path'] for s in batch]
    
    return result


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
        drop_last=True,
        collate_fn=lpr_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lpr_collate_fn
    )
    
    return train_loader, val_loader
