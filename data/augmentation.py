"""
Data Augmentation for License Plate Recognition.

This module implements various augmentation techniques specific to
license plate images, including geometric and photometric transforms.
"""

import random
import numpy as np
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import cv2


class LPRAugmentation:
    """
    Augmentation pipeline for license plate images.
    
    Applies geometric and photometric augmentations to simulate
    real-world conditions and improve model robustness.
    
    Supports plate-style-aware augmentation for Brazilian vs Mercosur plates:
    - Mercosur plates have white background with blue band, may need different
      brightness/contrast ranges to preserve blue band visibility
    - Brazilian plates have grey background, may tolerate different augmentations
    """
    
    def __init__(
        self,
        # Geometric augmentations
        rotation_range: Tuple[float, float] = (-15, 15),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        perspective_strength: float = 0.1,
        shear_range: Tuple[float, float] = (-10, 10),
        
        # Photometric augmentations
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        hue_range: Tuple[float, float] = (-0.1, 0.1),
        
        # Style-specific photometric adjustments
        # Mercosur: white background, preserve blue band visibility
        mercosur_brightness_range: Optional[Tuple[float, float]] = None,
        mercosur_contrast_range: Optional[Tuple[float, float]] = None,
        # Brazilian: grey background, more tolerance
        brazilian_brightness_range: Optional[Tuple[float, float]] = None,
        brazilian_contrast_range: Optional[Tuple[float, float]] = None,
        
        # Degradation augmentations
        blur_probability: float = 0.3,
        blur_kernel_range: Tuple[int, int] = (3, 7),
        noise_probability: float = 0.3,
        noise_std_range: Tuple[float, float] = (5, 25),
        jpeg_probability: float = 0.2,
        jpeg_quality_range: Tuple[int, int] = (50, 90),
        
        # Motion blur
        motion_blur_probability: float = 0.2,
        motion_blur_kernel_range: Tuple[int, int] = (5, 15),
        
        # Shadow and lighting
        shadow_probability: float = 0.2,
        
        # Enable/disable
        enable_geometric: bool = True,
        enable_photometric: bool = True,
        enable_degradation: bool = True,
        style_aware: bool = True  # Enable plate-style-aware augmentation
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            rotation_range: Range of rotation angles in degrees.
            scale_range: Range of scale factors.
            perspective_strength: Strength of perspective transform.
            shear_range: Range of shear angles in degrees.
            brightness_range: Range of brightness adjustment.
            contrast_range: Range of contrast adjustment.
            saturation_range: Range of saturation adjustment.
            hue_range: Range of hue adjustment.
            blur_probability: Probability of applying blur.
            blur_kernel_range: Range of blur kernel sizes.
            noise_probability: Probability of adding noise.
            noise_std_range: Range of noise standard deviation.
            jpeg_probability: Probability of JPEG compression.
            jpeg_quality_range: Range of JPEG quality.
            motion_blur_probability: Probability of motion blur.
            motion_blur_kernel_range: Range of motion blur kernel.
            shadow_probability: Probability of adding shadow.
            enable_geometric: Whether to enable geometric augmentations.
            enable_photometric: Whether to enable photometric augmentations.
            enable_degradation: Whether to enable degradation augmentations.
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.perspective_strength = perspective_strength
        self.shear_range = shear_range
        
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        
        # Style-specific ranges (use defaults if not specified)
        self.mercosur_brightness_range = mercosur_brightness_range or brightness_range
        self.mercosur_contrast_range = mercosur_contrast_range or contrast_range
        self.brazilian_brightness_range = brazilian_brightness_range or brightness_range
        self.brazilian_contrast_range = brazilian_contrast_range or contrast_range
        
        self.blur_probability = blur_probability
        self.blur_kernel_range = blur_kernel_range
        self.noise_probability = noise_probability
        self.noise_std_range = noise_std_range
        self.jpeg_probability = jpeg_probability
        self.jpeg_quality_range = jpeg_quality_range
        
        self.motion_blur_probability = motion_blur_probability
        self.motion_blur_kernel_range = motion_blur_kernel_range
        
        self.shadow_probability = shadow_probability
        
        self.enable_geometric = enable_geometric
        self.enable_photometric = enable_photometric
        self.enable_degradation = enable_degradation
        self.style_aware = style_aware
        
        # Current plate style (set dynamically)
        self.current_plate_style = None  # 'brazilian' or 'mercosul'
    
    def set_plate_style(self, style: str):
        """
        Set the current plate style for style-aware augmentation.
        
        Args:
            style: 'brazilian' or 'mercosul'
        """
        self.current_plate_style = style.lower()
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation pipeline to image.
        
        Args:
            image: Input image as numpy array (H, W, C).
            
        Returns:
            Augmented image as numpy array.
        """
        # Convert to PIL Image
        img = Image.fromarray(image)
        
        # Apply geometric augmentations
        if self.enable_geometric:
            img = self._apply_geometric(img)
        
        # Apply photometric augmentations (style-aware if enabled)
        if self.enable_photometric:
            img = self._apply_photometric(img)
        
        # Convert back to numpy for degradation
        image = np.array(img)
        
        # Apply degradation augmentations
        if self.enable_degradation:
            image = self._apply_degradation(image)
        
        return image
    
    def _apply_geometric(self, img: Image.Image) -> Image.Image:
        """Apply geometric augmentations."""
        width, height = img.size
        
        # Random rotation
        angle = random.uniform(*self.rotation_range)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(128, 128, 128))
        
        # Random scale
        scale = random.uniform(*self.scale_range)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.BILINEAR)
        
        # Center crop or pad to original size
        if scale > 1:
            # Crop
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            img = img.crop((left, top, left + width, top + height))
        elif scale < 1:
            # Pad
            new_img = Image.new('RGB', (width, height), (128, 128, 128))
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            new_img.paste(img, (left, top))
            img = new_img
        
        return img
    
    def _apply_photometric(self, img: Image.Image) -> Image.Image:
        """Apply photometric augmentations with optional style-aware adjustments."""
        # Select brightness and contrast ranges based on plate style
        if self.style_aware and self.current_plate_style:
            if self.current_plate_style == 'mercosul':
                brightness_range = self.mercosur_brightness_range
                contrast_range = self.mercosur_contrast_range
            else:  # brazilian
                brightness_range = self.brazilian_brightness_range
                contrast_range = self.brazilian_contrast_range
        else:
            brightness_range = self.brightness_range
            contrast_range = self.contrast_range
        
        # Brightness
        factor = random.uniform(*brightness_range)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
        
        # Contrast
        factor = random.uniform(*contrast_range)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
        
        # Saturation (less aggressive for Mercosur to preserve blue band colors)
        if self.style_aware and self.current_plate_style == 'mercosul':
            # More conservative saturation changes for Mercosur
            sat_range = (0.9, 1.1)  # Preserve blue band colors
        else:
            sat_range = self.saturation_range
        
        factor = random.uniform(*sat_range)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(factor)
        
        return img
    
    def _apply_degradation(self, image: np.ndarray) -> np.ndarray:
        """Apply degradation augmentations."""
        # Gaussian blur
        if random.random() < self.blur_probability:
            kernel_size = random.choice(range(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2))
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Motion blur
        if random.random() < self.motion_blur_probability:
            kernel_size = random.randint(*self.motion_blur_kernel_range)
            kernel = np.zeros((kernel_size, kernel_size))
            angle = random.uniform(0, 180)
            
            # Create motion blur kernel
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
            
            # Rotate kernel
            center = (kernel_size // 2, kernel_size // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
            kernel = kernel / kernel.sum()
            
            image = cv2.filter2D(image, -1, kernel)
        
        # Gaussian noise
        if random.random() < self.noise_probability:
            std = random.uniform(*self.noise_std_range)
            noise = np.random.randn(*image.shape) * std
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # JPEG compression artifacts
        if random.random() < self.jpeg_probability:
            quality = random.randint(*self.jpeg_quality_range)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', image, encode_param)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Shadow
        if random.random() < self.shadow_probability:
            image = self._add_shadow(image)
        
        return image
    
    def _add_shadow(self, image: np.ndarray) -> np.ndarray:
        """Add random shadow to image."""
        h, w = image.shape[:2]
        
        # Random shadow region
        x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
        
        # Create shadow mask
        shadow_factor = random.uniform(0.4, 0.7)
        
        image = image.astype(np.float32)
        image[:, x1:x2] = image[:, x1:x2] * shadow_factor
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image


class PerspectiveTransform:
    """
    Perspective transformation with corner tracking.
    
    Applies perspective distortion while tracking corner positions
    for corner loss supervision.
    """
    
    def __init__(self, strength: float = 0.1):
        """
        Initialize perspective transform.
        
        Args:
            strength: Maximum displacement as fraction of image size.
        """
        self.strength = strength
    
    def __call__(
        self, 
        image: np.ndarray,
        corners: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply perspective transform.
        
        Args:
            image: Input image (H, W, C).
            corners: Original corner coordinates (4, 2).
            
        Returns:
            Tuple of (transformed_image, transformed_corners).
        """
        h, w = image.shape[:2]
        
        # Source corners
        src = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
        
        # Random destination corners
        max_disp = self.strength * min(h, w)
        dst = src + np.random.uniform(-max_disp, max_disp, src.shape).astype(np.float32)
        
        # Compute perspective matrix
        M = cv2.getPerspectiveTransform(src, dst)
        
        # Transform image
        transformed = cv2.warpPerspective(image, M, (w, h), borderValue=(128, 128, 128))
        
        # Transform corners
        if corners is not None:
            # Convert corners to homogeneous coordinates
            corners_h = np.hstack([corners, np.ones((4, 1))])
            transformed_corners = (M @ corners_h.T).T
            transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:]
        else:
            # Return transformed source corners
            transformed_corners = dst
        
        return transformed, transformed_corners


class MixUp:
    """
    MixUp augmentation for regularization.
    
    Blends two samples together with a random weight.
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp.
        
        Args:
            alpha: Beta distribution parameter.
        """
        self.alpha = alpha
    
    def __call__(
        self,
        batch1: Dict[str, Any],
        batch2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply MixUp to two batches.
        
        Args:
            batch1: First batch.
            batch2: Second batch.
            
        Returns:
            Mixed batch.
        """
        # Sample mixing weight
        lam = np.random.beta(self.alpha, self.alpha)
        
        mixed = {}
        
        # Mix images
        for key in ['lr_frames', 'hr_image']:
            if key in batch1 and key in batch2:
                mixed[key] = lam * batch1[key] + (1 - lam) * batch2[key]
        
        # Use one set of labels (no mixing for discrete labels)
        for key in ['text_indices', 'text', 'layout', 'corners']:
            if key in batch1:
                mixed[key] = batch1[key]
        
        mixed['mix_lambda'] = lam
        
        return mixed


class CutOut:
    """
    CutOut augmentation that randomly masks regions.
    """
    
    def __init__(
        self,
        num_holes: int = 1,
        max_hole_size: float = 0.2
    ):
        """
        Initialize CutOut.
        
        Args:
            num_holes: Number of rectangular regions to cut out.
            max_hole_size: Maximum hole size as fraction of image size.
        """
        self.num_holes = num_holes
        self.max_hole_size = max_hole_size
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply CutOut to image."""
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        
        for _ in range(self.num_holes):
            hole_h = int(random.uniform(0, self.max_hole_size) * h)
            hole_w = int(random.uniform(0, self.max_hole_size) * w)
            
            y = random.randint(0, h - hole_h)
            x = random.randint(0, w - hole_w)
            
            mask[y:y + hole_h, x:x + hole_w] = 0
        
        # Apply mask
        if image.ndim == 3:
            mask = np.expand_dims(mask, -1)
        
        return (image * mask).astype(np.uint8)
