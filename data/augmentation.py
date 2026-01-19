"""
Data Augmentation for License Plate Recognition.

This module implements various augmentation techniques specific to
license plate images, including geometric and photometric transforms.

IMPORTANT: For proper paired augmentation (LR and HR stay aligned), use
`augment_pair()` instead of calling `__call__()` separately on each image.
"""

import random
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class AugmentationParams:
    """
    Sampled augmentation parameters for paired augmentation.
    
    These parameters are sampled once and applied to both LR frames and HR target
    to ensure they stay aligned for proper supervision.
    """
    # Geometric params
    rotation_angle: float = 0.0
    scale_factor: float = 1.0
    
    # Photometric params
    brightness_factor: float = 1.0
    contrast_factor: float = 1.0
    saturation_factor: float = 1.0
    
    # Degradation params (only applied to LR)
    apply_blur: bool = False
    blur_kernel_size: int = 3
    apply_motion_blur: bool = False
    motion_blur_kernel_size: int = 5
    motion_blur_angle: float = 0.0
    apply_noise: bool = False
    noise_std: float = 0.0
    apply_jpeg: bool = False
    jpeg_quality: int = 90
    apply_shadow: bool = False
    shadow_x1: int = 0
    shadow_x2: int = 0
    shadow_factor: float = 1.0


class LPRAugmentation:
    """
    Augmentation pipeline for license plate images.
    
    Applies geometric and photometric augmentations to simulate
    real-world conditions and improve model robustness.
    
    Supports plate-style-aware augmentation for Brazilian vs Mercosur plates:
    - Mercosur plates have white background with blue band, may need different
      brightness/contrast ranges to preserve blue band visibility
    - Brazilian plates have grey background, may tolerate different augmentations
    
    IMPORTANT: When geometric augmentations are applied, the transform parameters
    are stored and can be used to transform corner annotations via 
    get_last_transform_matrix() and transform_corners().
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
        
        # Store last geometric transform parameters for corner transformation
        # This is a 3x3 affine transformation matrix
        self._last_transform_matrix = None
        self._last_image_size = None  # (width, height)
    
    def set_plate_style(self, style: str):
        """
        Set the current plate style for style-aware augmentation.
        
        Args:
            style: 'brazilian' or 'mercosul'
        """
        self.current_plate_style = style.lower()
    
    def get_last_transform_matrix(self) -> Optional[np.ndarray]:
        """
        Get the transformation matrix from the last geometric augmentation.
        
        Returns:
            3x3 transformation matrix or None if no geometric transform was applied.
        """
        return self._last_transform_matrix
    
    def get_last_image_size(self) -> Optional[Tuple[int, int]]:
        """
        Get the image size (width, height) from the last augmentation.
        
        Returns:
            (width, height) tuple or None.
        """
        return self._last_image_size
    
    def transform_corners(
        self, 
        corners: np.ndarray, 
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Transform corner coordinates using the last geometric augmentation.
        
        Args:
            corners: Corner coordinates of shape (4, 2) in pixel coordinates.
            original_size: Original image size (width, height) before augmentation.
            
        Returns:
            Transformed corner coordinates of shape (4, 2).
        """
        if self._last_transform_matrix is None:
            return corners
        
        # Convert corners to homogeneous coordinates
        corners_h = np.hstack([corners, np.ones((4, 1))])
        
        # Apply transformation
        transformed = (self._last_transform_matrix @ corners_h.T).T
        
        # Convert back from homogeneous (handle perspective if present)
        if transformed.shape[1] == 3:
            transformed = transformed[:, :2] / transformed[:, 2:3]
        
        return transformed.astype(np.float32)
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation pipeline to image.
        
        Args:
            image: Input image as numpy array (H, W, C).
            
        Returns:
            Augmented image as numpy array.
            
        Note:
            After calling this method, use get_last_transform_matrix() and 
            transform_corners() to get the transformed corner coordinates.
            
        WARNING: This method applies augmentations independently. For paired
        LR/HR augmentation, use augment_pair() instead to keep images aligned.
        """
        # Reset transform matrix
        self._last_transform_matrix = None
        self._last_image_size = (image.shape[1], image.shape[0])  # (width, height)
        
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
    
    def sample_augmentation_params(self, image_size: Tuple[int, int]) -> AugmentationParams:
        """
        Sample all augmentation parameters once for paired augmentation.
        
        This method samples random values for all augmentation operations.
        The same params can then be applied to both LR frames and HR target
        to ensure they stay aligned.
        
        Args:
            image_size: (width, height) of the image being augmented.
            
        Returns:
            AugmentationParams with all sampled values.
        """
        width, height = image_size
        
        # Geometric params
        rotation_angle = random.uniform(*self.rotation_range) if self.enable_geometric else 0.0
        scale_factor = random.uniform(*self.scale_range) if self.enable_geometric else 1.0
        
        # Photometric params - select ranges based on plate style
        if self.style_aware and self.current_plate_style:
            if self.current_plate_style == 'mercosul':
                brightness_range = self.mercosur_brightness_range
                contrast_range = self.mercosur_contrast_range
                sat_range = (0.9, 1.1)  # Preserve blue band colors
            else:  # brazilian
                brightness_range = self.brazilian_brightness_range
                contrast_range = self.brazilian_contrast_range
                sat_range = self.saturation_range
        else:
            brightness_range = self.brightness_range
            contrast_range = self.contrast_range
            sat_range = self.saturation_range
        
        brightness_factor = random.uniform(*brightness_range) if self.enable_photometric else 1.0
        contrast_factor = random.uniform(*contrast_range) if self.enable_photometric else 1.0
        saturation_factor = random.uniform(*sat_range) if self.enable_photometric else 1.0
        
        # Degradation params (only applied to LR)
        apply_blur = self.enable_degradation and random.random() < self.blur_probability
        blur_kernel_size = random.choice(range(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2))
        
        apply_motion_blur = self.enable_degradation and random.random() < self.motion_blur_probability
        motion_blur_kernel_size = random.randint(*self.motion_blur_kernel_range)
        motion_blur_angle = random.uniform(0, 180)
        
        apply_noise = self.enable_degradation and random.random() < self.noise_probability
        noise_std = random.uniform(*self.noise_std_range)
        
        apply_jpeg = self.enable_degradation and random.random() < self.jpeg_probability
        jpeg_quality = random.randint(*self.jpeg_quality_range)
        
        apply_shadow = self.enable_degradation and random.random() < self.shadow_probability
        shadow_x1, shadow_x2 = sorted([random.randint(0, width), random.randint(0, width)])
        shadow_factor = random.uniform(0.4, 0.7)
        
        return AugmentationParams(
            rotation_angle=rotation_angle,
            scale_factor=scale_factor,
            brightness_factor=brightness_factor,
            contrast_factor=contrast_factor,
            saturation_factor=saturation_factor,
            apply_blur=apply_blur,
            blur_kernel_size=blur_kernel_size,
            apply_motion_blur=apply_motion_blur,
            motion_blur_kernel_size=motion_blur_kernel_size,
            motion_blur_angle=motion_blur_angle,
            apply_noise=apply_noise,
            noise_std=noise_std,
            apply_jpeg=apply_jpeg,
            jpeg_quality=jpeg_quality,
            apply_shadow=apply_shadow,
            shadow_x1=shadow_x1,
            shadow_x2=shadow_x2,
            shadow_factor=shadow_factor,
        )
    
    def augment_pair(
        self,
        lr_frames: List[np.ndarray],
        hr_image: np.ndarray,
        corners: Optional[np.ndarray] = None,
        params: Optional[AugmentationParams] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Apply paired augmentation to LR frames and HR target.
        
        This ensures LR and HR stay aligned by using the same geometric and
        photometric transforms. Degradations are only applied to LR frames.
        
        Args:
            lr_frames: List of LR frames as numpy arrays (H, W, C).
            hr_image: HR target image as numpy array (H, W, C).
            corners: Optional corner coordinates (4, 2) in HR image space.
            params: Pre-sampled augmentation params. If None, samples new ones.
            
        Returns:
            Tuple of (augmented_lr_frames, augmented_hr, transformed_corners, transform_matrix).
            - augmented_lr_frames: List of augmented LR frames
            - augmented_hr: Augmented HR image
            - transformed_corners: Transformed corner coordinates (or None if corners was None)
            - transform_matrix: 3x3 transformation matrix applied to HR image
        """
        # Sample params if not provided
        hr_h, hr_w = hr_image.shape[:2]
        if params is None:
            params = self.sample_augmentation_params((hr_w, hr_h))
        
        # Apply geometric + photometric to HR (no degradation)
        hr_aug, hr_transform = self._apply_geometric_with_params(
            Image.fromarray(hr_image), params
        )
        hr_aug = self._apply_photometric_with_params(hr_aug, params)
        hr_augmented = np.array(hr_aug)
        
        # Transform corners using HR transform
        transformed_corners = None
        if corners is not None:
            transformed_corners = self._transform_corners_with_matrix(
                corners, hr_transform
            )
        
        # Apply to each LR frame: geometric + photometric + degradation
        lr_augmented = []
        for lr_frame in lr_frames:
            lr_h, lr_w = lr_frame.shape[:2]
            
            # Compute scaling between LR and HR for geometric transforms
            scale_x = lr_w / hr_w
            scale_y = lr_h / hr_h
            
            # Apply geometric with scaled params
            lr_aug, _ = self._apply_geometric_with_params(
                Image.fromarray(lr_frame), params, scale=(scale_x, scale_y)
            )
            
            # Apply photometric (same params)
            lr_aug = self._apply_photometric_with_params(lr_aug, params)
            
            # Convert to numpy for degradation
            lr_np = np.array(lr_aug)
            
            # Apply degradation (only to LR)
            lr_np = self._apply_degradation_with_params(lr_np, params)
            
            lr_augmented.append(lr_np)
        
        return lr_augmented, hr_augmented, transformed_corners, hr_transform
    
    def _apply_geometric_with_params(
        self,
        img: Image.Image,
        params: AugmentationParams,
        scale: Tuple[float, float] = (1.0, 1.0),
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Apply geometric augmentations with pre-sampled parameters.
        
        Args:
            img: PIL Image to transform.
            params: Sampled augmentation parameters.
            scale: (scale_x, scale_y) for adapting transforms to different resolutions.
            
        Returns:
            Tuple of (transformed_image, transform_matrix).
        """
        if not self.enable_geometric:
            return img, np.eye(3)
        
        width, height = img.size
        
        # Initialize transformation matrix as identity
        transform_matrix = np.eye(3)
        
        # Apply rotation
        angle = params.rotation_angle
        angle_rad = np.deg2rad(angle)
        cx, cy = width / 2, height / 2
        
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a, cx - cx * cos_a + cy * sin_a],
            [sin_a, cos_a, cy - cx * sin_a - cy * cos_a],
            [0, 0, 1]
        ])
        transform_matrix = rotation_matrix @ transform_matrix
        
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(128, 128, 128))
        
        # Apply scale
        scale_factor = params.scale_factor
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height), Image.BILINEAR)
        
        scale_matrix = np.array([
            [scale_factor, 0, 0],
            [0, scale_factor, 0],
            [0, 0, 1]
        ])
        transform_matrix = scale_matrix @ transform_matrix
        
        # Center crop or pad to original size
        if scale_factor > 1:
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            img = img.crop((left, top, left + width, top + height))
            
            crop_matrix = np.array([
                [1, 0, -left],
                [0, 1, -top],
                [0, 0, 1]
            ])
            transform_matrix = crop_matrix @ transform_matrix
        elif scale_factor < 1:
            new_img = Image.new('RGB', (width, height), (128, 128, 128))
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            new_img.paste(img, (left, top))
            img = new_img
            
            pad_matrix = np.array([
                [1, 0, left],
                [0, 1, top],
                [0, 0, 1]
            ])
            transform_matrix = pad_matrix @ transform_matrix
        
        return img, transform_matrix
    
    def _apply_photometric_with_params(
        self,
        img: Image.Image,
        params: AugmentationParams,
    ) -> Image.Image:
        """Apply photometric augmentations with pre-sampled parameters."""
        if not self.enable_photometric:
            return img
        
        # Brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(params.brightness_factor)
        
        # Contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(params.contrast_factor)
        
        # Saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(params.saturation_factor)
        
        return img
    
    def _apply_degradation_with_params(
        self,
        image: np.ndarray,
        params: AugmentationParams,
    ) -> np.ndarray:
        """Apply degradation augmentations with pre-sampled parameters (LR only)."""
        if not self.enable_degradation:
            return image
        
        # Gaussian blur
        if params.apply_blur:
            kernel_size = params.blur_kernel_size
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Motion blur
        if params.apply_motion_blur:
            kernel_size = params.motion_blur_kernel_size
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
            
            center = (kernel_size // 2, kernel_size // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, params.motion_blur_angle, 1.0)
            kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
            kernel = kernel / (kernel.sum() + 1e-8)
            
            image = cv2.filter2D(image, -1, kernel)
        
        # Gaussian noise
        if params.apply_noise:
            noise = np.random.randn(*image.shape) * params.noise_std
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # JPEG compression artifacts
        if params.apply_jpeg:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), params.jpeg_quality]
            _, encoded = cv2.imencode('.jpg', image, encode_param)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Shadow
        if params.apply_shadow:
            image = image.astype(np.float32)
            image[:, params.shadow_x1:params.shadow_x2] *= params.shadow_factor
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def _transform_corners_with_matrix(
        self,
        corners: np.ndarray,
        transform_matrix: np.ndarray,
    ) -> np.ndarray:
        """Transform corner coordinates using a transformation matrix."""
        # Convert corners to homogeneous coordinates
        corners_h = np.hstack([corners, np.ones((4, 1))])
        
        # Apply transformation
        transformed = (transform_matrix @ corners_h.T).T
        
        # Convert back from homogeneous (handle perspective if present)
        if transformed.shape[1] == 3:
            transformed = transformed[:, :2] / transformed[:, 2:3]
        
        return transformed.astype(np.float32)
    
    def _apply_geometric(self, img: Image.Image) -> Image.Image:
        """Apply geometric augmentations and track transformation matrix."""
        width, height = img.size
        
        # Initialize transformation matrix as identity
        # Using 3x3 matrix to support affine transforms
        transform_matrix = np.eye(3)
        
        # Random rotation
        angle = random.uniform(*self.rotation_range)
        angle_rad = np.deg2rad(angle)
        
        # Rotation around image center
        cx, cy = width / 2, height / 2
        
        # Translation to origin, rotate, translate back
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a, cx - cx * cos_a + cy * sin_a],
            [sin_a, cos_a, cy - cx * sin_a - cy * cos_a],
            [0, 0, 1]
        ])
        transform_matrix = rotation_matrix @ transform_matrix
        
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(128, 128, 128))
        
        # Random scale
        scale = random.uniform(*self.scale_range)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.BILINEAR)
        
        # Scale matrix (around origin, then we'll handle crop/pad offset)
        scale_matrix = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ])
        transform_matrix = scale_matrix @ transform_matrix
        
        # Center crop or pad to original size
        if scale > 1:
            # Crop - offset for center crop
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            img = img.crop((left, top, left + width, top + height))
            
            # Crop offset (translate coordinates)
            crop_matrix = np.array([
                [1, 0, -left],
                [0, 1, -top],
                [0, 0, 1]
            ])
            transform_matrix = crop_matrix @ transform_matrix
        elif scale < 1:
            # Pad - offset for centering
            new_img = Image.new('RGB', (width, height), (128, 128, 128))
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            new_img.paste(img, (left, top))
            img = new_img
            
            # Pad offset (translate coordinates)
            pad_matrix = np.array([
                [1, 0, left],
                [0, 1, top],
                [0, 0, 1]
            ])
            transform_matrix = pad_matrix @ transform_matrix
        
        # Store the final transformation matrix
        self._last_transform_matrix = transform_matrix
        
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
