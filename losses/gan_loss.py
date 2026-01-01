"""
GAN Losses for Adversarial Training.

This module implements various GAN losses for training the image
restoration module, including standard GAN loss, LSGAN, and perceptual loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class GANLoss(nn.Module):
    """
    GAN loss for generator training.
    
    Supports multiple GAN variants:
        - 'vanilla': Standard cross-entropy GAN loss
        - 'lsgan': Least Squares GAN loss
        - 'hinge': Hinge loss
        - 'wgan': Wasserstein GAN loss
    """
    
    def __init__(
        self,
        gan_mode: str = 'vanilla',
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0
    ):
        """
        Initialize GAN loss.
        
        Args:
            gan_mode: Type of GAN loss.
            target_real_label: Label for real images.
            target_fake_label: Label for fake images.
        """
        super().__init__()
        
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        self.gan_mode = gan_mode
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode in ['hinge', 'wgan']:
            self.loss = None
        else:
            raise ValueError(f"Unknown GAN mode: {gan_mode}")
    
    def get_target_tensor(
        self, 
        prediction: torch.Tensor, 
        target_is_real: bool
    ) -> torch.Tensor:
        """
        Create target tensor with same shape as prediction.
        
        Args:
            prediction: Discriminator output.
            target_is_real: Whether target should be real or fake.
            
        Returns:
            Target tensor.
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        
        return target_tensor.expand_as(prediction)
    
    def forward(
        self, 
        prediction: torch.Tensor, 
        target_is_real: bool
    ) -> torch.Tensor:
        """
        Compute GAN loss.
        
        Args:
            prediction: Discriminator output for real or fake images.
            target_is_real: Whether the prediction should be real.
            
        Returns:
            GAN loss value.
        """
        if self.gan_mode in ['vanilla', 'lsgan']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'hinge':
            if target_is_real:
                loss = F.relu(1 - prediction).mean()
            else:
                loss = F.relu(1 + prediction).mean()
        elif self.gan_mode == 'wgan':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        
        return loss


class DiscriminatorLoss(nn.Module):
    """
    Complete discriminator training loss.
    
    Combines losses for real and fake images with optional
    gradient penalty for WGAN-GP style training.
    """
    
    def __init__(
        self,
        gan_mode: str = 'vanilla',
        use_gradient_penalty: bool = False,
        gp_weight: float = 10.0
    ):
        """
        Initialize discriminator loss.
        
        Args:
            gan_mode: Type of GAN loss.
            use_gradient_penalty: Whether to use gradient penalty.
            gp_weight: Weight for gradient penalty.
        """
        super().__init__()
        
        self.gan_loss = GANLoss(gan_mode)
        self.use_gradient_penalty = use_gradient_penalty
        self.gp_weight = gp_weight
    
    def gradient_penalty(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP.
        
        Args:
            discriminator: Discriminator network.
            real: Real images.
            fake: Fake images.
            
        Returns:
            Gradient penalty value.
        """
        batch_size = real.size(0)
        device = real.device
        
        # Random interpolation factor
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        
        # Interpolated images
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)
        
        # Discriminator output
        d_interpolated = discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Flatten gradients
        gradients = gradients.view(batch_size, -1)
        
        # Compute gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
    
    def forward(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute discriminator loss.
        
        Args:
            discriminator: Discriminator network.
            real: Real images.
            fake: Fake images (detached).
            
        Returns:
            Tuple of (total_loss, loss_dict).
        """
        # Real loss
        pred_real = discriminator(real)
        loss_real = self.gan_loss(pred_real, target_is_real=True)
        
        # Fake loss
        pred_fake = discriminator(fake.detach())
        loss_fake = self.gan_loss(pred_fake, target_is_real=False)
        
        # Total loss
        loss_d = (loss_real + loss_fake) / 2
        
        loss_dict = {
            'loss_d_real': loss_real.item(),
            'loss_d_fake': loss_fake.item(),
            'loss_d': loss_d.item()
        }
        
        # Gradient penalty
        if self.use_gradient_penalty:
            gp = self.gradient_penalty(discriminator, real, fake)
            loss_d = loss_d + self.gp_weight * gp
            loss_dict['gradient_penalty'] = gp.item()
        
        return loss_d, loss_dict


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features.
    
    Computes L1 or L2 distance between feature representations
    of real and generated images.
    """
    
    def __init__(
        self,
        layer_weights: dict = None,
        use_pretrained: bool = True,
        criterion: str = 'l1'
    ):
        """
        Initialize perceptual loss.
        
        Args:
            layer_weights: Dictionary mapping layer names to weights.
            use_pretrained: Whether to use pretrained VGG.
            criterion: Loss criterion ('l1' or 'l2').
        """
        super().__init__()
        
        if layer_weights is None:
            layer_weights = {
                'conv1_2': 0.1,
                'conv2_2': 0.1,
                'conv3_4': 1.0,
                'conv4_4': 1.0,
                'conv5_4': 1.0
            }
        
        self.layer_weights = layer_weights
        self.criterion = nn.L1Loss() if criterion == 'l1' else nn.MSELoss()
        
        # Load VGG with lazy import to avoid heavy dependency
        self.vgg = None
        self._use_pretrained = use_pretrained
    
    def _load_vgg(self, device: torch.device):
        """Lazy load VGG network."""
        if self.vgg is not None:
            return
        
        try:
            from torchvision.models import vgg19, VGG19_Weights
            
            if self._use_pretrained:
                vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            else:
                vgg = vgg19()
            
            # Extract feature layers
            self.vgg = vgg.features.to(device).eval()
            
            # Freeze parameters
            for param in self.vgg.parameters():
                param.requires_grad = False
            
            # Layer indices for VGG19
            self.layer_indices = {
                'conv1_2': 3,
                'conv2_2': 8,
                'conv3_4': 17,
                'conv4_4': 26,
                'conv5_4': 35
            }
        except ImportError:
            print("Warning: torchvision not available for perceptual loss")
            self.vgg = None
    
    def _extract_features(
        self, 
        x: torch.Tensor
    ) -> dict:
        """
        Extract features from VGG layers.
        
        Args:
            x: Input images (normalized to ImageNet range).
            
        Returns:
            Dictionary of layer features.
        """
        features = {}
        
        for name, idx in self.layer_indices.items():
            if name in self.layer_weights:
                x = self.vgg[:idx + 1](x)
                features[name] = x
        
        return features
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted/generated images.
            target: Target/real images.
            
        Returns:
            Perceptual loss value.
        """
        self._load_vgg(pred.device)
        
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)
        
        # Normalize to ImageNet range
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        # Extract features
        pred_features = self._extract_features(pred_norm)
        target_features = self._extract_features(target_norm)
        
        # Compute weighted loss
        loss = 0.0
        for name, weight in self.layer_weights.items():
            if name in pred_features:
                loss += weight * self.criterion(
                    pred_features[name], 
                    target_features[name].detach()
                )
        
        return loss


class PixelLoss(nn.Module):
    """
    Pixel-wise reconstruction loss.
    """
    
    def __init__(self, criterion: str = 'l1'):
        super().__init__()
        
        if criterion == 'l1':
            self.loss = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.loss = nn.MSELoss()
        elif criterion == 'smooth_l1':
            self.loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pixel loss.
        
        Args:
            pred: Predicted images.
            target: Target images.
            
        Returns:
            Pixel loss value.
        """
        return self.loss(pred, target)
