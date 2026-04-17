"""
Model and loss factory functions for BraTS segmentation.

Provides stable, minimal interfaces for building models and losses
based on configuration settings. This enables easy switching between
baseline and missing-aware variants without modifying training code.
"""

from __future__ import annotations

import torch.nn as nn

import config


def build_model(
    n_modalities: int | None = None,
    n_classes: int | None = None,
    dropout_p: float | None = None,
    model_variant: str | None = None,
) -> nn.Module:
    """
    Build a segmentation model based on configuration.
    
    Args:
        n_modalities: Number of input modalities (default: from config)
        n_classes: Number of output classes (default: from config)
        dropout_p: Dropout probability (default: from config)
        model_variant: Model type - "baseline" or "missing_aware" (default: from config)
    
    Returns:
        Initialized model instance
    
    Raises:
        ValueError: If model_variant is not recognized
    """
    n_modalities = n_modalities if n_modalities is not None else config.N_MODALITIES
    n_classes = n_classes if n_classes is not None else config.N_CLASSES
    dropout_p = dropout_p if dropout_p is not None else config.DROPOUT_P
    model_variant = model_variant if model_variant is not None else config.MODEL_VARIANT
    
    if model_variant == "baseline":
        from models.attention_unet import AttentionUNet
        return AttentionUNet(
            n_channels=n_modalities,
            n_classes=n_classes,
            dropout_p=dropout_p,
        )
    elif model_variant == "missing_aware":
        from models.missing_modality_unet import ModalityAwareUNet
        return ModalityAwareUNet(
            n_modalities=n_modalities,
            n_classes=n_classes,
            dropout_p=dropout_p,
            use_modality_stem=True,
        )
    else:
        raise ValueError(f"Unknown model_variant: {model_variant}. Use 'baseline' or 'missing_aware'.")


def build_loss(loss_variant: str | None = None) -> nn.Module:
    """
    Build a loss function based on configuration.
    
    Args:
        loss_variant: Loss type - "dice_bce", "dice", "bce" (default: from config)
    
    Returns:
        Initialized loss module
    
    Raises:
        ValueError: If loss_variant is not recognized
    """
    loss_variant = loss_variant if loss_variant is not None else config.LOSS_VARIANT
    
    if loss_variant == "dice_bce":
        return DiceBCELoss(dice_weight=config.DICE_WEIGHT, bce_weight=config.BCE_WEIGHT)
    elif loss_variant == "dice":
        from train import DiceLoss
        return DiceLoss()
    elif loss_variant == "bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss_variant: {loss_variant}. Use 'dice_bce', 'dice', or 'bce'.")


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for binary segmentation."""
    
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce_criterion = nn.BCEWithLogitsLoss()
    
    def _dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss."""
        smooth = 1.0
        inputs_flat = inputs.view(inputs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (inputs_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + smooth) / (inputs_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth)
        return 1 - dice.mean()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined Dice + BCE loss.
        
        Args:
            inputs: Logits tensor of shape (B, 1, H, W)
            targets: Binary mask tensor of shape (B, 1, H, W)
        
        Returns:
            Scalar loss value
        """
        inputs_sigmoid = torch.sigmoid(inputs)
        dice_loss = self._dice_loss(inputs_sigmoid, targets)
        bce_loss = self.bce_criterion(inputs, targets)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss
