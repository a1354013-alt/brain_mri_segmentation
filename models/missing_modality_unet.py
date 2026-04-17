"""
Missing-Aware U-Net for BraTS segmentation with missing modality support.

This module provides a ModalityAwareUNet that accepts both image and modality_mask,
enabling the model to explicitly know which modalities are present/missing.

The architecture is based on AttentionUNet with a lightweight modality-aware front-end.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityAwareStem(nn.Module):
    """
    Lightweight front-end that conditions early features on modality_mask.
    
    The modality_mask is a binary vector [1,1,0,1] indicating which modalities exist.
    We use this to gate/scale the initial convolution outputs per modality channel.
    """
    
    def __init__(self, n_modalities: int = 4, base_channels: int = 32):
        super().__init__()
        self.n_modalities = n_modalities
        self.base_channels = base_channels
        
        # Learnable per-modality scaling factors (initialized to 1.0)
        self.modality_weights = nn.Parameter(torch.ones(n_modalities))
        
        # Initial conv that processes each modality separately then combines
        self.individual_conv = nn.Conv2d(1, base_channels, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x: torch.Tensor, modality_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, H, W) where N is number of modalities
            modality_mask: Binary tensor of shape (B, N) or (N,) indicating presence
        Returns:
            Feature tensor of shape (B, base_channels, H, W)
        """
        B, N, H, W = x.shape
        
        if modality_mask is None:
            # Assume all modalities present
            modality_mask = torch.ones(B, N, device=x.device)
        
        if modality_mask.dim() == 1:
            modality_mask = modality_mask.unsqueeze(0).expand(B, -1)
        
        # Apply per-modality gating based on mask
        # Reshape weights for broadcasting: (1, N, 1, 1)
        weights = self.modality_weights.view(1, N, 1, 1)
        
        # Gate input channels by mask and learned weights
        gated_x = x * modality_mask.view(B, N, 1, 1) * weights
        
        # Process each modality through shared conv and sum
        features = None
        for i in range(N):
            mod_feat = self.individual_conv(gated_x[:, i:i+1, :, :])  # (B, C, H, W)
            if features is None:
                features = mod_feat
            else:
                features = features + mod_feat
        
        return features


class ConvBlock(nn.Module):
    """Standard conv block with BN, ReLU, and optional dropout."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate for skip connections."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ModalityAwareUNet(nn.Module):
    """
    Missing-aware U-Net for BraTS segmentation.
    
    This model extends AttentionUNet with explicit modality_mask conditioning.
    It can handle missing modalities during both training and inference.
    
    Usage:
        # With modality mask (recommended for missing modality scenarios)
        output = model(image, modality_mask=mask)
        
        # Without modality mask (falls back to standard behavior)
        output = model(image)
    """
    
    def __init__(
        self,
        n_modalities: int = 4,
        n_classes: int = 1,
        dropout_p: float = 0.2,
        use_modality_stem: bool = True,
    ):
        super().__init__()
        self.n_modalities = n_modalities
        self.n_classes = n_classes
        self.dropout_p = dropout_p
        self.use_modality_stem = use_modality_stem
        
        # Modality-aware stem (optional but recommended)
        if use_modality_stem:
            self.stem = ModalityAwareStem(n_modalities=n_modalities, base_channels=32)
            stem_out_channels = 32
        else:
            # Standard first conv when not using modality-aware stem
            self.stem = nn.Conv2d(n_modalities, 64, kernel_size=3, padding=1, bias=False)
            stem_out_channels = 64
        
        # Encoder
        if use_modality_stem:
            self.conv1 = ConvBlock(stem_out_channels, 64, dropout_p)
        else:
            self.conv1 = nn.Identity()
            # Already have 64 channels from stem
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(64, 128, dropout_p)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(128, 256, dropout_p)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = ConvBlock(256, 512, dropout_p)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.center = ConvBlock(512, 1024, dropout_p)

        # Decoder with attention gates
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.up_conv4 = ConvBlock(1024, 512, dropout_p)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.up_conv3 = ConvBlock(512, 256, dropout_p)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.up_conv2 = ConvBlock(256, 128, dropout_p)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.up_conv1 = ConvBlock(128, 64, dropout_p)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def _align_and_concat(self, x_skip: torch.Tensor, x_up: torch.Tensor) -> torch.Tensor:
        """Align decoder and skip features before concatenation."""
        if x_skip.shape[2:] != x_up.shape[2:]:
            diff_y = x_skip.size()[2] - x_up.size()[2]
            diff_x = x_skip.size()[3] - x_up.size()[3]

            pad_y = max(0, diff_y)
            pad_x = max(0, diff_x)
            if pad_y > 0 or pad_x > 0:
                x_up = F.pad(
                    x_up,
                    [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2],
                )

            crop_y = max(0, -diff_y)
            crop_x = max(0, -diff_x)
            if crop_y > 0 or crop_x > 0:
                y_start = crop_y // 2
                x_start = crop_x // 2
                x_up = x_up[:, :, y_start : y_start + x_skip.size()[2], x_start : x_start + x_skip.size()[3]]

        assert x_skip.shape[2:] == x_up.shape[2:], (
            f"Size mismatch after alignment: skip {x_skip.shape[2:]} vs up {x_up.shape[2:]}"
        )
        return torch.cat([x_skip, x_up], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image tensor of shape (B, N, H, W) where N is number of modalities
            modality_mask: Optional binary tensor of shape (B, N) or (N,) indicating
                          which modalities are present (1) or missing (0)
        
        Returns:
            Segmentation logits of shape (B, n_classes, H, W)
        """
        # Apply modality-aware stem
        if self.use_modality_stem:
            e1 = self.stem(x, modality_mask)
            e1 = self.conv1(e1)
        else:
            e1 = self.stem(x)
            e1 = self.conv1(e1) if hasattr(self, 'conv1') and not isinstance(self.conv1, nn.Identity) else e1
        
        # Encoder
        e2 = self.conv2(self.maxpool1(e1))
        e3 = self.conv3(self.maxpool2(e2))
        e4 = self.conv4(self.maxpool3(e3))

        # Bottleneck
        c = self.center(self.maxpool4(e4))

        # Decoder with attention
        d4 = self.up4(c)
        x4 = self.att4(g=d4, x=e4)
        d4 = self._align_and_concat(x4, d4)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=e3)
        d3 = self._align_and_concat(x3, d3)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = self._align_and_concat(x2, d2)
        d2 = self.up_conv2(d2)

        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = self._align_and_concat(x1, d1)
        d1 = self.up_conv1(d1)

        return self.outc(d1)
