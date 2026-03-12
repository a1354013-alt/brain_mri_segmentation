"""
Attention U-Net implementation with robust input size protection (v2.3)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, n_channels: int = 4, n_classes: int = 1, dropout_p: float = 0.2):
        super().__init__()
        
        # Encoder
        self.conv1 = ConvBlock(n_channels, 64, dropout_p)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(64, 128, dropout_p)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(128, 256, dropout_p)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = ConvBlock(256, 512, dropout_p)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Center
        self.center = ConvBlock(512, 1024, dropout_p)
        
        # Decoder
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
        """
        尺寸對齊保護 (v2.3)：
        1. 若 x_up 較小 (diff > 0) -> 使用 Padding
        2. 若 x_up 較大 (diff < 0) -> 使用 Center Crop
        """
        if x_skip.shape[2:] != x_up.shape[2:]:
            diff_y = x_skip.size()[2] - x_up.size()[2]
            diff_x = x_skip.size()[3] - x_up.size()[3]
            
            # 處理 Padding (diff > 0)
            pad_y = max(0, diff_y)
            pad_x = max(0, diff_x)
            if pad_y > 0 or pad_x > 0:
                x_up = F.pad(x_up, [pad_x // 2, pad_x - pad_x // 2,
                                    pad_y // 2, pad_y - pad_y // 2])
            
            # 處理 Crop (diff < 0)
            crop_y = max(0, -diff_y)
            crop_x = max(0, -diff_x)
            if crop_y > 0 or crop_x > 0:
                y_start = crop_y // 2
                x_start = crop_x // 2
                x_up = x_up[:, :, y_start:y_start + x_skip.size()[2], x_start:x_start + x_skip.size()[3]]
                
        return torch.cat([x_skip, x_up], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.conv1(x)
        e2 = self.conv2(self.maxpool1(e1))
        e3 = self.conv3(self.maxpool2(e2))
        e4 = self.conv4(self.maxpool3(e3))
        
        # Center
        c = self.center(self.maxpool4(e4))
        
        # Decoder with Attention & Alignment Protection
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
