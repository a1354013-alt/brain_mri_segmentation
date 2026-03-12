import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.attention_unet import AttentionUNet
from utils.dataset import BraTSDataset
import numpy as np
from tqdm import tqdm

def dice_coeff(pred, target):
    """計算 Dice 係數"""
    smooth = 1.0
    num = pred.size(0)
    m1 = pred.view(num, -1).float()
    m2 = target.view(num, -1).float()
    intersection = (m1 * m2).sum().float()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class DiceLoss(nn.Module):
    """Dice Loss 損失函數"""
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        return 1 - dice_coeff(inputs, targets)

def train_model(model, train_loader, val_loader, epochs=10, device='cpu'):
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # 驗證階段
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = torch.sigmoid(model(images))
                outputs = (outputs > 0.5).float()
                val_dice += dice_coeff(outputs, masks).item()
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Dice = {val_dice/len(val_loader):.4f}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 使用 AttentionUNet 並設定 4 通道輸入 (FLAIR, T1, T1ce, T2)
    model = AttentionUNet(n_channels=4, n_classes=1).to(device)
    print("Attention U-Net initialized with 4-channel input. Ready for training.")
