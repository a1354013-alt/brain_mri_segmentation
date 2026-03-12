"""
Training module with enhanced features:
- Per-sample Dice calculation
- Mixed precision training (AMP)
- Learning rate scheduling
- Gradient clipping
- Model checkpointing
- Training logging
"""
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import config


def dice_coeff_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    計算 per-sample Dice 係數
    
    Args:
        pred: 預測結果 (B, C, H, W)
        target: 真實標籤 (B, C, H, W)
        
    Returns:
        每個樣本的 Dice 係數 (B,)
    """
    smooth = 1.0
    batch_size = pred.size(0)
    
    # Flatten spatial dimensions for each sample
    pred_flat = pred.view(batch_size, -1).float()
    target_flat = target.view(batch_size, -1).float()
    
    # Calculate intersection and union for each sample
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice_per_sample = (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    
    return dice_per_sample


def dice_coeff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    計算 batch 平均 Dice 係數
    
    Args:
        pred: 預測結果 (B, C, H, W)
        target: 真實標籤 (B, C, H, W)
        
    Returns:
        平均 Dice 係數
    """
    return dice_coeff_per_sample(pred, target).mean()


class DiceLoss(nn.Module):
    """
    Dice Loss 損失函數
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        return 1 - dice_coeff(inputs, targets)


class Trainer:
    """
    訓練器類別，封裝所有訓練相關功能
    """
    def __init__(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        device: torch.device,
        use_amp: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp
        
        # Loss and optimizer
        self.criterion = DiceLoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',  # maximize Dice score
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Tensorboard writer
        self.writer = SummaryWriter(config.TENSORBOARD_DIR)
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'learning_rate': []
        }
        
        self.best_dice = 0.0
        
    def train_epoch(self, epoch: int) -> float:
        """
        訓練一個 epoch
        
        Args:
            epoch: 當前 epoch 數
            
        Returns:
            平均訓練損失
        """
        self.model.train()
        train_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_VALUE)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_VALUE)
                
                self.optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_train_loss = train_loss / len(self.train_loader)
        return avg_train_loss
    
    def validate(self) -> Tuple[float, float]:
        """
        驗證模型
        
        Returns:
            (平均驗證損失, 平均 Dice 係數)
        """
        self.model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for images, masks in pbar:
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Calculate Dice
                outputs_sigmoid = torch.sigmoid(outputs)
                outputs_binary = (outputs_sigmoid > 0.5).float()
                dice = dice_coeff(outputs_binary, masks)
                
                val_loss += loss.item()
                val_dice += dice.item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice.item():.4f}'})
        
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_dice = val_dice / len(self.val_loader)
        
        return avg_val_loss, avg_val_dice
    
    def save_checkpoint(self, epoch: int, dice: float) -> None:
        """
        儲存模型檢查點
        
        Args:
            epoch: 當前 epoch
            dice: Dice 係數
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dice': dice,
            'history': self.history
        }
        torch.save(checkpoint, config.MODEL_SAVE_PATH)
        print(f"✓ Model saved with Dice: {dice:.4f}")
    
    def train(self, epochs: int) -> None:
        """
        完整訓練流程
        
        Args:
            epochs: 訓練輪數
        """
        print(f"\n{'='*60}")
        print(f"開始訓練 - Device: {self.device}, AMP: {self.use_amp}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_dice = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_dice)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_dice)
            self.history['learning_rate'].append(current_lr)
            
            # Tensorboard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Dice/val', val_dice, epoch)
            self.writer.add_scalar('LearningRate', current_lr, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Dice:   {val_dice:.4f}")
            print(f"  LR:         {current_lr:.6f}")
            
            # Save best model
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.save_checkpoint(epoch, val_dice)
            
            print(f"  Best Dice:  {self.best_dice:.4f}\n")
        
        # Save training log
        self.save_training_log()
        
        # Plot and save loss curves
        self.plot_training_curves()
        
        self.writer.close()
        print(f"\n{'='*60}")
        print(f"訓練完成！最佳 Dice: {self.best_dice:.4f}")
        print(f"{'='*60}\n")
    
    def save_training_log(self) -> None:
        """
        儲存訓練記錄到 CSV
        """
        log_path = config.LOG_FILE
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_dice', 'learning_rate'])
            
            for i in range(len(self.history['train_loss'])):
                writer.writerow([
                    i + 1,
                    self.history['train_loss'][i],
                    self.history['val_loss'][i],
                    self.history['val_dice'][i],
                    self.history['learning_rate'][i]
                ])
        
        print(f"✓ Training log saved to {log_path}")
    
    def plot_training_curves(self) -> None:
        """
        繪製並儲存訓練曲線
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curve
        axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(epochs, self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Dice curve
        axes[1].plot(epochs, self.history['val_dice'], label='Val Dice', marker='o', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('Validation Dice Score')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        curve_path = os.path.join(config.OUTPUT_DIR, 'loss_curve.png')
        plt.savefig(curve_path, dpi=150)
        plt.close()
        
        print(f"✓ Training curves saved to {curve_path}")
