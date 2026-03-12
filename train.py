"""
Training module with enhanced checkpointing and Pathlib support
"""
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List
import config


def dice_coeff_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    smooth = 1.0
    batch_size = pred.size(0)
    pred_flat = pred.view(batch_size, -1).float()
    target_flat = target.view(batch_size, -1).float()
    intersection = (pred_flat * target_flat).sum(dim=1)
    return (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)


def dice_coeff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return dice_coeff_per_sample(pred, target).mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        return 1 - dice_coeff(inputs, targets)


class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        device: torch.device,
        output_dir: Path,
        checkpoint_path: Path,
        model_state_path: Path,
        use_amp: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.checkpoint_path = checkpoint_path
        self.model_state_path = model_state_path
        self.use_amp = use_amp
        
        self.criterion = DiceLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        self.scaler = GradScaler() if use_amp else None
        
        self.writer = SummaryWriter(str(output_dir / "tensorboard"))
        self.history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'lr': []}
        self.best_dice = 0.0
        
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_VALUE)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_VALUE)
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                val_loss += self.criterion(outputs, masks).item()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_dice += dice_coeff(preds, masks).item()
        return val_loss / len(self.val_loader), val_dice / len(self.val_loader)
    
    def save_checkpoints(self, epoch: int, dice: float) -> None:
        # 1. 儲存完整 Checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dice': dice,
            'history': self.history
        }
        torch.save(checkpoint, self.checkpoint_path)
        
        # 2. 儲存純模型權重
        torch.save(self.model.state_dict(), self.model_state_path)
        print(f"✓ Checkpoints saved to {self.output_dir}")
    
    def train(self, epochs: int) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_dice = self.validate()
            self.scheduler.step(val_dice)
            
            lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_dice)
            self.history['lr'].append(lr)
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Dice/val', val_dice, epoch)
            
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val Dice={val_dice:.4f}, LR={lr:.6f}")
            
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.save_checkpoints(epoch, val_dice)
        
        self.save_log()
        self.plot_curves()
        self.writer.close()

    def save_log(self) -> None:
        log_path = self.output_dir / "training_log.csv"
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_dice', 'lr'])
            for i in range(len(self.history['train_loss'])):
                writer.writerow([i+1, self.history['train_loss'][i], self.history['val_loss'][i], self.history['val_dice'][i], self.history['lr'][i]])

    def plot_curves(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_title('Loss'); axes[0].legend()
        axes[1].plot(self.history['val_dice'], label='Val Dice', color='green')
        axes[1].set_title('Dice'); axes[1].legend()
        plt.savefig(self.output_dir / "loss_curve.png")
        plt.close()
