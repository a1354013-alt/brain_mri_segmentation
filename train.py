"""
Training module (v3.1 stable iteration)
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config


# Avoid noisy matplotlib cache permissions issues on some Windows environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bms_mpl_cache"))


class NoOpWriter:
    """
    TensorBoard fallback that keeps training runnable without the tensorboard package.

    This is intentionally tiny: only the methods used by this project are provided.
    """

    def add_scalar(self, *_args, **_kwargs) -> None:
        return None

    def close(self) -> None:
        return None

    def __getattr__(self, _name: str):
        # Future-proof: if training code starts calling other SummaryWriter methods,
        # return a no-op callable instead of crashing.
        def _noop(*_args, **_kwargs):
            return None

        return _noop


def _make_summary_writer(tensorboard_dir: Optional[Path]):
    """
    Best-effort SummaryWriter creation.

    - If tensorboard_dir is None: disable logging (NoOpWriter)
    - If tensorboard is not installed: disable logging (NoOpWriter)
    """
    if tensorboard_dir is None:
        return NoOpWriter()
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore

        return SummaryWriter(str(tensorboard_dir))
    except (ModuleNotFoundError, ImportError):
        return NoOpWriter()
    except Exception as e:
        print(f"Warning: Failed to initialize TensorBoard SummaryWriter ({e}). Falling back to NoOpWriter.")
        return NoOpWriter()


def _make_grad_scaler(device_type: str, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.amp import GradScaler as AmpGradScaler  # type: ignore

        return AmpGradScaler(device_type=device_type, enabled=True)
    except Exception:
        from torch.cuda.amp import GradScaler as CudaGradScaler  # type: ignore

        return CudaGradScaler(enabled=True)


def _autocast_ctx(device_type: str, enabled: bool):
    """
    Compatibility wrapper for autocast across torch versions.

    Newer PyTorch: torch.amp.autocast(device_type=...)\n
    Older PyTorch: torch.cuda.amp.autocast(...)\n
    """
    if not enabled:
        # Cheap no-op context manager.
        from contextlib import nullcontext

        return nullcontext()
    try:
        from torch.amp import autocast as amp_autocast  # type: ignore

        return amp_autocast(device_type=device_type, enabled=True)
    except Exception:
        from torch.cuda.amp import autocast as cuda_autocast  # type: ignore

        return cuda_autocast(enabled=True)


def dice_coeff_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    smooth = 1.0
    batch_size = pred.size(0)
    pred_flat = pred.view(batch_size, -1).float()
    target_flat = target.view(batch_size, -1).float()
    intersection = (pred_flat * target_flat).sum(dim=1)
    return (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)


def dice_coeff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return dice_coeff_per_sample(pred, target).mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        return 1 - dice_coeff(inputs, targets)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader | None,
        device: torch.device,
        output_dir: Path,
        checkpoint_path: Path,
        model_state_path: Path,
        last_checkpoint_path: Path,
        last_model_state_path: Path,
        log_file: Path,
        tensorboard_dir: Path | None,
        use_amp: bool = True,
        total_epochs: int = config.EPOCHS,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.checkpoint_path = checkpoint_path
        self.model_state_path = model_state_path
        self.last_checkpoint_path = last_checkpoint_path
        self.last_model_state_path = last_model_state_path
        self.log_file = log_file
        self.total_epochs = total_epochs

        self.use_amp = use_amp and (device.type == "cuda")

        self.dice_criterion = DiceLoss()
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=5)
        self.scaler = _make_grad_scaler(device_type=device.type, enabled=self.use_amp)

        self.writer = _make_summary_writer(tensorboard_dir)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_precision": [],
            "val_recall": [],
            "lr": [],
        }
        self.best_dice = 0.0
        self._saved_best_without_val = False

    def _compute_loss(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        dice = self.dice_criterion(logits, masks)
        bce = self.bce_criterion(logits, masks)
        return (config.DICE_WEIGHT * dice) + (config.BCE_WEIGHT * bce)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.total_epochs} [Train]")
        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()

            if self.use_amp:
                with _autocast_ctx(device_type=self.device.type, enabled=True):
                    outputs = self.model(images)
                    loss = self._compute_loss(outputs, masks)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_VALUE)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self._compute_loss(outputs, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_VALUE)
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        return total_loss / len(self.train_loader)

    def validate(self, epoch: int) -> Tuple[float, float, float, float]:
        """
        Performs validation for one epoch.
        Returns (val_loss, val_dice, val_precision, val_recall).
        If no validation loader is provided (self.val_loader is None), returns zeros.
        """
        if self.val_loader is None:
            return 0.0, 0.0, 0.0, 0.0

        self.model.eval()
        val_loss, val_dice = 0.0, 0.0
        tp, fp, fn = 0.0, 0.0, 0.0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.total_epochs} [Val]")
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(pbar):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                val_loss += self._compute_loss(outputs, masks).item()

                preds = (torch.sigmoid(outputs) > config.THRESHOLD).float()
                val_dice += dice_coeff(preds, masks).item()

                # Precision/Recall (binary, aggregated across batch)
                tp += float((preds * masks).sum().item())
                fp += float((preds * (1.0 - masks)).sum().item())
                fn += float(((1.0 - preds) * masks).sum().item())
                pbar.set_postfix({"dice": f"{val_dice / (batch_idx + 1):.4f}"})
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        return (
            val_loss / len(self.val_loader),
            val_dice / len(self.val_loader),
            float(precision),
            float(recall),
        )

    def save_checkpoints(self, epoch: int, dice: float, is_best: bool = True) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "dice": dice,
            "history": self.history,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if is_best:
            torch.save(checkpoint, self.checkpoint_path)
            torch.save(self.model.state_dict(), self.model_state_path)
            print(f"Best model saved (Dice: {dice:.4f})")
        else:
            # v3.1 stable iteration: use the explicit output paths passed in from CLI/demo.
            torch.save(checkpoint, self.last_checkpoint_path)
            torch.save(self.model.state_dict(), self.last_model_state_path)

    def train(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(self.total_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_dice, val_precision, val_recall = self.validate(epoch)

            if self.val_loader is not None:
                self.scheduler.step(val_dice)

            lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(val_dice)
            self.history["val_precision"].append(val_precision)
            self.history["val_recall"].append(val_recall)
            self.history["lr"].append(lr)

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            if self.val_loader is not None:
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Dice/val", val_dice, epoch)
                self.writer.add_scalar("Precision/val", val_precision, epoch)
                self.writer.add_scalar("Recall/val", val_recall, epoch)

            print(
                f"Epoch {epoch + 1}: Loss={train_loss:.4f}, "
                f"Val Dice={val_dice:.4f}, Val P={val_precision:.4f}, Val R={val_recall:.4f}, LR={lr:.6f}"
            )

            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.save_checkpoints(epoch, val_dice, is_best=True)
            elif self.val_loader is None and epoch == 0 and not self._saved_best_without_val:
                # No validation: still create a "best" checkpoint for consistency (use epoch-0 weights).
                self._saved_best_without_val = True
                self.save_checkpoints(epoch, val_dice, is_best=True)

            # Always keep an explicit "last" checkpoint, even when no new best model is found.
            self.save_checkpoints(epoch, val_dice, is_best=False)

        self.save_log()
        self.plot_curves()
        self.writer.close()

    def save_log(self) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_precision", "val_recall", "lr"])
            for i in range(len(self.history["train_loss"])):
                writer.writerow(
                    [
                        i + 1,
                        self.history["train_loss"][i],
                        self.history["val_loss"][i],
                        self.history["val_dice"][i],
                        self.history["val_precision"][i],
                        self.history["val_recall"][i],
                        self.history["lr"][i],
                    ]
                )

    def plot_curves(self) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as e:
            print(f"Warning: matplotlib not available ({e}). Skipping curve plot.")
            return

        if self.val_loader is not None:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            ax_loss, ax_dice = axes
        else:
            fig, ax_loss = plt.subplots(1, 1, figsize=(8, 5))
            ax_dice = None

        ax_loss.plot(self.history["train_loss"], label="Train")
        if self.val_loader is not None:
            ax_loss.plot(self.history["val_loss"], label="Val")
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.legend()

        if ax_dice is not None:
            ax_dice.plot(self.history["val_dice"], label="Val Dice", color="green")
            ax_dice.set_title("Dice Score")
            ax_dice.set_xlabel("Epoch")
            ax_dice.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
