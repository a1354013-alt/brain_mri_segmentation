"""
Visualization utilities for qualitative segmentation review and MC Dropout uncertainty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import config


def enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers during evaluation for MC Dropout inference."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()


def mc_dropout_inference(
    model: nn.Module,
    image_tensor: torch.Tensor,
    n_iterations: int = config.MC_ITERATIONS,
    device: Optional[torch.device] = None,
    method: str = "var",
) -> tuple[np.ndarray, np.ndarray]:
    """Run MC Dropout inference and return `(prediction, uncertainty)` arrays."""
    if device is None:
        device = config.DEVICE

    model.eval()
    enable_dropout(model)

    preds = []
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        for _ in range(n_iterations):
            output = torch.sigmoid(model(image_tensor))
            preds.append(output.cpu().numpy())

    preds = np.array(preds)
    mean_pred = np.mean(preds, axis=0)

    if method == "entropy":
        p = np.clip(mean_pred, 1e-8, 1.0 - 1e-8)
        uncertainty = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    else:
        uncertainty = np.var(preds, axis=0)

    prediction = (mean_pred > config.THRESHOLD).astype(np.float32)
    return prediction, uncertainty


def plot_results_with_uncertainty(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Brain MRI Tumor Segmentation",
) -> None:
    """Render MRI, ground truth, prediction, uncertainty, and overlay panels."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"Error: matplotlib not available ({e}). Cannot render plots.")
        return

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(image[0], cmap="gray")
    axes[0].set_title("MRI (FLAIR)")
    axes[0].axis("off")

    axes[1].imshow(mask[0], cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(prediction[0], cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    im = axes[3].imshow(uncertainty[0], cmap="viridis")
    axes[3].set_title("Uncertainty Map")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    img_norm = (image[0] - image[0].min()) / (image[0].max() - image[0].min() + 1e-8)
    overlay = np.stack([img_norm] * 3, axis=-1)
    red_mask = np.zeros_like(overlay)
    red_mask[prediction[0] > 0, 0] = 1.0

    alpha = config.OVERLAY_ALPHA
    mask_idx = prediction[0] > 0
    overlay[mask_idx] = (1 - alpha) * overlay[mask_idx] + alpha * red_mask[mask_idx]

    axes[4].imshow(overlay)
    axes[4].set_title("Overlay (Alpha Blending)")
    axes[4].axis("off")

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
