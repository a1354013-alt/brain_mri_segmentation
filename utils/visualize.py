"""
Visualization utilities with Alpha Blending and MC Dropout (v3.1 stable iteration).
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import config


def enable_dropout(model: nn.Module) -> None:
    """
    只將 nn.Dropout 與 nn.Dropout2d 設為 train 模式 (v3.1 stable iteration)
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()


def mc_dropout_inference(
    model: nn.Module,
    image_tensor: torch.Tensor,
    n_iterations: int = config.MC_ITERATIONS,
    device: Optional[torch.device] = None,
    method: str = "var",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MC Dropout 推論 (v3.1 stable iteration)
    """
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

    preds = np.array(preds)  # (N, B, C, H, W)
    mean_pred = np.mean(preds, axis=0)

    if method == "entropy":
        # Predictive Entropy: -p*log(p) - (1-p)*log(1-p)
        p = np.clip(mean_pred, 1e-8, 1.0 - 1e-8)
        uncertainty = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    else:
        # Variance
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
    """
    視覺化結果：原圖、GT、預測、不確定性、疊加圖 (v3.1 stable iteration)
    """
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

    # Overlay (v3.1 stable iteration: alpha blending)
    img_norm = (image[0] - image[0].min()) / (image[0].max() - image[0].min() + 1e-8)
    overlay = np.stack([img_norm] * 3, axis=-1)

    # 建立紅色遮罩
    red_mask = np.zeros_like(overlay)
    red_mask[prediction[0] > 0, 0] = 1.0

    # Alpha Blending: 僅在預測區域進行混合
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
