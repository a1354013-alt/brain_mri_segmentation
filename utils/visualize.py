"""
Visualization utilities with MC Dropout inference
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import config


def mc_dropout_inference(
    model: nn.Module, 
    image_tensor: torch.Tensor, 
    n_iterations: int = 20, 
    device: torch.device = config.DEVICE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Monte Carlo Dropout 進行推論，估計預測的不確定性
    
    正確做法：
    1. model.eval() - 設定模型為評估模式
    2. 只將 Dropout 層設為 train() - 保持 dropout 啟用
    3. BatchNorm 保持在 eval 模式 - 不污染統計量
    
    Args:
        model: 訓練好的模型
        image_tensor: 輸入影像 (B, C, H, W)
        n_iterations: MC Dropout 迭代次數
        device: 運算裝置
        
    Returns:
        (prediction, uncertainty) tuple
    """
    # 設定模型為評估模式
    model.eval()
    
    # 只啟用 Dropout 層
    model.enable_dropout()
    
    preds = []
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        for _ in range(n_iterations):
            output = torch.sigmoid(model(image_tensor))
            preds.append(output.cpu().numpy())
            
    preds = np.array(preds)  # (n_iterations, B, C, H, W)
    mean_pred = np.mean(preds, axis=0)
    uncertainty = np.var(preds, axis=0)  # 使用變異數作為不確定性指標
    
    prediction = (mean_pred > 0.5).astype(np.float32)
    return prediction, uncertainty


def plot_results_with_uncertainty(
    image: np.ndarray, 
    mask: np.ndarray, 
    prediction: np.ndarray, 
    uncertainty: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Brain MRI Tumor Segmentation"
) -> None:
    """
    繪製影像、真實標籤、預測結果、不確定性地圖與疊加圖
    
    Args:
        image: 原始影像 (C, H, W)
        mask: 真實標籤 (1, H, W)
        prediction: 預測結果 (1, H, W)
        uncertainty: 不確定性地圖 (1, H, W)
        save_path: 儲存路徑
        title: 圖表標題
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # 原圖 (FLAIR)
    axes[0].imshow(image[0], cmap='gray')
    axes[0].set_title("Original MRI (FLAIR)", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(mask[0], cmap='gray')
    axes[1].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(prediction[0], cmap='gray')
    axes[2].set_title("Prediction", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Uncertainty Map (使用 viridis 而非 jet)
    im = axes[3].imshow(uncertainty[0], cmap='viridis')
    axes[3].set_title("Uncertainty Map", fontsize=12, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    # Overlay (原圖 + 預測)
    overlay = np.stack([image[0], image[0], image[0]], axis=-1)
    overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
    
    # 將預測結果疊加為紅色
    mask_overlay = prediction[0] > 0.5
    overlay[mask_overlay, 0] = 1.0  # Red channel
    
    axes[4].imshow(overlay)
    axes[4].set_title("Overlay (Red: Prediction)", fontsize=12, fontweight='bold')
    axes[4].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_uncertainty_histogram(
    uncertainty: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    繪製不確定性分佈直方圖
    
    Args:
        uncertainty: 不確定性地圖 (1, H, W)
        save_path: 儲存路徑
    """
    plt.figure(figsize=(10, 6))
    plt.hist(uncertainty.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Uncertainty (Variance)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Uncertainty Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Uncertainty histogram saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
