import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_results_with_uncertainty(image, mask, prediction, uncertainty, title="Brain MRI Tumor Segmentation"):
    """
    繪製影像、真實標籤、預測結果與不確定性地圖。
    """
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.title("Original MRI (FLAIR)")
    plt.imshow(image[0], cmap='gray') # 顯示 FLAIR 序列
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask[0], cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title("Model Prediction")
    plt.imshow(prediction[0], cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title("Uncertainty Map")
    plt.imshow(uncertainty[0], cmap='jet')
    plt.colorbar()
    plt.axis('off')
    
    plt.suptitle(title)
    plt.show()

def mc_dropout_inference(model, image_tensor, n_iterations=10, device='cpu'):
    """
    使用 Monte Carlo Dropout 進行推論，估計預測的不確定性。
    """
    model.train() # 保持 Dropout 開啟
    preds = []
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        for _ in range(n_iterations):
            output = torch.sigmoid(model(image_tensor))
            preds.append(output.cpu().numpy())
            
    preds = np.array(preds) # (n_iterations, B, C, H, W)
    mean_pred = np.mean(preds, axis=0)
    uncertainty = np.var(preds, axis=0) # 使用變異數作為不確定性指標
    
    prediction = (mean_pred > 0.5).astype(np.float32)
    return prediction, uncertainty
