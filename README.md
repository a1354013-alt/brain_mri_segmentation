# Brain MRI Tumor Segmentation with Attention U-Net

基於 **Attention U-Net** 和 **Monte Carlo Dropout** 的腦部 MRI 腫瘤分割專案，支援不確定性估計。

---

## 📋 專案簡介

本專案實作了一個端到端的醫學影像分割系統，使用 **Attention U-Net** 架構對 BraTS (Brain Tumor Segmentation) 資料集進行腦腫瘤分割。專案特色包括：

- **Attention U-Net 架構**：整合注意力機制，提升分割精度
- **Monte Carlo Dropout**：推論時估計模型不確定性
- **混合精度訓練 (AMP)**：加速訓練並降低記憶體使用
- **智能切片選擇**：自動選擇含腫瘤的切片進行訓練
- **完整的訓練監控**：Tensorboard、訓練曲線、CSV 日誌
- **CLI 介面**：簡潔的命令列操作

---

## 🏗️ 模型架構

### Attention U-Net

Attention U-Net 在標準 U-Net 的基礎上加入了 **Attention Gate**，能夠自動學習關注重要的特徵區域，抑制無關背景資訊。

```
                    Encoder                  Decoder
                    -------                  -------
Input (4, 128, 128)
    │
    ├─► ConvBlock(64) ────────────────► AttentionGate ──► Concat ──► ConvBlock(64)
    │       │                                                              │
    │   MaxPool                                                            │
    │       │                                                              │
    ├─► ConvBlock(128) ───────────► AttentionGate ──► Concat ──► ConvBlock(128)
    │       │                                                              │
    │   MaxPool                                                            │
    │       │                                                              │
    ├─► ConvBlock(256) ──────► AttentionGate ──► Concat ──► ConvBlock(256)
    │       │                                                              │
    │   MaxPool                                                            │
    │       │                                                              │
    ├─► ConvBlock(512) ─► AttentionGate ──► Concat ──► ConvBlock(512)
    │       │                                              │
    │   MaxPool                                            │
    │       │                                              │
    └─► ConvBlock(1024) ──────────────────────────────────┘
                │
                └─► Conv2d(1) ──► Output (1, 128, 128)
```

**關鍵特性**：
- **輸入**：4 通道 MRI 影像 (FLAIR, T1, T1ce, T2)
- **輸出**：單通道二元分割遮罩
- **Dropout**：每層加入 Dropout (p=0.2) 支援 MC Dropout
- **BatchNorm**：加速訓練並穩定梯度

---

## 🚀 快速開始

### 1. 環境安裝

```bash
# 建立虛擬環境 (推薦)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

**系統需求**：
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (可選，用於 GPU 加速)

---

### 2. 資料集準備

#### 方法 1：自動下載 (推薦)

```bash
python scripts/download_brats.py --auto
```

#### 方法 2：手動下載

1. 訪問 [BraTS 官方網站](https://www.med.upenn.edu/cbica/brats2020/data.html)
2. 註冊並下載 BraTS2020 Training Data
3. 解壓縮至 `data/Brats/` 目錄

#### 方法 3：Kaggle API

```bash
# 安裝 Kaggle CLI
pip install kaggle

# 設定 API Token (參考 scripts/download_brats.py 中的說明)
kaggle datasets download -d awsaf49/brats20-dataset-training-validation
unzip brats20-dataset-training-validation.zip -d data/
```

**資料集結構**：

```
data/Brats/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii.gz
│   ├── BraTS20_Training_001_t1.nii.gz
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_001_t2.nii.gz
│   └── BraTS20_Training_001_seg.nii.gz
├── BraTS20_Training_002/
│   └── ...
└── ...
```

---

### 3. 訓練模型

```bash
python main.py train
```

**訓練輸出**：
- `outputs/best_model.pth` - 最佳模型權重
- `outputs/training_log.csv` - 訓練記錄
- `outputs/loss_curve.png` - 訓練曲線
- `outputs/tensorboard/` - Tensorboard 日誌

**監控訓練**：

```bash
tensorboard --logdir outputs/tensorboard
```

---

### 4. 推論

```bash
# 對第一個病人進行推論
python main.py infer

# 對特定病人進行推論
python main.py infer --patient_id BraTS20_Training_001
```

**推論輸出**：
- `outputs/inference/{patient_id}_segmentation.png` - 分割結果視覺化

---

### 5. 快速測試

```bash
# 使用少量資料跑 1 epoch 測試流程
python main.py demo
```

---

## 📊 訓練方式

### 訓練流程

本專案採用以下訓練策略：

1. **資料前處理**
   - **智能切片選擇**：自動選擇含腫瘤最多的切片
   - **Percentile Clipping**：裁剪 1%-99% 的極端值
   - **Z-score Normalization**：標準化影像強度

2. **損失函數**
   - **Dice Loss**：直接優化 Dice 係數
   - **Per-sample 計算**：對每個樣本計算 Dice 後取平均

3. **優化器與調度**
   - **Optimizer**：Adam (lr=1e-4, weight_decay=1e-5)
   - **Scheduler**：ReduceLROnPlateau (監控 Dice，patience=5)
   - **Gradient Clipping**：限制梯度範數為 1.0

4. **訓練技巧**
   - **混合精度訓練 (AMP)**：加速訓練並降低記憶體
   - **Model Checkpointing**：自動儲存最佳模型
   - **Early Stopping**：透過 LR Scheduler 實現

### 訓練參數 (config.py)

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `IMAGE_SIZE` | 128 | 影像大小 |
| `BATCH_SIZE` | 4 | Batch 大小 |
| `EPOCHS` | 50 | 訓練輪數 |
| `LEARNING_RATE` | 1e-4 | 學習率 |
| `DROPOUT_P` | 0.2 | Dropout 機率 |
| `GRAD_CLIP_VALUE` | 1.0 | 梯度裁剪閾值 |

---

## 🔍 推論方式

### Monte Carlo Dropout 不確定性估計

**MC Dropout** 是一種簡單但有效的不確定性估計方法。在推論時：

1. **保持 Dropout 啟用**：讓模型產生隨機性
2. **多次前向傳播**：對同一輸入進行 N 次推論 (預設 20 次)
3. **計算統計量**：
   - **預測**：N 次結果的平均值
   - **不確定性**：N 次結果的變異數

**實作細節**：

```python
# 正確的 MC Dropout 實作
model.eval()              # 設定模型為評估模式
model.enable_dropout()    # 只啟用 Dropout 層，保持 BatchNorm 在 eval 模式

predictions = []
for _ in range(n_iterations):
    with torch.no_grad():
        output = model(image)
        predictions.append(output)

mean = np.mean(predictions, axis=0)      # 預測
uncertainty = np.var(predictions, axis=0) # 不確定性
```

**不確定性的意義**：
- **高不確定性區域**：模型對該區域的預測不穩定，可能需要人工審查
- **低不確定性區域**：模型對該區域的預測有信心

---

## 📈 範例結果

### 訓練曲線

訓練過程中會自動生成訓練曲線：

![Training Curve](outputs/loss_curve.png)

### 分割結果

推論輸出包含 5 個子圖：

1. **Original MRI (FLAIR)**：原始 FLAIR 序列影像
2. **Ground Truth**：真實標籤
3. **Prediction**：模型預測結果
4. **Uncertainty Map**：不確定性熱圖 (使用 viridis colormap)
5. **Overlay**：原圖疊加預測結果 (紅色表示腫瘤)

---

## 🧪 MC Dropout 不確定性說明

### 為什麼需要不確定性估計？

在醫學影像分割中，**不確定性估計**至關重要：

- **輔助診斷**：高不確定性區域提示醫生需要額外注意
- **主動學習**：選擇不確定性高的樣本進行標註
- **模型可靠性**：評估模型對特定樣本的信心程度

### MC Dropout 的優勢

相比其他不確定性估計方法（如 Ensemble、Bayesian Neural Networks），MC Dropout 具有：

- **簡單**：無需修改模型架構或訓練流程
- **高效**：推論時僅需多次前向傳播
- **理論基礎**：等價於近似貝葉斯推論

### 實驗觀察

在 BraTS 資料集上，我們觀察到：

- **腫瘤邊界**：通常具有較高不確定性
- **小腫瘤**：比大腫瘤具有更高不確定性
- **影像品質**：低品質影像會導致整體不確定性上升

---

## 📁 專案結構

```
brain_mri_segmentation/
├── data/                       # 資料集目錄
│   └── Brats/                  # BraTS 資料
├── models/                     # 模型定義
│   ├── __init__.py
│   └── attention_unet.py       # Attention U-Net 實作
├── utils/                      # 工具函數
│   ├── __init__.py
│   ├── dataset.py              # 資料集類別
│   └── visualize.py            # 視覺化與 MC Dropout
├── scripts/                    # 腳本
│   ├── __init__.py
│   └── download_brats.py       # 資料集下載腳本
├── outputs/                    # 訓練輸出
│   ├── best_model.pth          # 最佳模型
│   ├── training_log.csv        # 訓練記錄
│   ├── loss_curve.png          # 訓練曲線
│   ├── tensorboard/            # Tensorboard 日誌
│   └── inference/              # 推論結果
├── main.py                     # CLI 主程式
├── train.py                    # 訓練模組
├── config.py                   # 配置檔案
├── requirements.txt            # 依賴清單
├── .gitignore                  # Git 忽略檔案
└── README.md                   # 本文件
```

---

## 🎯 未來改進方向

### 模型架構
- [ ] 實作 3D U-Net 處理完整 3D 體積
- [ ] 嘗試 Transformer-based 架構 (如 UNETR)
- [ ] 多尺度特徵融合

### 訓練策略
- [ ] 加入 Focal Loss 處理類別不平衡
- [ ] 實作 Deep Supervision
- [ ] 使用 Cross-validation

### 資料增強
- [ ] 加入 Elastic Deformation
- [ ] 實作 MixUp / CutMix
- [ ] 使用 Albumentations 進階增強

### 不確定性估計
- [ ] 實作 Deep Ensemble
- [ ] 嘗試 Test-Time Augmentation (TTA)
- [ ] 比較不同不確定性指標 (Entropy, Mutual Information)

### 工程優化
- [ ] 支援分散式訓練 (DDP)
- [ ] 實作模型量化與剪枝
- [ ] 部署為 REST API 服務

### 評估指標
- [ ] 加入 Hausdorff Distance
- [ ] 計算 Sensitivity / Specificity
- [ ] 實作 Surface Dice

---

## 📝 引用

如果本專案對您的研究有幫助，請考慮引用：

```bibtex
@misc{brain_mri_segmentation,
  title={Brain MRI Tumor Segmentation with Attention U-Net},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/brain_mri_segmentation}}
}
```

**相關論文**：

- **Attention U-Net**: Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas." MIDL 2018.
- **U-Net**: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
- **MC Dropout**: Gal and Ghahramani. "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." ICML 2016.

---

## 📄 授權

本專案採用 MIT License。詳見 [LICENSE](LICENSE) 檔案。

---

## 🙏 致謝

- **BraTS Challenge**：提供高品質的腦腫瘤分割資料集
- **PyTorch**：強大的深度學習框架
- **Open Source Community**：眾多優秀的開源專案

---

## 📧 聯絡方式

如有任何問題或建議，歡迎透過以下方式聯絡：

- **Email**: your.email@example.com
- **GitHub Issues**: [提交 Issue](https://github.com/yourusername/brain_mri_segmentation/issues)

---

**⭐ 如果本專案對您有幫助，請給我們一個 Star！**
