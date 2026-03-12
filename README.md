# Brain MRI Tumor Segmentation (v2.1)

基於 **Attention U-Net** 的腦部 MRI 腫瘤分割專案，支援不確定性估計與高效 I/O。

---

## 📋 專案更新亮點 (v2.1)

- **Pathlib 全面化**：解決不同工作目錄下的路徑引用問題。
- **Patient-level Split**：資料切分以病人為單位，並透過固定隨機種子 (`RANDOM_SEED`) 確保可重現性。
- **Dataset I/O 優化**：
  - 初始化時自動檢查資料完整性（需包含 FLAIR 與 Seg）。
  - 預先計算並快取腫瘤切片索引，大幅提升訓練速度。
  - **訓練策略**：從含腫瘤切片中隨機抽樣。
  - **驗證策略**：固定使用腫瘤像素最多的切片，確保評估穩定。
- **任務定義**：預設執行 **Whole Tumor (WT)** 二元分割（Mask > 0）。
- **Checkpoint 雙重儲存**：
  - `best_checkpoint.pth`：包含模型、優化器、排程器與歷史記錄。
  - `best_model_state.pth`：僅包含模型權重，便於部署與推論。
- **MC Dropout 強化**：
  - 支援 `var` (變異數) 與 `entropy` (預測熵) 兩種不確定性指標。
  - 嚴格控制 BatchNorm 不受推論隨機性影響。
- **輸入尺寸保護**：模型內建 Padding 機制，支援非 2 的冪次方尺寸輸入。

---

## 🚀 快速開始

### 1. 環境安裝
```bash
pip install -r requirements.txt
```

### 2. 訓練與推論
```bash
# 訓練模式 (自動執行 Patient-level shuffle split)
python main.py train

# 推論模式 (支援不同不確定性指標)
python main.py infer --uncertainty entropy

# Demo 模式 (輸出隔離至 outputs/demo/，不覆蓋正式結果)
python main.py demo
```

---

## 🏗️ 技術細節

### 資料切分 (Patient-level Split)
為了避免資料洩漏，我們在病人層級進行切分。使用 `np.random.default_rng(config.RANDOM_SEED)` 對病人 ID 進行洗牌，確保每次運行得到的訓練/驗證集完全一致。

### MC Dropout 不確定性
推論時透過 `enable_dropout(model)` 僅開啟 Dropout 層，同時保持 `model.eval()` 狀態以固定 BatchNorm 的統計量。
- **Variance**: 衡量預測值的離散程度。
- **Entropy**: 衡量預測機率的分佈混亂度，適合量化分類不確定性。

---

## 📁 專案結構
```
brain_mri_segmentation/
├── config.py              # 專案根目錄與路徑配置
├── main.py                # CLI 入口
├── train.py               # 訓練邏輯與 Checkpoint 管理
├── models/
│   └── attention_unet.py  # 具備尺寸保護的模型
├── utils/
│   ├── dataset.py         # 高效 I/O 與完整性檢查
│   └── visualize.py       # 強化版視覺化與 MC Dropout
└── outputs/               # 正式輸出
    └── demo/              # Demo 模式專屬輸出
```
