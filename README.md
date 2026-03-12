# Brain MRI Tumor Segmentation (v2.2)

基於 **Attention U-Net** 的腦部 MRI 腫瘤分割專案，支援不確定性估計與高效 I/O。

---

## 📋 專案更新亮點 (v2.2)

- **資料完整性強化**：本專案嚴格要求 **4 模態 (flair, t1, t1ce, t2) + seg**。`dataset.py` 會自動過濾不完整的病人資料。
- **多 Worker RNG 修正**：修正了 DataLoader 在多 worker 模式下隨機切片重複的問題。透過 `worker_init_fn` 確保每個 worker 擁有獨立且可重現的隨機種子。
- **全面參數 Config 化**：`THRESHOLD` (預設 0.5) 與 `MC_ITERATIONS` (預設 20) 已移至 `config.py`，修改後會即時影響訓練評估與推論行為。
- **跨平台資料對齊**：`download_brats.py` 改用 Python `zipfile` 進行解壓，並具備自動結構對齊功能，確保資料夾結構符合 `data/Brats/<patient_id>/...`。
- **Checkpoint 一致性**：`best_checkpoint.pth` 現在完整包含 `scheduler_state_dict`，確保訓練可恢復性。
- **模型尺寸保護**：內建 Padding Helper，自動對齊 Encoder/Decoder 尺寸，避免非 2 的冪次方輸入導致崩潰。
- **空資料保護**：所有 CLI 命令均加入空資料檢查，避免 IndexError。

---

## 🏗️ 技術細節

### 1. 任務定義
本專案執行 **Whole Tumor (WT)** 二元分割。所有標籤值大於 0 的區域均被視為腫瘤區域。

### 2. 切分策略
採用 **Patient-level Split**。首先對所有病人 ID 進行隨機洗牌 (`shuffle`)，再依比例切分。這確保了同一病人的不同切片不會同時出現在訓練集與驗證集中，避免資料洩漏。

### 3. 可重現性說明
專案預設開啟 `torch.backends.cudnn.deterministic = True`。這能確保實驗結果的可重現性，但在某些 GPU 環境下可能會略微降低運算速度。

### 4. Resize 策略
- **影像 (Image)**：使用 `order=1` (Bilinear) 並開啟 `anti_aliasing=True` 以保持細節。
- **標籤 (Mask)**：使用 `order=0` (Nearest Neighbor) 以確保標籤值保持為二元。

---

## 🚀 快速開始

### 1. 環境安裝
```bash
pip install -r requirements.txt
```

### 2. 資料準備
```bash
# 自動下載並對齊結構
python scripts/download_brats.py --auto
```

### 3. 執行命令
```bash
# 訓練模式
python main.py train

# 推論模式
python main.py infer --patient_id BraTS20_Training_001 --uncertainty entropy

# Demo 模式 (輸出隔離至 outputs/demo/)
python main.py demo
```

---

## 📁 專案結構
```
brain_mri_segmentation/
├── config.py              # 全面參數配置
├── main.py                # CLI 入口 (含 Worker RNG 修正)
├── train.py               # 訓練邏輯 (含 Scheduler Checkpoint)
├── models/
│   └── attention_unet.py  # 具備尺寸對齊保護的模型
├── utils/
│   ├── dataset.py         # 4 模態檢查與 RNG 優化
│   └── visualize.py       # 參數化視覺化
└── scripts/
    └── download_brats.py  # 跨平台解壓與結構對齊
```
