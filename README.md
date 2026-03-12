# 腦部 MRI 腫瘤分割專題 (Brain MRI Tumor Segmentation Project) - 進階優化版

## 專案簡介

本專案是基於經典 U-Net 架構的進階版本，旨在建立一個高效且具備**可信度評估**的深度學習模型，從腦部核磁共振影像（MRI）中自動分割出腫瘤區域。我們整合了三項關鍵優化技術，使專案在技術深度和臨床應用潛力上更具吸引力。

## 專案核心優化亮點

| 優化方向 | 技術細節 | 專案價值 |
| :--- | :--- | :--- |
| **1. 注意力機制** | **Attention U-Net**：在編碼器與解碼器的跳躍連接中引入 **Attention Gate**。 | 讓模型自動聚焦於腫瘤區域，有效抑制背景雜訊，提升分割精準度。展現對進階網路架構的掌握。 |
| **2. 多模態融合** | **4 通道輸入**：同時處理 FLAIR, T1, T1ce, T2 四種 MRI 序列。 | 模擬真實臨床數據，利用不同序列的互補資訊，更全面地描繪腫瘤邊界，提升模型魯棒性。 |
| **3. 不確定性估計** | **Monte Carlo Dropout (MC Dropout)**：在推論時保持 Dropout 開啟，計算多次預測的變異數。 | 輸出**不確定性地圖**，評估模型對每個像素預測的可信度，符合醫療 AI 對安全性和可解釋性的高要求。 |

## 專案目標

1.  **資料處理**：開發模組以載入和預處理 BraTS 資料集中的 **4 通道 NIfTI** 格式多模態 MRI 影像。
2.  **模型實作**：實作 **Attention U-Net** 模型架構。
3.  **訓練與評估**：使用 PyTorch 框架進行模型訓練，並採用 **Dice Loss** 作為損失函數。
4.  **推論與視覺化**：提供 **MC Dropout 推論**與視覺化工具，直觀展示原始影像、真實標籤、預測結果與**不確定性地圖**。

## 專案結構

```
brain_mri_segmentation/
├── data/                       # 存放 BraTS 資料集 (需自行下載)
├── models/
│   └── attention_unet.py       # Attention U-Net 模型架構實作 (包含 Attention Gate)
├── utils/
│   ├── dataset.py              # 資料集載入與預處理類別 (支援 4 通道輸入)
│   └── visualize.py            # 視覺化工具 (支援 MC Dropout 不確定性地圖)
├── train.py                    # 模型訓練與評估腳本 (使用 Attention U-Net)
├── main.py                     # 專案主入口與執行範例
└── requirements.txt            # Python 依賴套件清單
└── README.md                   # 專案說明文件
```

## 環境建置與執行

請參考原版說明文件中的步驟，確保您已安裝所有依賴套件並準備好 BraTS 資料集。

### 核心變動

-   **模型**：已從 `models/unet.py` 替換為 `models/attention_unet.py`。
-   **輸入**：`utils/dataset.py` 現在會將 FLAIR, T1, T1ce, T2 堆疊為 **4 個輸入通道**。
-   **推論**：`utils/visualize.py` 中的 `mc_dropout_inference` 函數用於執行不確定性估計。

執行主腳本：
```bash
python main.py
```
`main.py` 將會展示如何初始化 4 通道 Attention U-Net 模型，並模擬 MC Dropout 推論的結果。

## 進階方向建議 (作品集深化)

除了已實作的三大優化，若要進一步提升作品集吸引力，建議：

1.  **定量分析報告**：在 `main.py` 中加入計算分割結果的**腫瘤體積 (Tumor Volume)** 和 **Dice Coefficient**，並將這些數值輸出為報告。
2.  **網頁部署**：使用 **Streamlit** 或 **Gradio** 建立一個簡單的 Web 介面，讓使用者可以上傳單張 MRI 影像並即時看到分割結果、不確定性地圖和定量報告。
3.  **3D 視覺化**：如果能處理 3D 數據，使用 `matplotlib` 或 `plotly` 嘗試將分割結果進行 3D 重建，這將是極具視覺衝擊力的展示。
