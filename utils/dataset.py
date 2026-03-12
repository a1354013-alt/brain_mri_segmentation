"""
BraTS Dataset implementation with enhanced preprocessing and I/O optimization
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from skimage.transform import resize
from pathlib import Path
from typing import List, Optional, Tuple, Callable


class BraTSDataset(Dataset):
    """
    針對 BraTS 資料集的 PyTorch Dataset 類別
    
    優化點：
    1. 初始化時檢查資料完整性
    2. 預先計算並快取腫瘤切片索引 (I/O 優化)
    3. 訓練時隨機抽樣含腫瘤切片，驗證時固定使用最佳切片
    """
    def __init__(
        self, 
        data_dir: Path, 
        patient_ids: List[str], 
        image_size: int = 128,
        transform: Optional[Callable] = None, 
        mode: str = 'train',
        seed: int = 42
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        
        # 1. 完整性檢查與快取初始化
        self.valid_patient_ids = []
        self.patient_cache = {} # 快取 slice 資訊
        
        self._prepare_dataset(patient_ids)

    def _prepare_dataset(self, patient_ids: List[str]) -> None:
        """
        檢查資料完整性並預先計算切片索引
        """
        skipped_patients = []
        modalities = ['flair', 't1', 't1ce', 't2']
        
        print(f"🔍 Scanning {len(patient_ids)} patients for data integrity...")
        
        for pid in patient_ids:
            p_path = self.data_dir / pid
            missing_files = []
            
            # 檢查影像檔
            for mod in modalities:
                if not (p_path / f"{pid}_{mod}.nii.gz").exists():
                    missing_files.append(f"{pid}_{mod}.nii.gz")
            
            # 檢查標籤檔
            mask_file = p_path / f"{pid}_seg.nii.gz"
            if not mask_file.exists():
                missing_files.append(f"{pid}_seg.nii.gz")
            
            if missing_files:
                skipped_patients.append((pid, missing_files))
                continue
            
            # 2. I/O 優化：預先讀取 seg 找切片
            try:
                mask_volume = nib.load(str(mask_file)).get_fdata()
                # Whole Tumor (WT) 二元化
                mask_binary = (mask_volume > 0).astype(np.uint8)
                
                # 找出所有含腫瘤的切片索引
                tumor_counts = np.sum(mask_binary, axis=(0, 1))
                tumor_indices = np.where(tumor_counts > 0)[0].tolist()
                
                if not tumor_indices:
                    # 如果完全沒腫瘤，取中間切片
                    best_idx = mask_volume.shape[2] // 2
                    tumor_indices = [best_idx]
                else:
                    best_idx = int(np.argmax(tumor_counts))
                
                self.patient_cache[pid] = {
                    'tumor_indices': tumor_indices,
                    'best_idx': best_idx
                }
                self.valid_patient_ids.append(pid)
                
            except Exception as e:
                skipped_patients.append((pid, [f"Error reading: {str(e)}"]))

        # 列印 Summary
        if skipped_patients:
            print(f"⚠️  Skipped {len(skipped_patients)} incomplete patients:")
            for pid, files in skipped_patients:
                print(f"   - {pid}: Missing {', '.join(files)}")
        
        print(f"✅ Dataset ready: {len(self.valid_patient_ids)} valid patients.")

    def __len__(self) -> int:
        return len(self.valid_patient_ids)

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """
        影像正規化：percentile clipping + z-score normalization
        """
        p1, p99 = np.percentile(img, (1, 99))
        img_clipped = np.clip(img, p1, p99)
        
        mean = np.mean(img_clipped)
        std = np.std(img_clipped)
        
        if std > 1e-8:
            return (img_clipped - mean) / std
        return img_clipped - mean

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pid = self.valid_patient_ids[idx]
        p_path = self.data_dir / pid
        cache = self.patient_cache[pid]
        
        # 3. 切片策略
        if self.mode == 'train':
            # 隨機抽一個含腫瘤的切片
            slice_idx = self.rng.choice(cache['tumor_indices'])
        else:
            # 驗證時固定使用腫瘤最多的切片
            slice_idx = cache['best_idx']
        
        # 載入 4 通道影像
        modalities = ['flair', 't1', 't1ce', 't2']
        images = []
        
        for mod in modalities:
            img_path = p_path / f"{pid}_{mod}.nii.gz"
            # 優化：只載入需要的 slice (nibabel 支援 proxy 讀取)
            img_proxy = nib.load(str(img_path))
            img_slice = img_proxy.dataobj[:, :, slice_idx]
            
            img_slice = self._normalize_image(np.array(img_slice))
            img_slice = resize(img_slice, (self.image_size, self.image_size), preserve_range=True)
            images.append(img_slice)
            
        image = np.stack(images, axis=0)
        
        # 載入標籤
        mask_path = p_path / f"{pid}_seg.nii.gz"
        mask_proxy = nib.load(str(mask_path))
        mask_slice = mask_proxy.dataobj[:, :, slice_idx]
        mask = resize(np.array(mask_slice), (self.image_size, self.image_size), order=0, preserve_range=True)
        # Whole Tumor (WT) 二元分割
        mask = (mask > 0).astype(np.float32)
        
        if self.transform:
            image_hwc = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image_hwc, mask=mask)
            image = np.transpose(augmented['image'], (2, 0, 1))
            mask = augmented['mask']
        
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float().unsqueeze(0)
