"""
BraTS Dataset implementation with enhanced preprocessing
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from skimage.transform import resize
from typing import List, Optional, Tuple, Callable


class BraTSDataset(Dataset):
    """
    針對 BraTS 資料集的 PyTorch Dataset 類別
    支援載入多模態影像 (FLAIR, T1, T1ce, T2) 與標籤
    
    Args:
        data_dir: 資料集根目錄
        patient_ids: 病人 ID 列表
        image_size: 影像大小
        transform: 資料增強函數
        mode: 'train' 或 'val'
        use_smart_slice: 是否使用智能切片選擇（選擇含腫瘤的切片）
    """
    def __init__(
        self, 
        data_dir: str, 
        patient_ids: List[str], 
        image_size: int = 128,
        transform: Optional[Callable] = None, 
        mode: str = 'train',
        use_smart_slice: bool = True
    ):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        self.use_smart_slice = use_smart_slice

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """
        影像正規化：percentile clipping + z-score normalization
        
        Args:
            img: 輸入影像
            
        Returns:
            正規化後的影像
        """
        # Percentile clipping (1%-99%)
        p1, p99 = np.percentile(img, (1, 99))
        img_clipped = np.clip(img, p1, p99)
        
        # Z-score normalization
        mean = np.mean(img_clipped)
        std = np.std(img_clipped)
        
        if std > 1e-8:
            img_normalized = (img_clipped - mean) / std
        else:
            img_normalized = img_clipped - mean
            
        return img_normalized

    def _select_slice(self, mask_volume: np.ndarray) -> int:
        """
        智能選擇含腫瘤的切片
        
        Args:
            mask_volume: 3D mask volume
            
        Returns:
            選擇的切片索引
        """
        if not self.use_smart_slice:
            return mask_volume.shape[2] // 2
        
        # 計算每個切片的腫瘤像素數
        tumor_counts = [np.sum(mask_volume[:, :, i] > 0) for i in range(mask_volume.shape[2])]
        
        # 選擇腫瘤最多的切片
        if max(tumor_counts) > 0:
            return int(np.argmax(tumor_counts))
        else:
            # 如果沒有腫瘤，返回中間切片
            return mask_volume.shape[2] // 2

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        獲取單個樣本
        
        Args:
            idx: 樣本索引
            
        Returns:
            (image, mask) tuple
        """
        patient_id = self.patient_ids[idx]
        patient_path = os.path.join(self.data_dir, patient_id)
        
        # 載入標籤以選擇最佳切片
        mask_path = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
        mask_volume = nib.load(mask_path).get_fdata()
        slice_idx = self._select_slice(mask_volume)
        
        # BraTS 四種序列
        modalities = ['flair', 't1', 't1ce', 't2']
        images = []
        
        for mod in modalities:
            path = os.path.join(patient_path, f"{patient_id}_{mod}.nii.gz")
            img_volume = nib.load(path).get_fdata()
            
            # 選取切片
            img_slice = img_volume[:, :, slice_idx]
            
            # 正規化
            img_slice = self._normalize_image(img_slice)
            
            # Resize
            img_slice = resize(img_slice, (self.image_size, self.image_size), preserve_range=True)
            images.append(img_slice)
            
        # 堆疊為 4 通道影像 (C, H, W)
        image = np.stack(images, axis=0)
        
        # 處理標籤
        mask_slice = mask_volume[:, :, slice_idx]
        mask = resize(mask_slice, (self.image_size, self.image_size), order=0, preserve_range=True)
        mask = (mask > 0).astype(np.float32)
        
        # 資料增強
        if self.transform:
            # albumentations 預設處理 HWC
            image_hwc = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image_hwc, mask=mask)
            image = np.transpose(augmented['image'], (2, 0, 1))
            mask = augmented['mask']
        
        # 轉換為 Tensor
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image, mask
