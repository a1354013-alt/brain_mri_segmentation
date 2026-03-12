"""
BraTS Dataset implementation with multi-worker RNG fix and I/O optimization
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
    
    修正點：
    1. 移除內部 RNG，改用 np.random.choice 以配合 DataLoader worker_init_fn
    2. 強化完整性檢查：必須包含 4 模態 (flair, t1, t1ce, t2) + seg
    3. 優化 Resize 參數：image (order=1, anti_aliasing=True), mask (order=0)
    """
    def __init__(
        self, 
        data_dir: Path, 
        patient_ids: List[str], 
        image_size: int = 128,
        transform: Optional[Callable] = None, 
        mode: str = 'train'
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        
        self.valid_patient_ids = []
        self.patient_cache = {}
        
        self._prepare_dataset(patient_ids)

    def _prepare_dataset(self, patient_ids: List[str]) -> None:
        skipped_patients = []
        modalities = ['flair', 't1', 't1ce', 't2']
        
        print(f"🔍 Scanning {len(patient_ids)} patients for 4-modality integrity...")
        
        for pid in patient_ids:
            p_path = self.data_dir / pid
            missing_files = []
            
            for mod in modalities:
                if not (p_path / f"{pid}_{mod}.nii.gz").exists():
                    missing_files.append(f"{pid}_{mod}.nii.gz")
            
            mask_file = p_path / f"{pid}_seg.nii.gz"
            if not mask_file.exists():
                missing_files.append(f"{pid}_seg.nii.gz")
            
            if missing_files:
                skipped_patients.append((pid, missing_files))
                continue
            
            try:
                mask_volume = nib.load(str(mask_file)).get_fdata()
                mask_binary = (mask_volume > 0).astype(np.uint8)
                
                tumor_counts = np.sum(mask_binary, axis=(0, 1))
                tumor_indices = np.where(tumor_counts > 0)[0].tolist()
                
                if not tumor_indices:
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

        if skipped_patients:
            print(f"⚠️  Skipped {len(skipped_patients)} incomplete patients (missing required modalities or seg):")
            for pid, files in skipped_patients:
                print(f"   - {pid}: Missing {', '.join(files)}")
        
        print(f"✅ Dataset ready: {len(self.valid_patient_ids)} valid patients.")

    def __len__(self) -> int:
        return len(self.valid_patient_ids)

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(img, (1, 99))
        img_clipped = np.clip(img, p1, p99)
        mean, std = np.mean(img_clipped), np.std(img_clipped)
        return (img_clipped - mean) / (std + 1e-8)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pid = self.valid_patient_ids[idx]
        p_path = self.data_dir / pid
        cache = self.patient_cache[pid]
        
        if self.mode == 'train':
            # 使用 np.random.choice 配合 worker_init_fn
            slice_idx = np.random.choice(cache['tumor_indices'])
        else:
            slice_idx = cache['best_idx']
        
        modalities = ['flair', 't1', 't1ce', 't2']
        images = []
        
        for mod in modalities:
            img_path = p_path / f"{pid}_{mod}.nii.gz"
            img_proxy = nib.load(str(img_path))
            img_slice = np.array(img_proxy.dataobj[:, :, slice_idx])
            
            img_slice = self._normalize_image(img_slice)
            # image: order=1, anti_aliasing=True
            img_slice = resize(img_slice, (self.image_size, self.image_size), order=1, anti_aliasing=True, preserve_range=True)
            images.append(img_slice)
            
        image = np.stack(images, axis=0)
        
        mask_path = p_path / f"{pid}_seg.nii.gz"
        mask_proxy = nib.load(str(mask_path))
        mask_slice = np.array(mask_proxy.dataobj[:, :, slice_idx])
        # mask: order=0
        mask = resize(mask_slice, (self.image_size, self.image_size), order=0, anti_aliasing=False, preserve_range=True)
        mask = (mask > 0).astype(np.float32)
        
        if self.transform:
            image_hwc = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image_hwc, mask=mask)
            image = np.transpose(augmented['image'], (2, 0, 1))
            mask = augmented['mask']
        
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float().unsqueeze(0)
