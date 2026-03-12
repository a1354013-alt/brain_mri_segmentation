"""
BraTS Dataset implementation with shared cache and lightweight validation (v2.6)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from skimage.transform import resize
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict
import config


class BraTSDataset(Dataset):
    """
    針對 BraTS 資料集的 PyTorch Dataset 類別 (v2.6)
    
    優化點：
    1. 快取共享：支援傳入外部 prepared_cache，避免重複掃描。
    2. 輕量化驗證：提供 quick_validate_patient 靜態方法。
    3. 逐切片掃描：極致節省記憶體。
    """
    def __init__(
        self, 
        data_dir: Path, 
        patient_ids: List[str], 
        image_size: int = 128,
        transform: Optional[Callable] = None, 
        mode: str = 'train',
        prepared_cache: Optional[Dict] = None
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        
        self.valid_patient_ids = []
        self.patient_cache = {}
        self.proxy_cache = {}
        
        if prepared_cache:
            self.valid_patient_ids = prepared_cache['valid_patient_ids']
            self.patient_cache = prepared_cache['patient_cache']
            self.proxy_cache = prepared_cache.get('proxy_cache', {})
            print(f"🚀 Dataset initialized with shared cache ({len(self.valid_patient_ids)} patients).")
        else:
            self._prepare_dataset(patient_ids)

    @staticmethod
    def quick_validate_patient(data_dir: Path, pid: str) -> bool:
        """
        輕量化驗證：僅檢查檔案存在，不讀取內容 (v2.6)
        """
        p_path = data_dir / pid
        modalities = ['flair', 't1', 't1ce', 't2', 'seg']
        for mod in modalities:
            if not (p_path / f"{pid}_{mod}.nii.gz").exists():
                return False
        return True

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
                mask_proxy = nib.load(str(mask_file))
                n_slices = mask_proxy.shape[2]
                
                # 逐切片掃描
                tumor_counts = []
                for s in range(n_slices):
                    slice_data = np.asarray(mask_proxy.dataobj[:, :, s])
                    tumor_counts.append(np.sum(slice_data > 0))
                
                tumor_counts = np.array(tumor_counts)
                tumor_indices = np.where(tumor_counts > 0)[0].tolist()
                
                if not tumor_indices:
                    best_idx = n_slices // 2
                    tumor_indices = [best_idx]
                else:
                    best_idx = int(np.argmax(tumor_counts))
                
                self.patient_cache[pid] = {
                    'tumor_indices': tumor_indices,
                    'best_idx': best_idx
                }
                self.valid_patient_ids.append(pid)
                
                if config.USE_PROXY_CACHE:
                    self.proxy_cache[pid] = {'seg': mask_proxy.dataobj}
                    for mod in modalities:
                        self.proxy_cache[pid][mod] = nib.load(str(p_path / f"{pid}_{mod}.nii.gz")).dataobj
                
            except Exception as e:
                skipped_patients.append((pid, [f"Error reading: {str(e)}"]))

        if skipped_patients:
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(config.SKIPPED_LOG, 'w') as f:
                for pid, files in skipped_patients:
                    f.write(f"{pid}: Missing {', '.join(files)}\n")
            
            print(f"⚠️  Skipped {len(skipped_patients)} patients. Full list in {config.SKIPPED_LOG}")
        
        print(f"✅ Dataset ready: {len(self.valid_patient_ids)} valid patients.")

    def get_cache(self) -> Dict:
        """
        獲取當前 Dataset 的快取，以便共享 (v2.6)
        """
        return {
            'valid_patient_ids': self.valid_patient_ids,
            'patient_cache': self.patient_cache,
            'proxy_cache': self.proxy_cache
        }

    def __len__(self) -> int:
        return len(self.valid_patient_ids)

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(img, (1, 99))
        img_clipped = np.clip(img, p1, p99)
        mean, std = np.mean(img_clipped), np.std(img_clipped)
        return (img_clipped - mean) / (std + 1e-8)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pid = self.valid_patient_ids[idx]
        cache = self.patient_cache[pid]
        
        if self.mode == 'train':
            slice_idx = np.random.choice(cache['tumor_indices'])
        else:
            slice_idx = cache['best_idx']
        
        modalities = ['flair', 't1', 't1ce', 't2']
        images = []
        
        for mod in modalities:
            if config.USE_PROXY_CACHE:
                img_slice = np.array(self.proxy_cache[pid][mod][:, :, slice_idx])
            else:
                img_path = self.data_dir / pid / f"{pid}_{mod}.nii.gz"
                img_proxy = nib.load(str(img_path))
                img_slice = np.array(img_proxy.dataobj[:, :, slice_idx])
                
            img_slice = self._normalize_image(img_slice)
            img_slice = resize(img_slice, (self.image_size, self.image_size), order=1, anti_aliasing=True, preserve_range=True)
            images.append(img_slice)
            
        image = np.stack(images, axis=0)
        
        if config.USE_PROXY_CACHE:
            mask_slice = np.array(self.proxy_cache[pid]['seg'][:, :, slice_idx])
        else:
            mask_path = self.data_dir / pid / f"{pid}_seg.nii.gz"
            mask_proxy = nib.load(str(mask_path))
            mask_slice = np.array(mask_proxy.dataobj[:, :, slice_idx])
            
        mask = resize(mask_slice, (self.image_size, self.image_size), order=0, anti_aliasing=False, preserve_range=True)
        mask = (mask > 0).astype(np.float32)
        
        if self.transform:
            image_hwc = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image_hwc, mask=mask)
            image = np.transpose(augmented['image'], (2, 0, 1))
            mask = augmented['mask']
        
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float().unsqueeze(0)
