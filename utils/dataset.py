"""
BraTS Dataset implementation with I/O optimization and robust scanning (v2.3)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from skimage.transform import resize
from pathlib import Path
from typing import List, Optional, Tuple, Callable
import config


class BraTSDataset(Dataset):
    """
    針對 BraTS 資料集的 PyTorch Dataset 類別 (v2.3)
    
    優化點：
    1. 快取 nibabel dataobj proxy 以減少磁碟 I/O 開銷
    2. 掃描日誌優化：僅列印前 10 筆錯誤，完整清單寫入 outputs/skipped_patients.txt
    3. 移除內部 RNG，配合 DataLoader worker_init_fn
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
        self.proxy_cache = {} # 快取 dataobj proxy
        
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
                mask_proxy = nib.load(str(mask_file))
                mask_volume = mask_proxy.get_fdata()
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
                
                # 預先載入 proxy (不載入整個 volume)
                self.proxy_cache[pid] = {
                    'seg': mask_proxy.dataobj
                }
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
            print("   First 10 skipped patients:")
            for pid, files in skipped_patients[:10]:
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
        cache = self.patient_cache[pid]
        proxies = self.proxy_cache[pid]
        
        if self.mode == 'train':
            slice_idx = np.random.choice(cache['tumor_indices'])
        else:
            slice_idx = cache['best_idx']
        
        modalities = ['flair', 't1', 't1ce', 't2']
        images = []
        
        for mod in modalities:
            # 從快取的 proxy 讀取特定 slice
            img_slice = np.array(proxies[mod][:, :, slice_idx])
            img_slice = self._normalize_image(img_slice)
            img_slice = resize(img_slice, (self.image_size, self.image_size), order=1, anti_aliasing=True, preserve_range=True)
            images.append(img_slice)
            
        image = np.stack(images, axis=0)
        
        # 從快取的 proxy 讀取 mask slice
        mask_slice = np.array(proxies['seg'][:, :, slice_idx])
        mask = resize(mask_slice, (self.image_size, self.image_size), order=0, anti_aliasing=False, preserve_range=True)
        mask = (mask > 0).astype(np.float32)
        
        if self.transform:
            image_hwc = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image_hwc, mask=mask)
            image = np.transpose(augmented['image'], (2, 0, 1))
            mask = augmented['mask']
        
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float().unsqueeze(0)
