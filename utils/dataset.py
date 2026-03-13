"""
BraTS Dataset with extreme memory optimization, shared cache subsetting, and robust error handling (v2.8 Final)
"""
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Optional
import config


class BraTSDataset(Dataset):
    def __init__(
        self, 
        data_dir: Path, 
        patient_ids: List[str], 
        image_size: int = 128, 
        mode: str = 'train',
        prepared_cache: Optional[Dict] = None,
        output_dir: Optional[Path] = None
    ):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.image_size = image_size
        self.mode = mode
        self.output_dir = output_dir or config.OUTPUT_DIR
        
        self.valid_patient_ids = []
        self.patient_cache = {}
        self.proxy_cache = {}
        
        if prepared_cache:
            # v2.8 Final: 強化快取共享子集化邏輯，確保安全性與分離
            cache_valid = prepared_cache.get("valid_patient_ids", [])
            cache_data = prepared_cache.get("patient_cache", {})
            cache_proxy = prepared_cache.get("proxy_cache", {})
            
            missing_in_cache = []
            for pid in patient_ids:
                # v2.8 Final: 更保守的檢查，避免 KeyError
                data = cache_data.get(pid)
                if pid in cache_valid and data is not None:
                    self.valid_patient_ids.append(pid)
                    self.patient_cache[pid] = data
                    if config.USE_PROXY_CACHE:
                        self.proxy_cache[pid] = cache_proxy.get(pid)
                else:
                    missing_in_cache.append(pid)
            
            # v2.8 Final: 統一輸出快取缺失摘要，並寫入對應的 output_dir
            if missing_in_cache:
                n_missing = len(missing_in_cache)
                print(f"⚠️  Prepared cache missing {n_missing} patients from provided list.")
                print(f"💡 Showing first 10 missing: {missing_in_cache[:10]}")
                
                log_path = self.output_dir / "prepared_cache_missing.txt"
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with open(log_path, "w") as f:
                    f.write("\n".join(missing_in_cache))
                print(f"📝 Full missing list saved to {log_path}")
                
            if not self.valid_patient_ids:
                print(f"❌ Error: No valid patients in provided patient_ids after filtering prepared_cache.")
        else:
            self._prepare_dataset()

    def _prepare_dataset(self):
        """
        掃描資料夾並預先計算切片索引 (v2.8 Final: 真正逐切片掃描，極致節省記憶體)
        """
        skipped_patients = []
        print(f"🔍 Scanning {len(self.patient_ids)} patients for {self.mode}...")
        
        for pid in self.patient_ids:
            p_dir = self.data_dir / pid
            modalities = ['flair', 't1', 't1ce', 't2']
            files = {mod: p_dir / f"{pid}_{mod}.nii.gz" for mod in modalities}
            files['seg'] = p_dir / f"{pid}_seg.nii.gz"
            
            if not all(f.exists() for f in files.values()):
                skipped_patients.append(f"Missing: {pid}")
                continue
            
            try:
                # v2.8 Final: 真正逐切片掃描，不將整個 3D Volume 載入記憶體
                mask_proxy = nib.load(str(files['seg']))
                shape = mask_proxy.header.get_data_shape()
                n_slices = shape[2]
                
                tumor_counts = []
                for i in range(n_slices):
                    # 直接從 proxy.dataobj 讀取單一切片，極致節省 RAM
                    slice_data = np.asarray(mask_proxy.dataobj[:, :, i])
                    tumor_counts.append(np.count_nonzero(slice_data > 0))
                
                tumor_slice_indices = [i for i, count in enumerate(tumor_counts) if count > 0]
                
                if not tumor_slice_indices:
                    skipped_patients.append(f"NoTumor: {pid}")
                    continue
                
                val_best_slice_idx = int(np.argmax(tumor_counts))
                
                self.valid_patient_ids.append(pid)
                self.patient_cache[pid] = {
                    'files': {k: str(v) for k, v in files.items()},
                    'tumor_slice_indices': tumor_slice_indices,
                    'val_best_slice_idx': val_best_slice_idx
                }
                
                if config.USE_PROXY_CACHE:
                    self.proxy_cache[pid] = {mod: nib.load(str(files[mod])) for mod in modalities}
                    self.proxy_cache[pid]['seg'] = mask_proxy
                    
            except Exception as e:
                skipped_patients.append(f"ReadError: {pid} ({str(e)})")
        
        if skipped_patients:
            print(f"⚠️  Skipped {len(skipped_patients)} patients. (First 10 shown in log)")
            log_path = self.output_dir / "skipped_patients.txt"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                f.write("\n".join(skipped_patients))
            print(f"📝 Full skip list saved to {log_path}")

    def get_cache(self) -> Dict:
        return {
            "valid_patient_ids": self.valid_patient_ids,
            "patient_cache": self.patient_cache,
            "proxy_cache": self.proxy_cache
        }

    @staticmethod
    def quick_validate_patient(data_dir: Path, pid: str) -> bool:
        """
        輕量化驗證：僅檢查檔案是否存在 (v2.8 Final)
        """
        p_dir = data_dir / pid
        modalities = ['flair', 't1', 't1ce', 't2', 'seg']
        for mod in modalities:
            if not (p_dir / f"{pid}_{mod}.nii.gz").exists():
                return False
        return True

    def __len__(self):
        return len(self.valid_patient_ids)

    def __getitem__(self, idx):
        pid = self.valid_patient_ids[idx]
        cache = self.patient_cache[pid]
        
        if self.mode == 'train':
            slice_idx = int(np.random.choice(cache['tumor_slice_indices']))
        else:
            slice_idx = cache['val_best_slice_idx']
            
        images = []
        modalities = ['flair', 't1', 't1ce', 't2']
        
        for mod in modalities:
            if config.USE_PROXY_CACHE and pid in self.proxy_cache:
                proxy = self.proxy_cache[pid][mod]
            else:
                proxy = nib.load(cache['files'][mod])
            
            img_slice = np.asarray(proxy.dataobj[:, :, slice_idx])
            img_slice = self._normalize(img_slice)
            images.append(img_slice)
            
        if config.USE_PROXY_CACHE and pid in self.proxy_cache:
            seg_proxy = self.proxy_cache[pid]['seg']
        else:
            seg_proxy = nib.load(cache['files']['seg'])
            
        mask_slice = np.asarray(seg_proxy.dataobj[:, :, slice_idx])
        mask_slice = (mask_slice > 0).astype(np.float32)
        
        from skimage.transform import resize
        images = [resize(img, (self.image_size, self.image_size), order=1, preserve_range=True, anti_aliasing=True) for img in images]
        mask_slice = resize(mask_slice, (self.image_size, self.image_size), order=0, preserve_range=True, anti_aliasing=False)
        
        image_tensor = torch.from_numpy(np.stack(images, axis=0)).float()
        mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0).float()
        
        return image_tensor, mask_tensor

    def _normalize(self, img):
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-8)
