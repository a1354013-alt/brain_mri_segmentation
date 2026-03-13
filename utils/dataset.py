"""
BraTS Dataset with optimized I/O, shared cache subsetting, and memory-friendly scanning (v2.7 Final)
"""
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import config


class BraTSDataset(Dataset):
    def __init__(
        self, 
        data_dir: Path, 
        patient_ids: List[str], 
        image_size: int = 128, 
        mode: str = 'train',
        prepared_cache: Optional[Dict] = None
    ):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.image_size = image_size
        self.mode = mode
        
        self.valid_patient_ids = []
        self.patient_cache = {}
        self.proxy_cache = {}
        
        if prepared_cache:
            # v2.7 Final: 修正快取共享子集化邏輯，確保訓練/驗證集分離
            cache_valid = prepared_cache.get("valid_patient_ids", [])
            cache_data = prepared_cache.get("patient_cache", {})
            cache_proxy = prepared_cache.get("proxy_cache", {})
            
            missing_in_cache = []
            for pid in patient_ids:
                if pid in cache_valid:
                    self.valid_patient_ids.append(pid)
                    self.patient_cache[pid] = cache_data[pid]
                    if config.USE_PROXY_CACHE:
                        self.proxy_cache[pid] = cache_proxy.get(pid)
                else:
                    # v2.7 Final: 收集缺失 PID 以便後續統一輸出
                    missing_in_cache.append(pid)
            
            # v2.7 Final: 統一輸出快取缺失摘要，避免洗版
            if missing_in_cache:
                n_missing = len(missing_in_cache)
                print(f"⚠️  Prepared cache missing {n_missing} patients from provided list.")
                print(f"💡 Showing first 10 missing: {missing_in_cache[:10]}")
                
                # 記錄完整缺失清單至檔案
                log_path = config.OUTPUT_DIR / "prepared_cache_missing.txt"
                config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                with open(log_path, "w") as f:
                    f.write("\n".join(missing_in_cache))
                print(f"📝 Full missing list saved to {log_path}")
                
            if not self.valid_patient_ids:
                print(f"❌ Error: No valid patients in provided patient_ids after filtering prepared_cache.")
        else:
            self._prepare_dataset()

    def _prepare_dataset(self):
        """
        掃描資料夾並預先計算切片索引 (v2.7 Final: 記憶體友善掃描)
        """
        skipped_patients = []
        print(f"🔍 Scanning {len(self.patient_ids)} patients for {self.mode}...")
        
        for pid in self.patient_ids:
            p_dir = self.data_dir / pid
            # 檢查 4 模態 + Seg
            modalities = ['flair', 't1', 't1ce', 't2']
            files = {mod: p_dir / f"{pid}_{mod}.nii.gz" for mod in modalities}
            files['seg'] = p_dir / f"{pid}_seg.nii.gz"
            
            if not all(f.exists() for f in files.values()):
                skipped_patients.append(f"Missing: {pid}")
                continue
            
            try:
                # 記憶體友善掃描：僅讀取 Seg 標籤
                mask_proxy = nib.load(str(files['seg']))
                # 使用 np.asarray(proxy.dataobj) 避免 get_fdata() 的 float64 轉換與記憶體膨脹
                mask_data = np.asarray(mask_proxy.dataobj)
                
                # 找出所有含腫瘤的切片索引 (Whole Tumor: mask > 0)
                # v2.7 Final: 逐切片掃描以極致節省記憶體
                tumor_counts = []
                for i in range(mask_data.shape[2]):
                    tumor_counts.append(np.count_nonzero(mask_data[:, :, i] > 0))
                
                tumor_slice_indices = [i for i, count in enumerate(tumor_counts) if count > 0]
                
                if not tumor_slice_indices:
                    skipped_patients.append(f"NoTumor: {pid}")
                    continue
                
                # 驗證集固定使用腫瘤最多的切片
                val_best_slice_idx = int(np.argmax(tumor_counts))
                
                self.valid_patient_ids.append(pid)
                self.patient_cache[pid] = {
                    'files': {k: str(v) for k, v in files.items()},
                    'tumor_slice_indices': tumor_slice_indices,
                    'val_best_slice_idx': val_best_slice_idx
                }
                
                # 若開啟 Proxy 快取，則快取 nibabel 對象以提升 I/O 效能
                if config.USE_PROXY_CACHE:
                    self.proxy_cache[pid] = {mod: nib.load(str(files[mod])) for mod in modalities}
                    self.proxy_cache[pid]['seg'] = mask_proxy
                    
            except Exception as e:
                skipped_patients.append(f"ReadError: {pid} ({str(e)})")
        
        if skipped_patients:
            print(f"⚠️  Skipped {len(skipped_patients)} patients. (First 10 shown in log)")
            log_path = config.OUTPUT_DIR / "skipped_patients.txt"
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
        輕量化驗證：僅檢查檔案是否存在 (v2.7 Final)
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
        
        # 選擇切片
        if self.mode == 'train':
            # 訓練集：從含腫瘤切片中隨機抽樣
            slice_idx = int(np.random.choice(cache['tumor_slice_indices']))
        else:
            # 驗證集：固定使用腫瘤最多的切片
            slice_idx = cache['val_best_slice_idx']
            
        # 讀取影像與標籤
        images = []
        modalities = ['flair', 't1', 't1ce', 't2']
        
        for mod in modalities:
            if config.USE_PROXY_CACHE and pid in self.proxy_cache:
                proxy = self.proxy_cache[pid][mod]
            else:
                proxy = nib.load(cache['files'][mod])
            
            # 僅載入需要的 slice
            img_slice = np.asarray(proxy.dataobj[:, :, slice_idx])
            img_slice = self._normalize(img_slice)
            images.append(img_slice)
            
        # 讀取標籤
        if config.USE_PROXY_CACHE and pid in self.proxy_cache:
            seg_proxy = self.proxy_cache[pid]['seg']
        else:
            seg_proxy = nib.load(cache['files']['seg'])
            
        mask_slice = np.asarray(seg_proxy.dataobj[:, :, slice_idx])
        # Whole Tumor (WT) 二元分割
        mask_slice = (mask_slice > 0).astype(np.float32)
        
        # Resize
        from skimage.transform import resize
        images = [resize(img, (self.image_size, self.image_size), order=1, preserve_range=True, anti_aliasing=True) for img in images]
        mask_slice = resize(mask_slice, (self.image_size, self.image_size), order=0, preserve_range=True, anti_aliasing=False)
        
        # 轉換為 Tensor (C, H, W)
        image_tensor = torch.from_numpy(np.stack(images, axis=0)).float()
        mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0).float()
        
        return image_tensor, mask_tensor

    def _normalize(self, img):
        """
        Percentile clip + Z-score normalization
        """
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-8)
import torch
