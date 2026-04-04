"""
BraTS Dataset with extreme memory optimization, shared cache subsetting, and robust error handling
(v3.1 Final Release Gold Master)
"""

from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

import config


class BraTSDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        patient_ids: List[str],
        image_size: int = 128,
        mode: str = "train",
        prepared_cache: Optional[dict] = None,
        output_dir: Optional[Path] = None,
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
            # 強化快取共享子集化邏輯，確保安全性與分離
            cache_valid = prepared_cache.get("valid_patient_ids", [])
            cache_data = prepared_cache.get("patient_cache", {})
            cache_proxy = prepared_cache.get("proxy_cache", {})

            missing_in_cache = []
            for pid in patient_ids:
                # 更保守的檢查，避免 KeyError
                data = cache_data.get(pid)
                if pid in cache_valid and data is not None:
                    self.valid_patient_ids.append(pid)
                    self.patient_cache[pid] = data

                    # 只有當 proxy_bundle 不為 None 時才放入，避免 getitem 崩潰
                    if config.USE_PROXY_CACHE:
                        proxy_bundle = cache_proxy.get(pid)
                        if proxy_bundle is not None:
                            self.proxy_cache[pid] = proxy_bundle
                else:
                    missing_in_cache.append(pid)

            # 統一輸出快取缺失摘要，並寫入對應的 output_dir
            if missing_in_cache:
                n_missing = len(missing_in_cache)
                print(f"⚠️  Prepared cache missing {n_missing} patients from provided list.")
                print(f"💡 Showing first 10 missing: {missing_in_cache[:10]}")

                # This file stores only the latest missing records for the given mode.
                # It is intentionally overwritten each run to keep logs concise.
                log_path = self.output_dir / f"prepared_cache_missing_{self.mode}.txt"
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with open(log_path, "w") as f:
                    f.write("\n".join(missing_in_cache))
                print(f"📝 Full missing list saved to {log_path}")

            if not self.valid_patient_ids:
                # 改為 raise ValueError 以利 CI/自動化流程抓錯
                raise ValueError("❌ Error: No valid patients in provided patient_ids after filtering prepared_cache.")
        else:
            self._prepare_dataset()

    def _prepare_dataset(self) -> None:
        """
        掃描資料夾並預先計算切片索引 (真正逐切片掃描，極致節省記憶體)
        """
        skipped_patients = []
        print(f"🔍 Scanning {len(self.patient_ids)} patients for {self.mode}...")

        for pid in self.patient_ids:
            p_dir = self.data_dir / pid
            modalities = ["flair", "t1", "t1ce", "t2"]
            files = {mod: p_dir / f"{pid}_{mod}.nii.gz" for mod in modalities}
            files["seg"] = p_dir / f"{pid}_seg.nii.gz"

            if not all(f.exists() for f in files.values()):
                skipped_patients.append(f"Missing: {pid}")
                continue

            try:
                # 真正逐切片掃描，不將整個 3D Volume 載入記憶體
                mask_proxy = nib.load(str(files["seg"]))
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
                    "files": {k: str(v) for k, v in files.items()},
                    "tumor_slice_indices": tumor_slice_indices,
                    "val_best_slice_idx": val_best_slice_idx,
                }

                # 若開啟 Proxy 快取，則快取 nibabel 對象以提升 I/O 效能
                if config.USE_PROXY_CACHE:
                    self.proxy_cache[pid] = {mod: nib.load(str(files[mod])) for mod in modalities}
                    self.proxy_cache[pid]["seg"] = mask_proxy

            except Exception as e:
                skipped_patients.append(f"ReadError: {pid} ({str(e)})")

        if skipped_patients:
            print(f"⚠️  Skipped {len(skipped_patients)} patients. (First 10 shown in log)")
            log_path = self.output_dir / "skipped_patients.txt"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                f.write("\n".join(skipped_patients))
            print(f"📝 Full skip list saved to {log_path}")

    def get_cache(self) -> dict:
        return {
            "valid_patient_ids": self.valid_patient_ids,
            "patient_cache": self.patient_cache,
            "proxy_cache": self.proxy_cache,
        }

    @staticmethod
    def quick_validate_patient(data_dir: Path, pid: str) -> bool:
        """
        輕量化驗證：僅檢查檔案是否存在
        """
        p_dir = data_dir / pid
        modalities = ["flair", "t1", "t1ce", "t2", "seg"]
        for mod in modalities:
            if not (p_dir / f"{pid}_{mod}.nii.gz").exists():
                return False
        return True

    def __len__(self) -> int:
        return len(self.valid_patient_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pid = self.valid_patient_ids[idx]
        cache = self.patient_cache[pid]

        if self.mode == "train":
            slice_idx = int(np.random.choice(cache["tumor_slice_indices"]))
        else:
            slice_idx = cache["val_best_slice_idx"]

        images = []
        modalities = ["flair", "t1", "t1ce", "t2"]

        for mod in modalities:
            # 強化防呆檢查，確保 proxy_cache[pid] 存在且不為 None
            proxy_bundle = self.proxy_cache.get(pid)
            if config.USE_PROXY_CACHE and proxy_bundle is not None and mod in proxy_bundle:
                proxy = proxy_bundle[mod]
            else:
                proxy = nib.load(cache["files"][mod])

            img_slice = np.asarray(proxy.dataobj[:, :, slice_idx])
            img_slice = self._normalize(img_slice)
            images.append(img_slice)

        proxy_bundle = self.proxy_cache.get(pid)
        if config.USE_PROXY_CACHE and proxy_bundle is not None and "seg" in proxy_bundle:
            seg_proxy = proxy_bundle["seg"]
        else:
            seg_proxy = nib.load(cache["files"]["seg"])

        mask_slice = np.asarray(seg_proxy.dataobj[:, :, slice_idx])
        mask_slice = (mask_slice > 0).astype(np.float32)

        # resize 已移至頂部 import
        images = [
            resize(img, (self.image_size, self.image_size), order=1, preserve_range=True, anti_aliasing=True)
            for img in images
        ]
        mask_slice = resize(
            mask_slice, (self.image_size, self.image_size), order=0, preserve_range=True, anti_aliasing=False
        )

        image_tensor = torch.from_numpy(np.stack(images, axis=0)).float()
        mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0).float()

        return image_tensor, mask_tensor

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-8)
