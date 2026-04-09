"""
BraTS dataset utilities with memory-aware scanning and robust validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import config


class BraTSDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        patient_ids: list[str],
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

        self.valid_patient_ids: list[str] = []
        self.patient_cache: dict = {}
        self.proxy_cache: dict = {}

        if prepared_cache:
            self._load_prepared_cache(prepared_cache)
        else:
            self._prepare_dataset()

    def _load_prepared_cache(self, prepared_cache: dict) -> None:
        """
        Filter a prepared cache down to the requested patient ids.

        This is used only when an external caller has already scanned the dataset and wants
        to subset the cache safely. Missing patient ids are logged for debugging.
        """
        cache_valid = prepared_cache.get("valid_patient_ids", [])
        cache_data = prepared_cache.get("patient_cache", {})
        cache_proxy = prepared_cache.get("proxy_cache", {})

        missing_in_cache: list[str] = []
        for pid in self.patient_ids:
            data = cache_data.get(pid)
            if pid in cache_valid and data is not None:
                self.valid_patient_ids.append(pid)
                self.patient_cache[pid] = data

                if config.USE_PROXY_CACHE:
                    proxy_bundle = cache_proxy.get(pid)
                    if proxy_bundle is not None:
                        self.proxy_cache[pid] = proxy_bundle
            else:
                missing_in_cache.append(pid)

        if missing_in_cache:
            print(f"Warning: Prepared cache missing {len(missing_in_cache)} patients from provided list.")
            print(f"Showing first 10 missing: {missing_in_cache[:10]}")
            log_path = self.output_dir / f"prepared_cache_missing_{self.mode}.txt"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(missing_in_cache))
            print(f"Full missing list saved to {log_path}")

        if not self.valid_patient_ids:
            raise ValueError("Error: No valid patients in provided patient_ids after filtering prepared_cache.")

    @staticmethod
    def _pick_stat_slices(n_slices: int, k: int) -> list[int]:
        if n_slices <= 0 or k <= 0:
            return []
        if k >= n_slices:
            return list(range(n_slices))
        idx = np.linspace(0, n_slices - 1, num=k, dtype=int)
        return sorted(set(int(i) for i in idx))

    @staticmethod
    def _estimate_norm_stats(proxy, slice_indices: list[int]) -> dict:
        """Estimate robust normalization stats from a small set of slices."""
        if not slice_indices:
            return {}
        vals = []
        for i in slice_indices:
            sl = np.asarray(proxy.dataobj[:, :, i]).astype(np.float32, copy=False)
            vals.append(sl.reshape(-1))
        flat = np.concatenate(vals, axis=0)
        p1, p99 = np.percentile(flat, [1, 99])
        clipped = np.clip(flat, p1, p99)
        return {
            "p1": float(p1),
            "p99": float(p99),
            "mean": float(np.mean(clipped)),
            "std": float(np.std(clipped)),
        }

    def _prepare_dataset(self) -> None:
        """
        Scan the requested patients, validate shapes, find tumor slices, and build
        lightweight cache metadata.
        """
        skipped_patients: list[str] = []
        print(f"Scanning {len(self.patient_ids)} patients for {self.mode}...")

        for pid in self.patient_ids:
            p_dir = self.data_dir / pid
            modalities = ["flair", "t1", "t1ce", "t2"]
            files = {mod: p_dir / f"{pid}_{mod}.nii.gz" for mod in modalities}
            files["seg"] = p_dir / f"{pid}_seg.nii.gz"

            if not all(f.exists() for f in files.values()):
                skipped_patients.append(f"Missing: {pid}")
                continue

            try:
                mask_proxy = nib.load(str(files["seg"]))
                shape = mask_proxy.header.get_data_shape()
                if len(shape) != 3:
                    skipped_patients.append(f"BadSegDim: {pid} ({shape})")
                    continue

                n_slices = int(shape[2])
                if n_slices <= 0:
                    skipped_patients.append(f"BadSegShape: {pid} ({shape})")
                    continue

                modality_shapes_ok = True
                for mod in modalities:
                    try:
                        m_shape = nib.load(str(files[mod])).header.get_data_shape()
                        if m_shape != shape:
                            modality_shapes_ok = False
                            skipped_patients.append(f"ShapeMismatch: {pid} ({mod} {m_shape} vs seg {shape})")
                            break
                    except Exception as e:
                        modality_shapes_ok = False
                        skipped_patients.append(f"ModReadError: {pid} ({mod}: {e})")
                        break
                if not modality_shapes_ok:
                    continue

                tumor_counts = []
                for i in range(n_slices):
                    slice_data = np.asarray(mask_proxy.dataobj[:, :, i])
                    tumor_counts.append(int(np.count_nonzero(slice_data > 0)))

                tumor_slice_indices = [i for i, count in enumerate(tumor_counts) if count > 0]
                non_tumor_slice_indices = [i for i, count in enumerate(tumor_counts) if count == 0]

                if not tumor_slice_indices:
                    skipped_patients.append(f"NoTumor: {pid}")
                    continue

                val_best_slice_idx = int(np.argmax(tumor_counts))
                stat_slices = self._pick_stat_slices(n_slices, int(config.STATS_N_SLICES))
                norm_stats: dict = {}

                self.valid_patient_ids.append(pid)
                self.patient_cache[pid] = {
                    "files": {k: str(v) for k, v in files.items()},
                    "tumor_slice_indices": tumor_slice_indices,
                    "non_tumor_slice_indices": non_tumor_slice_indices,
                    "val_best_slice_idx": val_best_slice_idx,
                    "norm_stats": norm_stats,
                }

                if config.USE_PROXY_CACHE:
                    self.proxy_cache[pid] = {mod: nib.load(str(files[mod])) for mod in modalities}
                    self.proxy_cache[pid]["seg"] = mask_proxy
                    for mod in modalities:
                        norm_stats[mod] = self._estimate_norm_stats(self.proxy_cache[pid][mod], stat_slices)
                else:
                    for mod in modalities:
                        proxy = nib.load(str(files[mod]))
                        norm_stats[mod] = self._estimate_norm_stats(proxy, stat_slices)

            except Exception as e:
                skipped_patients.append(f"ReadError: {pid} ({e})")

        if skipped_patients:
            print(f"Warning: Skipped {len(skipped_patients)} patients. (First 10 shown in log)")
            log_path = self.output_dir / "skipped_patients.txt"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(skipped_patients))
            print(f"Full skip list saved to {log_path}")

    def get_cache(self) -> dict:
        return {
            "valid_patient_ids": self.valid_patient_ids,
            "patient_cache": self.patient_cache,
            "proxy_cache": self.proxy_cache,
        }

    @staticmethod
    def quick_validate_patient(
        data_dir: Path,
        pid: str,
        require_tumor: bool = False,
        strict_tumor_check: bool = False,
    ) -> bool:
        """
        Lightweight validation for CLI selection.

        Checks:
        - file presence
        - NIfTI readability
        - 3D shape validity
        - modality/seg shape consistency
        - optional tumor presence
        """
        p_dir = data_dir / pid
        modalities = ["flair", "t1", "t1ce", "t2"]
        seg_path = p_dir / f"{pid}_seg.nii.gz"
        if not seg_path.exists():
            return False
        for mod in modalities:
            if not (p_dir / f"{pid}_{mod}.nii.gz").exists():
                return False

        try:
            seg_img = nib.load(str(seg_path))
            seg_shape = seg_img.header.get_data_shape()
            if len(seg_shape) != 3:
                return False

            for mod in modalities:
                img = nib.load(str(p_dir / f"{pid}_{mod}.nii.gz"))
                if img.header.get_data_shape() != seg_shape:
                    return False

            if require_tumor:
                z = int(seg_shape[2])
                if z <= 0:
                    return False
                has_tumor = False
                if strict_tumor_check:
                    for i in range(z):
                        sl = np.asarray(seg_img.dataobj[:, :, i])
                        if np.any(sl > 0):
                            has_tumor = True
                            break
                else:
                    sample_idx = sorted(set([0, z // 2, max(0, z - 1)]))
                    for i in sample_idx:
                        sl = np.asarray(seg_img.dataobj[:, :, i])
                        if np.any(sl > 0):
                            has_tumor = True
                            break
                if not has_tumor:
                    return False
        except Exception:
            return False

        return True

    def __len__(self) -> int:
        return len(self.valid_patient_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pid = self.valid_patient_ids[idx]
        cache = self.patient_cache[pid]

        if self.mode == "train":
            tumor_idx = cache.get("tumor_slice_indices", [])
            non_tumor_idx = cache.get("non_tumor_slice_indices", [])
            use_neg = (len(non_tumor_idx) > 0) and (np.random.rand() < float(config.NEG_SLICE_PROB))
            pick_from = non_tumor_idx if use_neg else tumor_idx
            if not pick_from:
                pick_from = tumor_idx or non_tumor_idx
            slice_idx = int(np.random.choice(pick_from))
        else:
            slice_idx = cache["val_best_slice_idx"]

        images = []
        modalities = ["flair", "t1", "t1ce", "t2"]
        norm_stats = cache.get("norm_stats", {}) or {}

        for mod in modalities:
            proxy_bundle = self.proxy_cache.get(pid)
            if config.USE_PROXY_CACHE and proxy_bundle is not None and mod in proxy_bundle:
                proxy = proxy_bundle[mod]
            else:
                proxy = nib.load(cache["files"][mod])

            img_slice = np.asarray(proxy.dataobj[:, :, slice_idx])
            img_slice = self._normalize(img_slice, stats=norm_stats.get(mod))
            images.append(img_slice)

        proxy_bundle = self.proxy_cache.get(pid)
        if config.USE_PROXY_CACHE and proxy_bundle is not None and "seg" in proxy_bundle:
            seg_proxy = proxy_bundle["seg"]
        else:
            seg_proxy = nib.load(cache["files"]["seg"])

        mask_slice = np.asarray(seg_proxy.dataobj[:, :, slice_idx])
        mask_slice = (mask_slice > 0).astype(np.float32)

        image_tensor = torch.from_numpy(np.stack(images, axis=0)).float()
        mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0).float()

        if image_tensor.shape[-1] != self.image_size or image_tensor.shape[-2] != self.image_size:
            image_tensor_b = image_tensor.unsqueeze(0)
            mask_tensor_b = mask_tensor.unsqueeze(0)
            image_tensor_b = F.interpolate(
                image_tensor_b,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            mask_tensor_b = F.interpolate(mask_tensor_b, size=(self.image_size, self.image_size), mode="nearest")
            image_tensor = image_tensor_b.squeeze(0)
            mask_tensor = mask_tensor_b.squeeze(0)

        return image_tensor, mask_tensor

    def _normalize(self, img: np.ndarray, stats: Optional[dict] = None) -> np.ndarray:
        """
        Percentile clip + z-score normalization.

        If per-patient stats are available, use them to avoid expensive percentile computation
        in `__getitem__`.
        """
        img = img.astype(np.float32, copy=False)
        if stats and all(k in stats for k in ("p1", "p99", "mean", "std")):
            p1 = float(stats["p1"])
            p99 = float(stats["p99"])
            mean = float(stats["mean"])
            std = float(stats["std"])
        else:
            p1, p99 = np.percentile(img, [1, 99])
            clipped = np.clip(img, p1, p99)
            mean = float(np.mean(clipped))
            std = float(np.std(clipped))

        img = np.clip(img, p1, p99)
        return (img - mean) / (std + 1e-8)
