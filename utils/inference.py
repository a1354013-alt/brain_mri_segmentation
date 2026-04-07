"""
Volume inference utilities (2D slice model -> 3D NIfTI output).

This project trains a 2D segmentation model on slices. For evaluation and downstream use,
it's common to run inference across all slices and export a 3D mask aligned to the original NIfTI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

import config
from .visualize import enable_dropout


def _normalize_slice(img: np.ndarray, stats: Optional[dict]) -> np.ndarray:
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


def _load_patient_files(data_dir: Path, pid: str) -> dict:
    p_dir = Path(data_dir) / pid
    modalities = ["flair", "t1", "t1ce", "t2"]
    files = {mod: p_dir / f"{pid}_{mod}.nii.gz" for mod in modalities}
    files["seg"] = p_dir / f"{pid}_seg.nii.gz"
    return files


def _prepare_input_tensor(
    proxies: dict,
    slice_idx: int,
    image_size: int,
    norm_stats: Optional[dict],
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    modalities = ["flair", "t1", "t1ce", "t2"]
    slices = []
    for mod in modalities:
        sl = np.asarray(proxies[mod].dataobj[:, :, slice_idx])
        sl = _normalize_slice(sl, stats=(norm_stats or {}).get(mod) if isinstance(norm_stats, dict) else None)
        slices.append(sl)

    img = torch.from_numpy(np.stack(slices, axis=0)).float()  # (4,H,W)
    h, w = int(img.shape[-2]), int(img.shape[-1])
    img = img.unsqueeze(0).to(device)  # (1,4,H,W)
    if (h, w) != (image_size, image_size):
        img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return img, (h, w)


def mc_predict_slice(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_iterations: int,
    method: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (binary_pred, uncertainty, mean_prob) tensors with shape (1,1,H,W).
    """
    model.eval()
    enable_dropout(model)
    preds = []
    with torch.no_grad():
        for _ in range(int(n_iterations)):
            out = torch.sigmoid(model(x))
            preds.append(out)
    preds_t = torch.stack(preds, dim=0)  # (N,1,1,H,W)
    mean_prob = preds_t.mean(dim=0)

    if method == "entropy":
        p = torch.clamp(mean_prob, 1e-8, 1.0 - 1e-8)
        uncertainty = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    else:
        uncertainty = preds_t.var(dim=0)

    binary_pred = (mean_prob > float(config.THRESHOLD)).float()
    return binary_pred, uncertainty, mean_prob


def predict_patient_volume(
    model: torch.nn.Module,
    data_dir: Path,
    pid: str,
    image_size: int,
    device: torch.device,
    n_iterations: int = config.MC_ITERATIONS,
    method: str = "var",
    norm_stats: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, nib.Nifti1Image]:
    """
    Runs slice-wise MC dropout inference for a patient and returns:
    - pred_vol: (H,W,Z) uint8
    - unc_vol: (H,W,Z) float32
    - prob_vol: (H,W,Z) float32
    - seg_ref_img: reference NIfTI (for affine/header)
    """
    files = _load_patient_files(data_dir, pid)
    proxies = {k: nib.load(str(v)) for k, v in files.items() if k != "seg"}
    seg_ref_img = nib.load(str(files["seg"]))

    shape = seg_ref_img.header.get_data_shape()
    h, w, z = int(shape[0]), int(shape[1]), int(shape[2])
    pred_vol = np.zeros((h, w, z), dtype=np.uint8)
    unc_vol = np.zeros((h, w, z), dtype=np.float32)
    prob_vol = np.zeros((h, w, z), dtype=np.float32)

    for slice_idx in range(z):
        x, (orig_h, orig_w) = _prepare_input_tensor(
            proxies=proxies,
            slice_idx=slice_idx,
            image_size=image_size,
            norm_stats=norm_stats,
            device=device,
        )

        pred, unc, prob = mc_predict_slice(model, x, n_iterations=n_iterations, method=method)

        # Resize back to original H/W.
        if (orig_h, orig_w) != (image_size, image_size):
            pred = F.interpolate(pred, size=(orig_h, orig_w), mode="nearest")
            unc = F.interpolate(unc, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            prob = F.interpolate(prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        pred_vol[:, :, slice_idx] = pred[0, 0].detach().cpu().numpy().astype(np.uint8)
        unc_vol[:, :, slice_idx] = unc[0, 0].detach().cpu().numpy().astype(np.float32)
        prob_vol[:, :, slice_idx] = prob[0, 0].detach().cpu().numpy().astype(np.float32)

    return pred_vol, unc_vol, prob_vol, seg_ref_img


def save_nifti_like(ref_img: nib.Nifti1Image, data: np.ndarray, save_path: Path) -> None:
    """
    Save `data` as a NIfTI aligned to `ref_img` (same affine + header template).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    hdr = ref_img.header.copy()
    nii = nib.Nifti1Image(data, affine=ref_img.affine, header=hdr)
    nib.save(nii, str(save_path))
