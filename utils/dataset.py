import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from skimage.transform import resize

class BraTSDataset(Dataset):
    """
    針對 BraTS 資料集的 PyTorch Dataset 類別。
    支援載入多模態影像 (FLAIR, T1, T1ce, T2) 與標籤。
    """
    def __init__(self, data_dir, patient_ids, transform=None, mode='train'):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_path = os.path.join(self.data_dir, patient_id)
        
        # BraTS 四種序列
        modalities = ['flair', 't1', 't1ce', 't2']
        images = []
        
        for mod in modalities:
            path = os.path.join(patient_path, f"{patient_id}_{mod}.nii.gz")
            img = nib.load(path).get_fdata()
            
            # 選取中間切片
            slice_idx = img.shape[2] // 2
            img_slice = img[:, :, slice_idx]
            
            # 正規化
            img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)
            img_slice = resize(img_slice, (128, 128), preserve_range=True)
            images.append(img_slice)
            
        # 堆疊為 4 通道影像 (C, H, W)
        image = np.stack(images, axis=0) # (4, 128, 128)
        
        # 載入標籤
        mask_path = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
        mask_img = nib.load(mask_path).get_fdata()
        mask_slice = mask_img[:, :, mask_img.shape[2] // 2]
        mask = resize(mask_slice, (128, 128), order=0, preserve_range=True)
        mask = (mask > 0).astype(np.float32)
        
        if self.transform:
            # 注意：albumentations 預設處理 HWC，需轉換
            image_hwc = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image_hwc, mask=mask)
            image = np.transpose(augmented['image'], (2, 0, 1))
            mask = augmented['mask']
        
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image, mask
