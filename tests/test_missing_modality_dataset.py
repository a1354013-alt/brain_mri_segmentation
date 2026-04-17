"""
Tests for missing modality simulation in BraTSDataset.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


class TestMissingModalityDataset(unittest.TestCase):
    """Test missing modality simulation functionality."""

    def test_sample_missing_modalities_none_policy(self):
        """Test that 'none' policy returns no missing modalities."""
        from utils.dataset import BraTSDataset
        
        # Create a mock dataset with none policy
        with patch.object(BraTSDataset, '_prepare_dataset', return_value=None):
            ds = BraTSDataset(
                data_dir=Path("/fake"),
                patient_ids=["p1"],
                enable_missing_modality=False,
                missing_modality_policy="none",
            )
            ds.valid_patient_ids = ["p1"]
            ds.patient_cache = {"p1": {"tumor_slice_indices": [0], "val_best_slice_idx": 0}}
        
        missing_flags = ds._sample_missing_modalities()
        self.assertEqual(missing_flags, [False, False, False, False])

    def test_sample_missing_modalities_fixed_policy(self):
        """Test that 'fixed' policy correctly marks specified modalities as missing."""
        from utils.dataset import BraTSDataset
        
        with patch.object(BraTSDataset, '_prepare_dataset', return_value=None):
            ds = BraTSDataset(
                data_dir=Path("/fake"),
                patient_ids=["p1"],
                enable_missing_modality=True,
                missing_modality_policy="fixed",
                fixed_missing_modalities=["flair"],
                random_seed=42,
            )
            ds.valid_patient_ids = ["p1"]
            ds.patient_cache = {"p1": {"tumor_slice_indices": [0], "val_best_slice_idx": 0}}
        
        missing_flags = ds._sample_missing_modalities()
        # flair is index 0
        self.assertTrue(missing_flags[0])
        self.assertFalse(missing_flags[1])  # t1
        self.assertFalse(missing_flags[2])  # t1ce
        self.assertFalse(missing_flags[3])  # t2

    def test_sample_missing_modalities_random_one_policy(self):
        """Test that 'random_one' policy drops exactly one modality."""
        from utils.dataset import BraTSDataset
        
        with patch.object(BraTSDataset, '_prepare_dataset', return_value=None):
            ds = BraTSDataset(
                data_dir=Path("/fake"),
                patient_ids=["p1"],
                enable_missing_modality=True,
                missing_modality_policy="random_one",
                random_seed=42,
            )
            ds.valid_patient_ids = ["p1"]
            ds.patient_cache = {"p1": {"tumor_slice_indices": [0], "val_best_slice_idx": 0}}
        
        # Run multiple times to verify always exactly one is missing
        for _ in range(10):
            missing_flags = ds._sample_missing_modalities()
            self.assertEqual(sum(missing_flags), 1, "Exactly one modality should be missing")

    def test_sample_missing_modalities_reproducible(self):
        """Test that missing modality sampling is reproducible with same seed."""
        from utils.dataset import BraTSDataset
        
        results1 = []
        results2 = []
        
        for _ in range(2):
            with patch.object(BraTSDataset, '_prepare_dataset', return_value=None):
                ds = BraTSDataset(
                    data_dir=Path("/fake"),
                    patient_ids=["p1"],
                    enable_missing_modality=True,
                    missing_modality_policy="random_one",
                    random_seed=123,
                )
                ds.valid_patient_ids = ["p1"]
                ds.patient_cache = {"p1": {"tumor_slice_indices": [0], "val_best_slice_idx": 0}}
            
            results1.append(ds._sample_missing_modalities())
        
        for _ in range(2):
            with patch.object(BraTSDataset, '_prepare_dataset', return_value=None):
                ds = BraTSDataset(
                    data_dir=Path("/fake"),
                    patient_ids=["p1"],
                    enable_missing_modality=True,
                    missing_modality_policy="random_one",
                    random_seed=123,
                )
                ds.valid_patient_ids = ["p1"]
                ds.patient_cache = {"p1": {"tumor_slice_indices": [0], "val_best_slice_idx": 0}}
            
            results2.append(ds._sample_missing_modalities())
        
        self.assertEqual(results1[0], results2[0])
        self.assertEqual(results1[1], results2[1])

    def test_modality_mask_correct_format(self):
        """Test that modality_mask has correct format (1=present, 0=missing)."""
        import torch
        from utils.dataset import BraTSDataset
        
        with patch.object(BraTSDataset, '_prepare_dataset', return_value=None):
            ds = BraTSDataset(
                data_dir=Path("/fake"),
                patient_ids=["p1"],
                enable_missing_modality=True,
                missing_modality_policy="fixed",
                fixed_missing_modalities=["t1ce"],
                random_seed=42,
            )
            ds.valid_patient_ids = ["p1"]
            ds.patient_cache = {"p1": {"tumor_slice_indices": [0], "val_best_slice_idx": 0}}
        
        missing_flags = ds._sample_missing_modalities()
        modality_mask = torch.tensor([0.0 if m else 1.0 for m in missing_flags], dtype=torch.float32)
        
        # t1ce is index 2, should be 0 (missing)
        self.assertEqual(modality_mask.shape, (4,))
        self.assertEqual(modality_mask[2].item(), 0.0)
        self.assertEqual(modality_mask[0].item(), 1.0)  # flair present
        self.assertEqual(modality_mask[1].item(), 1.0)  # t1 present
        self.assertEqual(modality_mask[3].item(), 1.0)  # t2 present


if __name__ == "__main__":
    unittest.main()
