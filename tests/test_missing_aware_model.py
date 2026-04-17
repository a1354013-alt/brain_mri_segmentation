"""
Tests for missing-aware model (ModalityAwareUNet).
"""

import unittest

import torch


class TestMissingAwareModel(unittest.TestCase):
    """Test ModalityAwareUNet functionality."""

    def test_model_init(self):
        """Test that model initializes correctly."""
        from models.missing_modality_unet import ModalityAwareUNet
        
        model = ModalityAwareUNet(n_modalities=4, n_classes=1, dropout_p=0.2)
        self.assertIsNotNone(model)

    def test_forward_without_mask(self):
        """Test forward pass without modality_mask."""
        from models.missing_modality_unet import ModalityAwareUNet
        
        model = ModalityAwareUNet(n_modalities=4, n_classes=1, dropout_p=0.2)
        model.eval()
        
        x = torch.randn(1, 4, 32, 32)
        with torch.no_grad():
            out = model(x)
        
        self.assertEqual(out.shape, (1, 1, 32, 32))

    def test_forward_with_mask(self):
        """Test forward pass with modality_mask."""
        from models.missing_modality_unet import ModalityAwareUNet
        
        model = ModalityAwareUNet(n_modalities=4, n_classes=1, dropout_p=0.2)
        model.eval()
        
        x = torch.randn(1, 4, 32, 32)
        mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])  # t1ce missing
        
        with torch.no_grad():
            out = model(x, modality_mask=mask)
        
        self.assertEqual(out.shape, (1, 1, 32, 32))

    def test_forward_batch_size_variations(self):
        """Test forward pass with different batch sizes."""
        from models.missing_modality_unet import ModalityAwareUNet
        
        model = ModalityAwareUNet(n_modalities=4, n_classes=1, dropout_p=0.2)
        model.eval()
        
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 4, 32, 32)
            mask = torch.ones(batch_size, 4)
            
            with torch.no_grad():
                out = model(x, modality_mask=mask)
            
            self.assertEqual(out.shape, (batch_size, 1, 32, 32))

    def test_baseline_vs_missing_aware_interface_compatibility(self):
        """Test that both baseline and missing-aware models can be called similarly."""
        from models.attention_unet import AttentionUNet
        from models.missing_modality_unet import ModalityAwareUNet
        
        baseline = AttentionUNet(n_channels=4, n_classes=1, dropout_p=0.2)
        missing_aware = ModalityAwareUNet(n_modalities=4, n_classes=1, dropout_p=0.2)
        
        baseline.eval()
        missing_aware.eval()
        
        x = torch.randn(1, 4, 32, 32)
        
        with torch.no_grad():
            out_baseline = baseline(x)
            out_missing = missing_aware(x)  # Without mask should work
            out_missing_with_mask = missing_aware(x, modality_mask=torch.ones(1, 4))
        
        self.assertEqual(out_baseline.shape, (1, 1, 32, 32))
        self.assertEqual(out_missing.shape, (1, 1, 32, 32))
        self.assertEqual(out_missing_with_mask.shape, (1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
