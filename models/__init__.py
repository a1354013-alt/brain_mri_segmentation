from __future__ import annotations

from importlib import import_module

__all__ = ["AttentionUNet", "UNet", "ModalityAwareUNet"]


def __getattr__(name: str):
    if name == "AttentionUNet":
        return import_module(".attention_unet", __name__).AttentionUNet
    if name == "UNet":
        return import_module(".unet", __name__).UNet
    if name == "ModalityAwareUNet":
        return import_module(".missing_modality_unet", __name__).ModalityAwareUNet
    raise AttributeError(name)
