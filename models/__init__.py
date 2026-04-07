from __future__ import annotations

from importlib import import_module

__all__ = ["AttentionUNet", "UNet"]


def __getattr__(name: str):
    if name == "AttentionUNet":
        return import_module(".attention_unet", __name__).AttentionUNet
    if name == "UNet":
        return import_module(".unet", __name__).UNet
    raise AttributeError(name)
