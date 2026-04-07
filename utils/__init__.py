from __future__ import annotations

from importlib import import_module

__all__ = [
    "BraTSDataset",
    "mc_dropout_inference",
    "plot_results_with_uncertainty",
    "predict_patient_volume",
    "save_nifti_like",
]


def __getattr__(name: str):
    if name == "BraTSDataset":
        return import_module(".dataset", __name__).BraTSDataset
    if name in {"mc_dropout_inference", "plot_results_with_uncertainty"}:
        m = import_module(".visualize", __name__)
        return getattr(m, name)
    if name in {"predict_patient_volume", "save_nifti_like"}:
        m = import_module(".inference", __name__)
        return getattr(m, name)
    raise AttributeError(name)
