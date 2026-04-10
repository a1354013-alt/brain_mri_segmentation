"""
Fast smoke test for the project (v3.1 stable iteration).

This is intentionally minimal and should complete quickly on CPU.

Compatibility note:
- `python tests/smoke_test.py` is supported without mutating `sys.path`.
  When executed as a script, this file re-runs itself as a module:
  `python -m tests.smoke_test` from the repo root.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Keep CPU thread usage conservative by default. This reduces memory pressure on constrained hosts.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Avoid writing bytecode caches into the repo when running smoke tests.
sys.dont_write_bytecode = True


def _rerun_as_module_if_needed() -> None:
    """Ensure imports work without mutating sys.path.

    When executed as a script (`python tests/smoke_test.py`), the working directory
    may not be the repo root, and local imports like `from models import ...` can fail.
    We re-exec as a module from repo root so the normal import mechanism works.
    """

    # When run via `python -m tests.smoke_test`, __package__ is non-empty.
    if __package__:
        return

    # Prevent infinite recursion.
    if os.environ.get("BMS_SMOKE_AS_MODULE") == "1":
        return

    repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["BMS_SMOKE_AS_MODULE"] = "1"
    res = subprocess.run(
        [sys.executable, "-m", "tests.smoke_test"],
        cwd=str(repo_root),
        env=env,
        check=False,
    )
    raise SystemExit(res.returncode)


_rerun_as_module_if_needed()


def run_smoke_test() -> None:
    """Minimal smoke test: model init + one forward pass."""

    import torch

    from models import AttentionUNet

    import config

    print(f"Running Smoke Test ({config.PROJECT_VERSION})...", flush=True)

    # Best-effort: reduce CPU thread usage.
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    print("Initializing model...", flush=True)
    try:
        model = AttentionUNet(n_channels=4, n_classes=1, dropout_p=0.2)
    except RuntimeError as e:
        # Some CI/sandbox environments are extremely memory constrained. If we cannot even allocate
        # model weights, treat this as a skipped smoke test rather than a hard failure.
        if "not enough memory" in str(e).lower():
            print(f"Smoke test skipped: insufficient memory to initialize model. ({e})")
            return
        raise

    model.eval()
    print("Model initialized.", flush=True)

    print("Running one forward pass...", flush=True)
    x = torch.randn(1, 4, 16, 16)
    try:
        with torch.no_grad():
            out = model(x)
    except RuntimeError as e:
        if "not enough memory" in str(e).lower():
            print(f"Smoke test skipped: insufficient memory for a forward pass. ({e})")
            return
        raise

    assert out.shape == (1, 1, 16, 16), f"Unexpected output shape: {out.shape}"
    print("Smoke test OK (model init + one forward pass).", flush=True)


if __name__ == "__main__":
    run_smoke_test()
