"""
Project-wide Python runtime tweaks.

Why this file exists:
- On some Windows environments, writing .pyc files under __pycache__ can fail with "Access denied"
  (often due to AV / controlled folder access / corporate policies). Failing to write bytecode is
  not a functional requirement for this project, so we disable it to avoid noisy errors.
- Some terminals default to legacy encodings (e.g. cp950) which can crash when scripts print
  non-encodable characters. We reconfigure stdout/stderr to UTF-8 when possible.

Python automatically imports `sitecustomize` (if present on sys.path) during interpreter startup.
Putting this file in repo root makes it active for all project commands executed from this folder.
"""

from __future__ import annotations

import os
import sys


def _try_force_utf8_stdio() -> None:
    # Best-effort only. If the host forbids reconfigure, we don't hard-fail.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


# This module is auto-imported by Python when present. To avoid surprising global behavior changes,
# all tweaks are opt-in via environment variables.

# Disable writing .pyc files (helps when __pycache__ writes are blocked).
if _env_flag("BMS_DONT_WRITE_BYTECODE"):
    sys.dont_write_bytecode = True

# Force UTF-8 stdio for terminals that cannot encode some characters.
if _env_flag("BMS_FORCE_UTF8"):
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    _try_force_utf8_stdio()
