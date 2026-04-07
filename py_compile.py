"""
Project-local `py_compile` shim (no bytecode writes).

Why this exists:
- Some Windows environments (AV / policy) block atomic renames and/or writing `.pyc` files, causing
  `python -m compileall` / `python -m py_compile` to fail even when source code is valid.
- This project uses compile checks as a *syntax verification* step only. Bytecode cache files are
  not part of the deliverable and should not be created in the repo.

Behavior:
- `compile()` compiles source in-memory and never writes `.pyc` files.

This keeps `compileall`/`py_compile` usable as a final verification step while maintaining a clean
workspace and release boundary.
"""

from __future__ import annotations

import builtins
import tokenize
from enum import Enum
from pathlib import Path


class PyCompileError(Exception):
    def __init__(self, exc_type, exc_value, file: str, msg: str | None = None):
        self.exc_type = exc_type
        self.exc_value = exc_value
        self.file = file
        self.msg = msg
        super().__init__(self.__str__())

    def __str__(self) -> str:
        base = f"Failed to compile {self.file}"
        if self.msg:
            return f"{base}: {self.msg}"
        return base


def compile(  # noqa: A001 - match stdlib name
    file: str,
    cfile: str | None = None,
    dfile: str | None = None,
    doraise: bool = False,
    optimize: int = -1,
    **_kwargs,
):
    """
    Compile `file` in-memory and return a plausible output path (without writing).
    """
    path = Path(file)
    try:
        with tokenize.open(str(path)) as f:
            src = f.read()
        builtins.compile(src, dfile or str(path), "exec", dont_inherit=True, optimize=optimize)
        return cfile or str(path)
    except Exception as e:  # noqa: BLE001 - stdlib-compatible behavior
        if doraise:
            raise
        raise PyCompileError(type(e), e, str(path), msg=str(e))


class PycInvalidationMode(Enum):
    # Keep names/values compatible with stdlib for compileall argument parsing.
    TIMESTAMP = "timestamp"
    CHECKED_HASH = "checked-hash"
    UNCHECKED_HASH = "unchecked-hash"
