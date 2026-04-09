"""
Project-local `py_compile` shim that performs syntax checks without writing bytecode.

Why this exists:
- Some Windows environments (AV / policy) block atomic renames and/or writing `.pyc`
  files, causing `python -m compileall` / `python -m py_compile` to fail even when
  source code is valid.
- This project uses compile checks as a syntax verification step only. Bytecode cache
  files are not part of the deliverable and should not be created in the repo.
"""

from __future__ import annotations

import argparse
import builtins
import sys
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


def compile(  # noqa: A001 - keep stdlib-compatible API
    file: str,
    cfile: str | None = None,
    dfile: str | None = None,
    doraise: bool = False,
    optimize: int = -1,
    **_kwargs,
):
    """
    Compile `file` in-memory and return a plausible output path without writing `.pyc`.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Syntax-check Python files without writing bytecode.")
    parser.add_argument("files", nargs="+", help="Python source files to compile-check.")
    parser.add_argument("-d", dest="dfile", default=None)
    parser.add_argument("-O", dest="optimize", action="count", default=0)
    args = parser.parse_args(argv)

    optimize = int(args.optimize or 0)
    failed = False
    for file_name in args.files:
        try:
            compile(file_name, dfile=args.dfile, doraise=True, optimize=optimize)
        except Exception as e:  # noqa: BLE001 - CLI parity with stdlib
            print(e, file=sys.stderr)
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
