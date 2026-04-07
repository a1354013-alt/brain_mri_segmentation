"""
Create a clean source zip for distribution.

This script avoids bundling local/dev artifacts such as:
- .git/
- __pycache__/, *.pyc
- outputs/, data/

Usage:
  python scripts/make_release_zip.py --out brain_mri_segmentation_src.zip
"""

from __future__ import annotations

import argparse
import fnmatch
import zipfile
from pathlib import Path


DEFAULT_EXCLUDES = [
    ".git/*",
    "**/.git/*",
    "__pycache__/*",
    "**/__pycache__/*",
    "**/*.pyc",
    "**/*.pyc*",
    "tests/_tmp_*/*",
    "tests/_tmp_*",
    "outputs/*",
    "data/*",
    ".venv/*",
    "venv/*",
    "_kaggle_download/*",
    "scripts/_patch_*",
    "scripts/_patch_*/*",
    ".ruff_cache/*",
]


def _match_any(rel_posix: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(rel_posix, pat):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a clean source zip for this repo.")
    parser.add_argument("--out", type=str, default="brain_mri_segmentation_src.zip")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_path = Path(args.out).resolve()

    files = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        # Do not include the output zip itself if it lives inside the repo.
        if p.resolve() == out_path:
            continue
        rel = p.relative_to(repo_root).as_posix()
        if _match_any(rel, DEFAULT_EXCLUDES):
            continue
        files.append((p, rel))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for abs_path, rel in files:
            zf.write(abs_path, arcname=rel)

    print(f"Wrote {out_path} with {len(files)} files.")


if __name__ == "__main__":
    main()
