"""
Create a clean source zip for distribution.

This script avoids bundling local/dev artifacts such as:
- .git/
- __pycache__/, *.pyc
- outputs/, data/
- archive/ (historical only)

Usage:
  python scripts/make_release_zip.py --out brain_mri_segmentation_src.zip

Output location:
- All release zips are written to `dist/` (created if missing).
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import zipfile
from pathlib import Path


DEFAULT_EXCLUDES = [
    ".git/*",
    "**/.git/*",
    "archive/*",
    "**/archive/*",
    "__pycache__/*",
    "**/__pycache__/*",
    "**/*.pyc",
    "**/*.pyc*",
    "**/*.pyo",
    "**/*.pyo*",
    "**/*.pth",
    "**/*.pt",
    "**/*.png",
    "tests/_tmp*/*",
    "tests/_tmp*",
    "dist/*",
    "**/dist/*",
    "*.zip",
    "**/*.zip",
    "outputs/*",
    "data/*",
    "build/*",
    "**/build/*",
    ".venv/*",
    "venv/*",
    "_kaggle_download/*",
    "scripts/_patch_*",
    "scripts/_patch_*/*",
    "scripts/_patch_test.txt",
    "**/align_conflicts.txt",
    "**/skipped_patients.txt",
    "**/prepared_cache_missing_*.txt",
    ".ruff_cache/*",
    ".pytest_cache/*",
    "**/.pytest_cache/*",
    ".mypy_cache/*",
    "**/.mypy_cache/*",
    ".DS_Store",
    "**/.DS_Store",
    "Thumbs.db",
    "**/Thumbs.db",
]


def _match_any(rel_posix: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(rel_posix, pat):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a clean source zip for this repo.")
    parser.add_argument("--out", type=str, default="brain_mri_segmentation_src.zip")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if dist/ contains non-deliverable artifacts that cannot be cleaned.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dist_dir = repo_root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    # Enforce release output under dist/ to avoid zip-in-zip and repo root clutter.
    out_path = (dist_dir / Path(args.out).name).resolve()
    # Ensure dist/ contains a single unambiguous deliverable: remove any other items.
    for p in list(dist_dir.iterdir()):
        if p.resolve() == out_path:
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
        except Exception:
            # Best-effort only. Some environments block deletions (AV / policy). We don't hard-fail packaging.
            pass

    remaining = [p.name for p in dist_dir.iterdir() if p.resolve() != out_path]
    if remaining:
        msg = "Warning: dist/ contains non-deliverable artifacts and could not be cleaned automatically:"
        if args.strict:
            msg = msg.replace("Warning", "Error")
        print(msg)
        for name in sorted(remaining):
            print(f"  - dist/{name}")
        print("Suggested: python scripts/clean_project.py --clean-dist")
        if args.strict:
            raise SystemExit(2)

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
