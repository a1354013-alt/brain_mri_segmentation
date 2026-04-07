"""
Project cleanup utility.

Removes Python bytecode caches that should never be part of a release artifact:
- __pycache__ directories
- *.pyc / *.pyo (including variant suffixes like *.pyc.*)
- tests/_tmp* directories
- repo-root *.zip files (old/manual zips)
- optional: dist/*.zip (old release artifacts)

Usage:
  python scripts/clean_project.py
  python scripts/clean_project.py --dry-run
  python scripts/clean_project.py --clean-dist
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _iter_bytecode_files(repo_root: Path):
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith(".pyc") or name.endswith(".pyo") or ".pyc." in name or ".pyo." in name:
            yield p


def _collect_artifacts(repo_root: Path):
    artifacts = []
    # Temp test dirs
    artifacts.extend([p for p in repo_root.glob("tests/_tmp*") if p.exists()])
    # Bytecode caches
    artifacts.extend([p for p in repo_root.rglob("__pycache__") if p.is_dir()])
    artifacts.extend(list(_iter_bytecode_files(repo_root)))
    # Manual zips at repo root
    artifacts.extend([p for p in repo_root.glob("*.zip") if p.exists()])
    # dist should only contain release zip(s)
    dist_dir = repo_root / "dist"
    if dist_dir.exists():
        artifacts.extend([p for p in dist_dir.glob("pycacheprefix") if p.exists()])
        artifacts.extend([p for p in dist_dir.glob("_compile_tmp_*") if p.exists()])
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Python bytecode caches from this repo.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be removed without deleting.")
    parser.add_argument(
        "--clean-dist",
        action="store_true",
        help="Also remove dist/*.zip (old release artifacts). dist/ itself is kept.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail (exit non-zero) if artifacts remain after cleanup (useful in CI).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    removed = 0
    failed = 0

    # dist/pycacheprefix and other non-deliverable dist artifacts should never persist.
    dist_dir = repo_root / "dist"
    dist_pycache = dist_dir / "pycacheprefix"
    if dist_pycache.exists():
        if args.dry_run:
            print(f"[dry-run] rmtree {dist_pycache}")
        else:
            try:
                shutil.rmtree(dist_pycache, ignore_errors=False)
                removed += 1
            except Exception as e:
                failed += 1
                print(f"Warning: failed to remove {dist_pycache}: {e}")

    # Remove known build/temp caches (best-effort).
    for d in [repo_root / ".pytest_cache", repo_root / ".mypy_cache", repo_root / ".ruff_cache", repo_root / "build"]:
        if not d.exists():
            continue
        if args.dry_run:
            print(f"[dry-run] rmtree {d}")
            continue
        try:
            shutil.rmtree(d, ignore_errors=False)
            removed += 1
        except Exception as e:
            failed += 1
            print(f"Warning: failed to remove {d}: {e}")

    # Remove repo-local test temp dirs.
    for d in repo_root.glob("tests/_tmp*"):
        if not d.exists():
            continue
        if args.dry_run:
            print(f"[dry-run] rmtree {d}")
            continue
        try:
            shutil.rmtree(d, ignore_errors=False)
            removed += 1
        except Exception as e:
            failed += 1
            print(f"Warning: failed to remove {d}: {e}")

    # Remove repo-root zips (manual artifacts). Release zips should live under dist/.
    for z in repo_root.glob("*.zip"):
        if args.dry_run:
            print(f"[dry-run] unlink {z}")
            continue
        try:
            z.unlink(missing_ok=True)
            removed += 1
        except Exception as e:
            failed += 1
            print(f"Warning: failed to remove {z}: {e}")

    # Optional dist cleanup: remove old zips, keep the folder.
    if args.clean_dist:
        for z in dist_dir.glob("*.zip"):
            if args.dry_run:
                print(f"[dry-run] unlink {z}")
                continue
            try:
                z.unlink(missing_ok=True)
                removed += 1
            except Exception as e:
                failed += 1
                print(f"Warning: failed to remove {z}: {e}")

        # Also remove any non-zip files/dirs left in dist/ (deliverable should be a single zip).
        if dist_dir.exists():
            for p in dist_dir.iterdir():
                if p.is_file() and p.suffix.lower() == ".zip":
                    continue
                if args.dry_run:
                    print(f"[dry-run] remove {p}")
                    continue
                try:
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=False)
                    else:
                        p.unlink(missing_ok=True)
                    removed += 1
                except Exception as e:
                    failed += 1
                    print(f"Warning: failed to remove {p}: {e}")

    # Remove __pycache__ directories first (covers most pyc files).
    for d in repo_root.rglob("__pycache__"):
        if not d.is_dir():
            continue
        if args.dry_run:
            print(f"[dry-run] rmtree {d}")
            continue
        try:
            shutil.rmtree(d, ignore_errors=False)
            removed += 1
        except Exception as e:
            failed += 1
            print(f"Warning: failed to remove {d}: {e}")

    # Remove any remaining loose bytecode files (including suffix variants).
    for f in _iter_bytecode_files(repo_root):
        if args.dry_run:
            print(f"[dry-run] unlink {f}")
            continue
        try:
            f.unlink(missing_ok=True)
            removed += 1
        except Exception as e:
            failed += 1
            print(f"Warning: failed to remove {f}: {e}")

    msg = f"Cleanup complete. removed={removed}"
    if failed:
        msg += f" failed={failed} (some files may be locked by the OS/AV)"
    print(msg)

    remaining = _collect_artifacts(repo_root)
    if remaining:
        header = "\nWarning: project is still dirty after cleanup (some artifacts could not be removed):"
        if args.strict:
            header = header.replace("Warning", "Error")
        print(header)
        for p in sorted({str(p) for p in remaining}):
            print(f"- {p}")
        if args.strict:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
