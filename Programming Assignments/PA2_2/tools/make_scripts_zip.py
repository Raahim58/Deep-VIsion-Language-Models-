"""
Zip all source .py / .yaml / .md / .txt / .cfg / .toml files into scripts.zip.

Usage:
    python tools/make_scripts_zip.py [--out scripts.zip]
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

INCLUDE_EXTENSIONS = {".py", ".yaml", ".yml", ".md", ".txt", ".cfg", ".toml", ".gitignore"}
EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", "runs", "wandb", ".eggs"}


def collect_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file():
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            if p.suffix in INCLUDE_EXTENSIONS or p.name in {".gitignore"}:
                files.append(p)
    return sorted(files)


def make_zip(root: Path, out: Path):
    files = collect_files(root)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arc_name = f.relative_to(root)
            zf.write(f, arc_name)
            print(f"  + {arc_name}")
    print(f"\n[ZIP] Created {out}  ({out.stat().st_size / 1024:.1f} KB, {len(files)} files)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",  default="scripts.zip")
    p.add_argument("--root", default=".")
    args = p.parse_args()
    make_zip(Path(args.root).resolve(), Path(args.out))


if __name__ == "__main__":
    main()
