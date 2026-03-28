from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


INCLUDE_SUFFIXES = {".py", ".yaml", ".md", ".toml", ".txt", ".ipynb"}
INCLUDE_FILES = {".gitignore"}
EXCLUDE_PARTS = {"runs", "__pycache__", ".ipynb_checkpoints", ".venv"}


def should_include(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    return path.name in INCLUDE_FILES or path.suffix in INCLUDE_SUFFIXES


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    zip_path = repo_root / "scripts.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(repo_root.rglob("*")):
            if path.is_file() and should_include(path):
                archive.write(path, arcname=path.relative_to(repo_root))
    print(f"Created {zip_path}")


if __name__ == "__main__":
    main()
