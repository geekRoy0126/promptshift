"""Download and extract datasets into <storage root>/data/<name>/.

Usage:
    python scripts/download_data.py --dataset kvasir_seg [--root PATH] [--force]

Idempotent: skips the download if the archive exists and skips extraction if the
extracted folder exists. Prints the archive's sha256 so it can be logged.
More datasets (COCO, ISIC, BUSI) are added to SOURCES by their owners in P1-5.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from promptshift.paths import get_paths

SOURCES: dict[str, dict] = {
    "kvasir_seg": {
        "url": "https://datasets.simula.no/downloads/kvasir-seg.zip",
        "archive": "kvasir-seg.zip",
        "extracted": "Kvasir-SEG",  # top-level folder inside the zip
    },
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while block := resp.read(1 << 20):
            out.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    if total and done != total:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"incomplete download: {done} of {total} bytes")
    tmp.replace(dest)


def safe_extract(archive: Path, target: Path) -> None:
    """Extract a zip, refusing entries that would escape the target folder."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            dest = (target / name).resolve()
            if target not in dest.parents and dest != target:
                raise RuntimeError(f"unsafe path in archive: {name}")
        zf.extractall(target)


def fetch(name: str, data_dir: Path, force: bool = False) -> Path:
    src = SOURCES[name]
    out_dir = data_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / src["archive"]
    extracted = out_dir / src["extracted"]
    if force:
        archive.unlink(missing_ok=True)
        shutil.rmtree(extracted, ignore_errors=True)
    if not archive.exists():
        download(src["url"], archive)
    else:
        print(f"archive exists: {archive}")
    print(f"sha256 {sha256_of(archive)}  {archive.name}")
    if not extracted.exists():
        print(f"extracting to {out_dir}")
        safe_extract(archive, out_dir)
    else:
        print(f"already extracted: {extracted}")
    if not extracted.is_dir():
        raise RuntimeError(f"expected folder {extracted} after extraction")
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", required=True, choices=sorted(SOURCES))
    parser.add_argument("--root", type=Path, default=None, help="storage root (default: paths.py)")
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    args = parser.parse_args()
    paths = get_paths(args.root).ensure()
    folder = fetch(args.dataset, paths.data, args.force)
    print(f"ready: {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
