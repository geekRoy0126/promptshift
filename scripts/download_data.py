"""Download and extract datasets into <storage root>/data/<name>/.

Usage:
    python scripts/download_data.py --dataset kvasir_seg [--root PATH] [--force]
    python scripts/download_data.py --dataset coco_val2017 [--skip-images]

Idempotent: skips the download if the archive exists and skips extraction if the
expected output exists. Prints each archive's sha256 so it can be logged.
ISIC (owner B) and BUSI (owner C) are added to SOURCES by their owners in P1-5.
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

# Each archive: url, archive file name, the path that must exist after extraction,
# optional `members` (extract only these), optional `images: True` (skippable).
SOURCES: dict[str, list[dict]] = {
    "kvasir_seg": [
        {
            "url": "https://datasets.simula.no/downloads/kvasir-seg.zip",
            "archive": "kvasir-seg.zip",
            "extracted": "Kvasir-SEG",
        },
    ],
    "coco_val2017": [
        {
            "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "archive": "annotations_trainval2017.zip",
            "extracted": "annotations/instances_val2017.json",
            "members": ["annotations/instances_val2017.json"],
        },
        {
            "url": "http://images.cocodataset.org/zips/val2017.zip",
            "archive": "val2017.zip",
            "extracted": "val2017",
            "images": True,
        },
    ],
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


def safe_extract(archive: Path, target: Path, members: list[str] | None = None) -> None:
    """Extract a zip, refusing entries that would escape the target folder."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        names = members if members is not None else zf.namelist()
        for name in names:
            dest = (target / name).resolve()
            if target not in dest.parents and dest != target:
                raise RuntimeError(f"unsafe path in archive: {name}")
        zf.extractall(target, members=names)


def fetch_archive(src: dict, out_dir: Path, force: bool) -> Path:
    archive = out_dir / src["archive"]
    extracted = out_dir / src["extracted"]
    if force:
        archive.unlink(missing_ok=True)
        if extracted.is_dir():
            shutil.rmtree(extracted)
        else:
            extracted.unlink(missing_ok=True)
    if extracted.exists() and not archive.exists():
        print(f"already extracted (archive removed): {extracted}")
        return extracted
    if not archive.exists():
        download(src["url"], archive)
    else:
        print(f"archive exists: {archive}")
    print(f"sha256 {sha256_of(archive)}  {archive.name}")
    if not extracted.exists():
        print(f"extracting {archive.name} to {out_dir}")
        safe_extract(archive, out_dir, src.get("members"))
    else:
        print(f"already extracted: {extracted}")
    if not extracted.exists():
        raise RuntimeError(f"expected {extracted} after extraction")
    return extracted


def fetch(name: str, data_dir: Path, force: bool = False, skip_images: bool = False) -> Path:
    out_dir = data_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in SOURCES[name]:
        if skip_images and src.get("images"):
            print(f"skipping image archive {src['archive']} (--skip-images)")
            continue
        fetch_archive(src, out_dir, force)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", required=True, choices=sorted(SOURCES))
    parser.add_argument("--root", type=Path, default=None, help="storage root (default: paths.py)")
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    parser.add_argument("--skip-images", action="store_true", help="skip large image archives")
    args = parser.parse_args()
    paths = get_paths(args.root).ensure()
    folder = fetch(args.dataset, paths.data, args.force, args.skip_images)
    print(f"ready: {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
