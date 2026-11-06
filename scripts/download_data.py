"""Download and extract datasets into <storage root>/data/<name>/.

Usage:
    python scripts/download_data.py --dataset kvasir_seg [--root PATH] [--force]
    python scripts/download_data.py --dataset coco_val2017 [--skip-images]
    python scripts/download_data.py --dataset isic2018 [--skip-images]
    python scripts/download_data.py --dataset busi

Idempotent: skips the download if the archive exists and skips extraction if the
expected output exists. Prints each archive's sha256 so it can be logged.
ISIC 2018: official training set only (masks 27 MB, images 11 GB). With --skip-images only the
member list of the image zip is saved (read with HTTP range requests, no full download).
BUSI: downloaded from Kaggle with kagglehub (public dataset; credentials, if needed, come from
KAGGLE_USERNAME / KAGGLE_KEY or Colab secrets; nothing is printed or saved).
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
    "isic2018": [
        {
            "url": "https://isic-archive.s3.amazonaws.com/challenges/2018/"
            "ISIC2018_Task1_Training_GroundTruth.zip",
            "archive": "ISIC2018_Task1_Training_GroundTruth.zip",
            "extracted": "ISIC2018_Task1_Training_GroundTruth",
        },
        {
            "url": "https://isic-archive.s3.amazonaws.com/challenges/2018/"
            "ISIC2018_Task1-2_Training_Input.zip",
            "archive": "ISIC2018_Task1-2_Training_Input.zip",
            "extracted": "ISIC2018_Task1-2_Training_Input",
            "images": True,
        },
    ],
}
KAGGLE = {"busi": ("aryashah2k/breast-ultrasound-images-dataset", "Dataset_BUSI_with_GT")}


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


def tree_hash(folder: Path) -> tuple[int, str]:
    """(number of files, sha256 over sorted relative paths + file hashes)."""
    h = hashlib.sha256()
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    for p in files:
        h.update(p.relative_to(folder).as_posix().encode() + b"|" + sha256_of(p).encode())
    return len(files), h.hexdigest()


def fetch_kaggle(name: str, out_dir: Path, force: bool) -> None:
    handle, folder = KAGGLE[name]
    target = out_dir / folder
    if target.is_dir() and force:
        shutil.rmtree(target)
    if not target.is_dir():
        import kagglehub

        src = Path(kagglehub.dataset_download(handle))
        print(f"kagglehub: {handle} -> {src}")  # path ends in versions/<n>
        if not (src / folder).is_dir():
            raise RuntimeError(f"{folder} not found in {src}")
        shutil.copytree(src / folder, target)
    else:
        print(f"already present: {target}")
    n, digest = tree_hash(target)
    print(f"tree sha256 {digest}  ({n} files in {folder})")


def save_remote_index(src: dict, out_dir: Path) -> None:
    """Member list of a remote zip (central directory only, via range requests)."""
    import json

    from promptshift.data.remote_zip import open_remote_zip

    index = out_dir / (src["extracted"] + ".index.json")
    if index.is_file():
        print(f"remote index exists: {index}")
        return
    with open_remote_zip(src["url"]) as zf:
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
    index.write_text(json.dumps(names, indent=0) + "\n", encoding="utf-8")
    print(f"wrote {index} ({len(names)} members, read remotely)")


def fetch(name: str, data_dir: Path, force: bool = False, skip_images: bool = False) -> Path:
    out_dir = data_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if name in KAGGLE:
        fetch_kaggle(name, out_dir, force)
        return out_dir
    for src in SOURCES[name]:
        if skip_images and src.get("images"):
            print(f"skipping image archive {src['archive']} (--skip-images)")
            if name == "isic2018":
                save_remote_index(src, out_dir)
            continue
        fetch_archive(src, out_dir, force)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", required=True, choices=sorted([*SOURCES, *KAGGLE]))
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
