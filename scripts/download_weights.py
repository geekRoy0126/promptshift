"""Download model weights into <storage root>/weights/ and verify them.

Usage:
    python scripts/download_weights.py --model sam_vit_b [--model medsam ...] [--root PATH]
    python scripts/download_weights.py --all

Checks:
- SAM: the official file names end with the first 6 hex digits of the file's md5.
- MedSAM: sha256 recorded on 2026-11-05 from the official Google Drive folder
  linked in the MedSAM README (docs/decisions.md).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from promptshift.paths import get_paths

WEIGHTS = {
    "sam_vit_b": {
        "file": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "md5_prefix": "01ec64",
    },
    "sam_vit_h": {
        "file": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "md5_prefix": "4b8939",
    },
    "medsam": {
        "file": "medsam_vit_b.pth",
        "gdrive_id": "1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_",
        "sha256": "34b34b78c1d18cb8c6bf84cf9c00e135d6d6c965699f3c0e31ef1bc9dcb5be74",
    },
}


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def verify(name: str, path: Path) -> bool:
    spec = WEIGHTS[name]
    if "md5_prefix" in spec:
        ok = digest(path, "md5").startswith(spec["md5_prefix"])
    else:
        ok = digest(path, "sha256") == spec["sha256"]
    print(f"{'OK  ' if ok else 'FAIL'} {name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return ok


def fetch(name: str, weights_dir: Path) -> bool:
    spec = WEIGHTS[name]
    path = weights_dir / spec["file"]
    if path.is_file():
        return verify(name, path)
    tmp = path.with_suffix(".part")
    if "url" in spec:
        print(f"downloading {spec['url']}")
        urllib.request.urlretrieve(spec["url"], tmp)
    else:
        import gdown

        gdown.download(id=spec["gdrive_id"], output=str(tmp), quiet=False)
    tmp.replace(path)
    return verify(name, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", action="append", choices=sorted(WEIGHTS), default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    names = sorted(WEIGHTS) if args.all else args.model
    if not names:
        parser.error("give --model NAME (repeatable) or --all")
    weights_dir = get_paths(args.root).ensure().weights
    results = {n: fetch(n, weights_dir) for n in names}
    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"verification FAILED for: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
