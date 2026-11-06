"""Read members of a large remote zip with HTTP range requests (no full download).

Used for ISIC 2018: the training image zip is 11 GB, but checking the image count needs only
the zip's central directory, and a 5-sample figure needs only 5 members.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from collections.abc import Callable

Fetch = Callable[[int, int], bytes]  # (start, end inclusive) -> bytes


def http_fetch(url: str, timeout: int = 60) -> tuple[int, Fetch]:
    head = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(head, timeout=timeout) as r:
        size = int(r.headers["Content-Length"])
        if r.headers.get("Accept-Ranges", "bytes") != "bytes":
            raise RuntimeError(f"{url} does not support range requests")

    def fetch(start: int, end: int) -> bytes:
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if len(data) != end - start + 1:
            raise RuntimeError(f"range {start}-{end}: got {len(data)} bytes")
        return data

    return size, fetch


class RangeFile(io.RawIOBase):
    """Seekable read-only file over a fetch(start, end) function."""

    def __init__(self, size: int, fetch: Fetch) -> None:
        self.size, self.fetch, self.pos, self.requests = size, fetch, 0, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        data = self.fetch(self.pos, end)
        self.requests += 1
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote_zip(url: str, buffer: int = 1 << 20) -> zipfile.ZipFile:
    size, fetch = http_fetch(url)
    return zipfile.ZipFile(io.BufferedReader(RangeFile(size, fetch), buffer))
