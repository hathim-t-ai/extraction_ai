from __future__ import annotations

"""Small helper utilities for the synthetic generator."""

import hashlib
from datetime import datetime
from pathlib import Path

__all__ = [
    "timestamped_dir",
    "sha256sum",
]


def timestamped_dir(base: str | Path, prefix: str = "batch") -> Path:
    """Create and return a timestamped output directory."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = base / f"{ts}_{prefix}"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def sha256sum(path: str | Path, chunk_size: int = 8192) -> str:
    """Return SHA-256 hex digest of *path*."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest() 