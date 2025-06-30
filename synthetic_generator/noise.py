from __future__ import annotations

"""Optional visual noise utilities (blur, rotation) for synthetic PDFs.

Currently it's a no-op unless cfg.noise.enabled.
"""

from pathlib import Path
from typing import List

from PIL import Image, ImageFilter


def apply_noise(pdf_path: Path, cfg) -> None:  # noqa: ANN001
    """Apply simple Gaussian blur to every page if enabled.

    If pdf2image is unavailable or noise disabled, does nothing.
    """
    if not getattr(cfg, "noise", None) or not cfg.noise.enabled:
        return

    try:
        from pdf2image import convert_from_path  # type: ignore
    except ImportError:
        print("[noise] pdf2image not installed – skipping noise step")
        return

    pages: List[Image.Image] = convert_from_path(str(pdf_path))
    blurred = [p.convert("RGB").filter(ImageFilter.GaussianBlur(radius=cfg.noise.blur_sigma)) for p in pages]
    tmp = pdf_path.with_suffix(".noisy.pdf")
    blurred[0].save(tmp, save_all=True, append_images=blurred[1:])
    pdf_path.unlink(missing_ok=True)
    tmp.rename(pdf_path) 