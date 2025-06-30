from __future__ import annotations

"""mistral_ocr_extraction.py

Wrapper around the **Mistral AI OCR** endpoint.  Mirrors the public API of
`Paddleocr/paddleocr_extraction.py` and `Donut/donut_extraction.py` so that
call-sites can switch between extractors with minimal changes.

The extractor operates **page-by-page**:

1. Rasterise each PDF page to an RGB `PIL.Image` (via PyMuPDF).
2. Convert the image to **base64** and send it to the `/v1/ocr` endpoint via the
   official `mistralai` Python client (>=0.4.0).
3. Parse the JSON response – each page comes back as an `OCRPageObject` which
   contains a Markdown text representation (`markdown` field).

The resulting :class:`pandas.DataFrame` contains one row per page with the
following columns:

    – pdf       : PDF filename
    – page      : 1-based page index
    – markdown  : Raw Markdown text returned by the OCR engine
    – ocr_json  : Full JSON response for later debugging (string-encoded)

Example
-------
>>> from MistralOCR.mistral_ocr_extraction import MistralOCRExtractor
>>> mx = MistralOCRExtractor("Focus")
>>> df = mx.extract_from_pdf("data/raw_pdfs/YourBank-Bank Statement Example Final.pdf")
>>> df.head()

Notes
-----
* Mistral currently offers multiple OCR-capable models; **"mistral-ocr-latest"** is used as
  the default as of June 2025.  Override via the `model_name` parameter.
* Authentication relies on the ``MISTRAL_API_KEY`` environment variable (already
  populated from the project ``.env`` file).
* For documents already hosted online you can skip the rasterisation step and
  pass a `DocumentURLChunk` directly – not covered by this helper.
"""

from pathlib import Path
from typing import Any, Dict, List
import base64
import json
import logging
import os

import fitz  # PyMuPDF type: ignore
from PIL import Image
import pandas as pd
from mistralai import Mistral

__all__ = [
    "MistralOCRExtractor",
]

logger = logging.getLogger(__name__)
_DEFAULT_DPI: int = 200  # 300 DPI is slower; 200 is enough for Mistral focus
_DEFAULT_MODEL: str = "mistral-ocr-latest"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _pdf_pages_to_images(pdf_path: Path | str, dpi: int = _DEFAULT_DPI) -> List[Image.Image]:
    """Render *pdf_path* at *dpi* and return a list of RGB images."""
    doc = fitz.open(str(pdf_path))
    images: List[Image.Image] = []

    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images


def _pil_to_base64(img: Image.Image) -> str:
    """Encode a PIL Image as *base64* PNG."""
    import io

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# -----------------------------------------------------------------------------
# Main extractor
# -----------------------------------------------------------------------------

class MistralOCRExtractor:
    """Simple façade around the Mistral AI OCR endpoint."""

    def __init__(self, model_name: str = _DEFAULT_MODEL, dpi: int = _DEFAULT_DPI) -> None:
        """Create a new extractor instance.

        Parameters
        ----------
        model_name : str, default "Focus"
            OCR model to use.  Check the Mistral docs for the latest options.
        dpi : int, default 200
            Rasterisation resolution.  Higher DPI improves accuracy at the cost
            of latency & bandwidth (base64 payload size).
        """
        self.model_name = model_name
        self.dpi = dpi
        # The client lazily opens a persistent HTTP session; safe to reuse.
        api_key = os.getenv("MISTRAL_API_KEY", "")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY environment variable not set")
        self._client = Mistral(api_key=api_key)
        logger.info("Initialised Mistral client with model '%s'", model_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _predict_markdown(self, img: Image.Image) -> Dict[str, Any]:
        """Send *img* to Mistral OCR and return the raw JSON response."""
        b64_png = _pil_to_base64(img)
        document = {
            "type": "image_url",
            "image_url": f"data:image/png;base64,{b64_png}",
        }
        # Optional: for small docs we can omit pages, other params use defaults
        try:
            response = self._client.ocr.process(
                model=self.model_name,
                document=document,
                include_image_base64=False,
            )
            # The SDK returns a Pydantic model; convert to dict for portability
            if hasattr(response, "model_dump"):
                return response.model_dump()
            return json.loads(json.dumps(response))  # fallback generic serialise
        except Exception as exc:
            logger.error("Mistral OCR request failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_from_pdf(
        self,
        pdf_path: Path | str,
        output_jsonl: Path | str | None = None,
    ) -> pd.DataFrame:
        """Run OCR on *pdf_path* and return a tidy DataFrame.

        Parameters
        ----------
        pdf_path : Path | str
            Input PDF file.
        output_jsonl : Path | str | None, optional
            Persist full per-page JSON responses to this path (newline-delimited).
        """
        pdf_path = Path(pdf_path)
        images = _pdf_pages_to_images(pdf_path, dpi=self.dpi)

        records: List[dict[str, Any]] = []
        for page_num, img in enumerate(images, start=1):
            ocr_json = self._predict_markdown(img)
            markdown = ""
            # Extract Markdown if present
            try:
                pages = ocr_json.get("pages") or ocr_json.get("data") or []
                if pages and isinstance(pages, list):
                    markdown = pages[0].get("markdown", "") if isinstance(pages[0], dict) else ""
                else:
                    markdown = ocr_json.get("markdown", "")
            except Exception:
                pass  # leave as empty string

            records.append({
                "pdf": pdf_path.name,
                "page": page_num,
                "markdown": markdown,
                "ocr_json": json.dumps(ocr_json, ensure_ascii=False),
            })

        df = pd.DataFrame.from_records(records, columns=["pdf", "page", "markdown", "ocr_json"])

        if output_jsonl is not None:
            out_path = Path(output_jsonl)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return df


# -----------------------------------------------------------------------------
# CLI helper (mirrors Donut/PaddleOCR style)
# -----------------------------------------------------------------------------

def cli(argv: list[str] | None = None) -> None:  # pragma: no cover
    import argparse, textwrap

    parser = argparse.ArgumentParser(
        prog="mistral_ocr_extract",
        description=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument("pdf", type=Path, help="Path to input PDF")
    parser.add_argument("output", type=Path, nargs="?", help="Optional JSONL output path")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Mistral OCR model name (default: Focus)")
    parser.add_argument("--dpi", type=int, default=_DEFAULT_DPI, help="Rasterisation DPI (default: 200)")
    args = parser.parse_args(args=argv)

    extractor = MistralOCRExtractor(model_name=args.model, dpi=args.dpi)
    df = extractor.extract_from_pdf(args.pdf, args.output)
    print(df.head())


if __name__ == "__main__":  # pragma: no cover
    cli() 