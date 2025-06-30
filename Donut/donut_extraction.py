from __future__ import annotations

"""donut_extraction.py

OCR-free extraction wrapper around the HuggingFace Donut model.

This module mirrors the public API of `Paddleocr/paddleocr_extraction.py` so that
call-sites can switch between extractors with minimal changes.

Example
-------
>>> from Donut.donut_extraction import DonutExtractor
>>> dx = DonutExtractor()
>>> df = dx.extract_from_pdf("data/raw_pdfs/ENBD-statements-pdf.pdf",
...                          "Outputs/ENBD-statements-pdf_donut.jsonl")
>>> df.head()

Notes
-----
Donut produces **structured JSON strings** rather than plain text; therefore the
output CSV/JSONL contains one record per *page* with the entire prediction for
that page. Downstream parsers are expected to post-process the JSON.

"""

from pathlib import Path
from typing import Any, Dict, List
import json
import logging

import fitz  # PyMuPDF type: ignore
from PIL import Image
import pandas as pd
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

__all__ = ["DonutExtractor"]

logger = logging.getLogger(__name__)
_DEFAULT_DPI: int = 150
_MODEL_NAME: str = "naver-clova-ix/donut-base"


def _pdf_pages_to_images(pdf_path: Path | str, dpi: int = _DEFAULT_DPI) -> List[Image.Image]:
    """Render each page of *pdf_path* as a RGB :class:`PIL.Image`."""
    doc = fitz.open(str(pdf_path))
    images: List[Image.Image] = []

    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images


class DonutExtractor:
    """Lightweight façade around the Donut model for page-level predictions."""

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self.device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.info("Loading Donut model '%s' on %s", model_name, self.device)
        self.processor: DonutProcessor = DonutProcessor.from_pretrained(model_name)
        self.model: VisionEncoderDecoderModel = (
            VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _predict(self, image: Image.Image) -> Dict[str, Any]:
        """Run Donut on *image* and return the decoded JSON structure."""
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        decoder_input_ids = self.processor.tokenizer("<s>", add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.config.max_length,
                early_stopping=True,
            )
        decoded: str = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        try:
            return self.processor.token2json(decoded)
        except Exception:
            # Fallback: wrap raw string
            return {"text": decoded}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_from_pdf(
        self,
        pdf_path: Path | str,
        output_jsonl: Path | str | None = None,
    ) -> pd.DataFrame:
        """Run Donut over each page and return predictions as a DataFrame.

        Parameters
        ----------
        pdf_path : Path | str
            Input PDF.
        output_jsonl : optional
            Path to save newline-delimited JSON predictions.
        """
        pdf_path = Path(pdf_path)
        images = _pdf_pages_to_images(pdf_path)

        records: List[dict[str, Any]] = []
        for page_num, img in enumerate(images, start=1):
            pred = self._predict(img)
            records.append({
                "pdf": pdf_path.name,
                "page": page_num,
                "prediction": json.dumps(pred, ensure_ascii=False),
            })

        df = pd.DataFrame.from_records(records, columns=["pdf", "page", "prediction"])

        if output_jsonl:
            output_path = Path(output_jsonl)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return df


# -----------------------------------------------------------------------------
# CLI helper (mirrors PaddleOCR design)
# -----------------------------------------------------------------------------

def cli(argv: List[str] | None = None) -> None:  # pragma: no cover
    import argparse, textwrap, sys

    parser = argparse.ArgumentParser(
        prog="donut_extract",
        description=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument("pdf", type=Path, help="Path to input PDF")
    parser.add_argument("output", type=Path, nargs="?", help="Optional JSONL output path")
    args = parser.parse_args(args=argv)

    extractor = DonutExtractor()
    df = extractor.extract_from_pdf(args.pdf, args.output)
    print(df.head())


if __name__ == "__main__":  # pragma: no cover
    cli() 