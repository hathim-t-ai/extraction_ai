#!/usr/bin/env python
"""
Demo: LayoutLMv3 token-level extraction on sample bank statements.

This script runs a **pre-trained** LayoutLMv3 model (fine-tuned on FUNSD)
on two example PDFs shipped in the repository:
  1. YourBank-Bank Statement Example Final.pdf
  2. ENBD-statements-pdf.pdf

For each page in the PDF we:
  • Extract words and their bounding boxes using **pdfplumber**
  • Normalise bounding boxes to the 0-1000 coordinate space expected by LayoutLM
  • Feed the image, words and boxes into the LayoutLMv3 processor
  • Run token-classification to predict an **entity label** per word
  • Save results to `demo_outputs/<pdf_name>_layoutlm_tokens.csv`

The output CSV columns are:
  pdf,page,word,label,x0,y0,x1,y1

Note: The chosen model (`microsoft/layoutlmv3-finetuned-funsd`) is trained on
form-like documents (FUNSD dataset) – it is *not* specialised for bank
statements, but serves as a quick demonstration of model I/O and inference
flow. For production-grade extraction you should fine-tune on
bank-statement-specific annotations.

Usage (from repository root):
  rye run python scripts/demo_layoutlm_extraction.py

Requirements (added via Rye):
  rye add transformers pdfplumber pillow torch torchvision --dev
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Tuple

import pdfplumber
import torch
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoProcessor

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

MODEL_NAME: str = "microsoft/layoutlmv3-base"
PDF_FILES: List[Path] = [
  Path("data/raw_pdfs/ENBD-statements-pdf.pdf"),
]
OUTPUT_DIR: Path = Path("demo_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

dev: torch.device = (
  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
logger.info("Using device: %s", dev)

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def extract_words_boxes(page: pdfplumber.page.Page) -> Tuple[List[str], List[List[int]]]:
  """Return words and LayoutLM-scaled bounding boxes for a pdfplumber page."""
  words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
  if not words:
    return [], []

  # Render page image to compute absolute pixel dimensions
  image: Image.Image = page.to_image(resolution=150).original.convert("RGB")
  width, height = image.size

  word_texts: List[str] = []
  bboxes: List[List[int]] = []
  for w in words:
    word_texts.append(w["text"])
    x0, top, x1, bottom = w["x0"], w["top"], w["x1"], w["bottom"]
    # Normalise to 0-1000 as expected by LayoutLM family
    bboxes.append(
      [
        int(1000 * x0 / width),
        int(1000 * top / height),
        int(1000 * x1 / width),
        int(1000 * bottom / height),
      ]
    )

  return word_texts, bboxes


def run_layoutlm_on_pdf(pdf_path: Path, model, processor) -> Path:
  """Run LayoutLMv3 over *all* pages of a PDF and write CSV output."""
  output_csv = OUTPUT_DIR / f"{pdf_path.stem}_layoutlm_tokens.csv"
  if output_csv.exists():
    output_csv.unlink()  # overwrite

  writer = None
  try:
    pdf_obj = pdfplumber.open(str(pdf_path))
    use_plumber = True
  except Exception as e:
    logger.warning("pdfplumber failed on %s (%s). Falling back to PyMuPDF.", pdf_path.name, e)
    use_plumber = False

  with open(output_csv, "w", newline="") as csv_f:
    writer = csv.writer(csv_f)
    writer.writerow(["pdf", "page", "word", "label", "x0", "y0", "x1", "y1"])

    if use_plumber:
      with pdf_obj as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
          words, boxes = extract_words_boxes(page)
          if not words:
            logger.warning("No words detected on %s page %d – skipping", pdf_path.name, page_idx)
            continue

          image: Image.Image = page.to_image(resolution=150).original.convert("RGB")
          labels = _predict_labels(words, boxes, image, model, processor)
          for word, label, box in zip(words, labels, boxes):
            writer.writerow([pdf_path.name, page_idx, word, label, *box])
    else:
      try:
        for page_idx, words, boxes, image in extract_words_boxes_pymupdf(pdf_path):
          if not words or image is None:
            logger.warning("No words on %s page %d", pdf_path.name, page_idx)
            continue
          labels = _predict_labels(words, boxes, image, model, processor)
          for word, label, box in zip(words, labels, boxes):
            writer.writerow([pdf_path.name, page_idx, word, label, *box])
      except Exception as pym_err:
        logger.error("PyMuPDF failed on %s: %s", pdf_path.name, pym_err)
        logger.warning("Skipping %s after PyMuPDF failure", pdf_path.name)
        return output_csv  # empty or partially filled

  logger.info("Saved token predictions to %s", output_csv.relative_to(Path.cwd()))
  return output_csv


def _predict_labels(words: List[str], boxes: List[List[int]], image: Image.Image, model, processor) -> List[str]:
  """Helper to encode inputs and run model to obtain predicted label strings."""
  encoding = processor(
    image,
    words,
    boxes=boxes,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
  )
  encoding = {k: v.to(dev) for k, v in encoding.items()}
  with torch.no_grad():
    outputs = model(**encoding)
  logits = outputs.logits
  predictions = logits.argmax(-1).squeeze().tolist()
  return [model.config.id2label[p] for p in predictions][: len(words)]


def extract_words_boxes_pymupdf(doc_path: Path) -> List[Tuple[int, List[str], List[List[int]], Image.Image]]:
    """Yield page index, words, boxes, image for a PDF using PyMuPDF."""
    import fitz  # PyMuPDF

    pm_doc = fitz.open(str(doc_path))
    for page_idx in range(len(pm_doc)):
        page = pm_doc[page_idx]
        # words: list of (x0, y0, x1, y1, word, block_no, line_no, word_no)
        words_data = page.get_text("words")
        if not words_data:
            yield page_idx + 1, [], [], None
            continue
        # page pixmap for image
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # medium dpi
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        width, height = image.size

        words: List[str] = []
        bboxes: List[List[int]] = []
        for w in words_data:
            x0, y0, x1, y1, txt, *_ = w
            words.append(txt)
            bboxes.append([
                int(1000 * x0 / width),
                int(1000 * y0 / height),
                int(1000 * x1 / width),
                int(1000 * y1 / height),
            ])
        yield page_idx + 1, words, bboxes, image

# ----------------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------------

def main() -> None:
  logger.info("Loading model: %s", MODEL_NAME)
  processor = AutoProcessor.from_pretrained(MODEL_NAME)
  # Disable internal OCR to allow custom word+box inputs
  if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "apply_ocr"):
    processor.image_processor.apply_ocr = False
  model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME).to(dev)
  model.eval()

  for pdf in PDF_FILES:
    if not pdf.exists():
      logger.error("PDF not found: %s", pdf)
      continue
    try:
      run_layoutlm_on_pdf(pdf, model, processor)
    except Exception as e:
      logger.error("Failed to process %s: %s", pdf, e)

  logger.info("Extraction complete. CSV files located in %s", OUTPUT_DIR)


if __name__ == "__main__":
  main() 