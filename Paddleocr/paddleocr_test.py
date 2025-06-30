#!/usr/bin/env python
"""Run PaddleOCR extraction on the two reference PDFs for a quick evaluation.

Usage
-----
$ rye run python scripts/paddleocr_test.py

The script extracts text from the following hard-coded PDFs:
    • YourBank-Bank Statement Example Final.pdf
    • ENBD-statements-pdf.pdf

For each file a CSV with OCR results will be written to the *Outputs/*
folder alongside existing artefacts, suffixed with *_paddleocr.csv*.
"""

from pathlib import Path

from pdf_parser.paddleocr_extraction import PaddleOCRExtractor
from pdf_parser.postprocess.row_parser import parse_transactions

ROOT = Path(__file__).resolve().parent.parent
PDF_FILES = [
    ROOT / "data" / "raw_pdfs" / "YourBank-Bank Statement Example Final.pdf",
    ROOT / "data" / "raw_pdfs" / "ENBD-statements-pdf.pdf",
]
OUTPUT_DIR = ROOT / "Outputs"


def main() -> None:
    extractor = PaddleOCRExtractor()

    for pdf_path in PDF_FILES:
        csv_path = OUTPUT_DIR / f"{pdf_path.stem}_paddleocr.csv"
        print(f"[PaddleOCR] Processing {pdf_path.name} -> {csv_path}")
        df = extractor.extract_from_pdf(pdf_path, csv_path)
        print(f"    {len(df)} OCR lines captured")

        # Post-process into transaction table
        parsed_csv = OUTPUT_DIR / f"{pdf_path.stem}_parsed.csv"
        tx_df = parse_transactions(csv_path)
        tx_df.to_csv(parsed_csv, index=False)
        print(f"    → structured rows: {len(tx_df)} written to {parsed_csv}\n")


if __name__ == "__main__":
    main() 