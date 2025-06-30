# MistralOCR package

"""Lightweight package exposing OCR extraction wrapper for Mistral AI.

Usage
-----
>>> from MistralOCR.mistral_ocr_extraction import MistralOCRExtractor
>>> extractor = MistralOCRExtractor()
>>> df = extractor.extract_from_pdf("data/raw_pdfs/ENBD-statements-pdf.pdf")
print(df.head())
""" 