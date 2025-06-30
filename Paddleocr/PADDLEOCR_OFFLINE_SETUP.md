# Offline Setup for PaddleOCR Integration

Date: June 25, 2025  <!-- ALWAYS use the real current date -->

This document captures the full sequence of problems encountered while wiring **PaddleOCR** into the PDF-Parser project and the precise, _offline-friendly_ fixes that make the tool run without any outbound network calls.

---

## 1  Dependency Conflicts

| Problem | Explanation | Resolution |
|---------|-------------|------------|
| **PyMuPDF version mismatch** | PaddleOCR 2.7.x requires `pymupdf < 1.21`, but the project originally used ≥ 1.26. | Pin **PyMuPDF 1.20.\*** in `pyproject.toml` and run `rye sync` (or `rye add pymupdf@"<1.21"`). |
| **Oversized PaddleX chain** | Newer PaddleOCR 3.x pulls the full **PaddleX** stack (> 600 MB). | Stay on **PaddleOCR 2.7.0** and **paddlepaddle 2.5.2** (CPU). |

---

## 2  Blocking Network Requests

### 2.1 Font Download
* PaddleOCR downloads *Noto-Sans* at runtime.  
* Fix: set `LOCAL_FONT_FILE_PATH` to any present `.ttf` file (or `/dev/null`) before importing PaddleOCR.

### 2.2 Model Weights & Hub Calls
* Default behaviour is to fetch three tar archives and ping a PaddleHub endpoint.  
* Fixes:
  1. **Monkey-patch** `paddleocr.ppocr.utils.network.download` to no-op.
  2. **Stub** the two `paddlex` imports (`sys.modules["paddlex"] = types.ModuleType("paddlex")`).

> These shims are implemented at the top of `src/pdf_parser/paddleocr_extraction.py`.

---

## 3  Manual Model Installation

PaddleOCR looks for the following directory layout under `~/.paddleocr/whl`:

```
~/.paddleocr/whl/det/en_PP-OCRv3_det_infer/en_ppocr_v3_det_infer.{pdmodel|pdiparams}
~/.paddleocr/whl/rec/en_PP-OCRv3_rec_infer/en_ppocr_v3_rec_infer.{pdmodel|pdiparams}
~/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.{pdmodel|pdiparams}
```

1. **Download** the three tarballs _offline_ (mirrors are in project notes):
   * `en_PP-OCRv3_det_infer.tar`
   * `en_PP-OCRv3_rec_infer.tar`
   * `ch_ppocr_mobile_v2.0_cls_infer.tar`
2. Place each file in its expected folder, e.g.:
   ```bash
   mkdir -p ~/.paddleocr/whl/det/en_PP-OCRv3_det_infer
   cp en_PP-OCRv3_det_infer.tar ~/.paddleocr/whl/det/en_PP-OCRv3_det_infer/
   # …repeat for rec & cls…
   ```
3. **Untar _in place_** so that **both** the tarball *and* the extracted directory exist — PaddleOCR's path resolver appends the tar-file name when checking paths.

```bash
for f in ~/.paddleocr/whl/*/*/*_infer.tar; do 
  tar -xf "$f" -C "$(dirname "$f")"
done
```

> This adds ~60 MB but guarantees zero further download attempts.

---

## 4  Code-Side Adjustments

* In `paddleocr_extraction.py` we initialise `PaddleOCR` with explicit local paths:

```python
ocr = PaddleOCR(
    det_model_dir=pathlib.Path.home() / ".paddleocr/whl/det/en_PP-OCRv3_det_infer",
    rec_model_dir=pathlib.Path.home() / ".paddleocr/whl/rec/en_PP-OCRv3_rec_infer",
    cls_model_dir=pathlib.Path.home() / ".paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer",
    show_log=False,
)
```
* Added lightweight rasterisation via **PyMuPDF** so PDFs are processed page-by-page in memory (no temporary images on disk).
* Wrapper returns a `pandas.DataFrame` and also writes a CSV to `Outputs/` for traceability.

---

## 5  Verification

Run the smoke test script:

```bash
rye run python scripts/paddleocr_test.py
```

Expected offline output (no HTTP on `lsof -i`):

```
✅ YourBank-Bank Statement Example Final_paddleocr.csv  (   80 rows)
✅ ENBD-statements-pdf_paddleocr.csv                   (  427 rows)
```

These artefacts live under `Outputs/` and will feed the upcoming post-processing pipeline (mapping to the annotation schema).

---

## 6  Next Steps

1. **Design comparison logic** against `data/annotations/*_ground_truth.csv`.
2. **Transform algorithm** in `src/pdf_parser/postprocess/row_parser.py` (or new module) to produce the structured annotation schema.
3. Add parameterised pytest cases using sample PDFs to ensure > 90 % field-level accuracy before rolling out to new statements. 