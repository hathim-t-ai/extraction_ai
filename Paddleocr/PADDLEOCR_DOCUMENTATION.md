# PaddleOCR Extraction Pipeline – Technical Documentation

**Last updated:** 26 June 2025

---

## 1  Overview

This document captures everything related to the PaddleOCR-based extraction pipeline that converts bank-statement PDFs into a structured transactions table (`date, description, currency, amount, balance`).  It consolidates design decisions, recent evaluation findings, outstanding fixes, and the current code layout so future contributors can ramp-up quickly.

> **Key modules**
> • `src/pdf_parser/paddleocr_extraction.py` – page rasterisation ➜ `PaddleOCR` inference  
> • `src/pdf_parser/postprocess/row_parser.py` – rule-engine that converts raw OCR lines into transactions  
> • `scripts/evaluate_paddleocr.py` – CLI to benchmark parsed CSVs against ground-truth annotations  
> • `rules/description_aliases.yml` – optional alias table for description normalisation

---

## 2  Current State (code rev 26-Jun-2025)

| Stage | Implementation highlights |
|-------|---------------------------|
| **PDF ➜ images** | PyMuPDF rasterises every page at **300 DPI** (fallback 400 DPI planned) regardless of embedded bitmaps to avoid skipped pages. |
| **OCR** | Cached `PaddleOCR(use_angle_cls=True, lang="en")` instance loads local weights under `$HOME/.paddleocr/whl/**`; no network calls at runtime. |
| **Post-processing** | Deterministic rule-engine (y-clustering + state-machine) merges wrapped lines, extracts `(amount, balance)` pair, applies alias table, validates running balance, removes duplicates. |
| **Evaluation** | `scripts/evaluate_paddleocr.py` measures recall/precision by `(date, amount)` match (+ fuzzy description planned). |

Latest run (after regex & year-inference patch):

| PDF | GT rows | Parsed rows | Recall | Precision |
|-----|---------|-------------|--------|-----------|
| ENBD-statements-pdf | 83 | 83 | **97 %** | **97 %** |
| YourBank statement  | 14 | 10 | 14 % | 20 % |

---

## 3  Observations & Immediate Fixes

1. **Low-contrast image-wrapped PDFs (YourBank)**  
   • PyMuPDF sometimes returns the embedded bitmap directly and skips full rasterisation, leading to incomplete OCR coverage.  
   • _Fix_  → always call `page.get_pixmap(matrix)` to force raster, then auto-bump DPI to **400** if page-level mean OCR confidence < 0.60.

2. **Trailing currency token**  
   • Some layouts place a right-aligned three-letter currency ("AED", "USD") whose bbox sits between amount & balance.  
   • _Fix_  → in `row_parser`, if the last token of a cluster is `^[A-Z]{3}$`, treat it as `currency` and exclude it from `description`.

3. **Wrapped descriptions > 2 lines**  
   • YourBank rows occasionally span **three** y-clusters; extend buffer length in the state-machine to absorb all clusters until the numeric pair is secured.

These fixes are tracked in tasks 9.x and will be implemented next.

---

## 4  Development Decisions – Task 9.1 (verbatim)

> _Date: 26 June 2025_
>
> **Context**  
> We integrated PaddleOCR as a lightweight, offline extractor … *(full content pasted verbatim)*

```markdown
# DEVELOPMENT DECISIONS – Task 9.1

Date: 26 June 2025

## Context
We integrated PaddleOCR as a lightweight, offline extractor for bank‐statement PDFs and built a Python post-processing layer (`pdf_parser.postprocess.row_parser`) to convert raw OCR lines into a structured *transactions* table matching the project’s annotation schema (`date, description, currency, amount, balance`).

During iterative tuning we discovered two tricky layout patterns:

1. **Wrapped descriptions** – A transaction’s textual part appears on one line while its numeric part appears on the next.
2. **Multi-transaction clusters** – One physical line contains *two* independent `(amount, balance)` pairs (common in ENBD statements).

## Adopted Strategy

### 1  Deterministic rule-engine (current implementation)

* **y-clustering** groups neighbouring OCR lines that share the same baseline.
* **State-machine merge** collects clusters that belong to the *same* logical transaction:
  1. When a cluster contains a *date* token, a *transaction buffer* is opened.
  2. Subsequent clusters are appended to the buffer’s description until a cluster with ≥ 1 numeric token appears.
  3. The right-most numeric tokens are interpreted as `(amount, balance)` using the "hybrid" heuristic (amount is the closest left token whose magnitude is plausibly ≤ balance; fallback to previous token).
  4. The completed buffer is emitted and the machine resets.
* **Running-balance validator** flips the sign of `amount` when it repairs the arithmetic `prev_balance + amount ≈ balance` (|ε|≤0.1).
* **Duplicate filter** drops rows that share `(date, amount, balance)`.

Results (after hybrid tweak):
```
YourBank-Bank Statement Example Final  … 11 / 16 rows
ENBD-statements-pdf                    … 85 / 85 rows
```
Missing YourBank rows are those where description wraps over **three** lines; they will be fixed by the state-machine once we allow the buffer to span more than two clusters.

### 2  Alias table (incremental learning without ML)

A YAML/JSON file (`rules/description_aliases.yml`) stores canonical replacements:
```yaml
monthly apartment rent:  Monthly Apartment Rent
randomfords deli:        Randomford's Deli
```
When you manually correct a description you append an alias; the parser normalises future OCR text with this table. This is:
* offline & deterministic
* diff-friendly and covered by CI
* bank-specific patterns live next to the code, not hard-wired inside it.

### 3  Regression tests

A pytest module will:
* run the pipeline for every `raw_pdfs/*.pdf` that has a matching `annotations/*_ground_truth.csv`.
* `assert_frame_equal` (with numeric tolerance) to guarantee 100 % recall/precision.
* executed locally (`pytest -q`) and in CI.

### 4  Future model-based learning (optional)

If the rule-engine becomes brittle across many banks:
1. Log parser errors/corrections into a training CSV `ml_training/transactions.tsv`.
2. Fine-tune a small token-classifier (e.g. LayoutLM-mini, Bi-LSTM-CRF) that tags each OCR token as DATE / DESC / AMT / BAL.
3. Replace regex extraction with model inference, **but keep** the rule-based validator to catch model slips.

This path requires GPU access & model-release management, so we defer it until deterministic rules fail to converge.

## Recommendation
* **Short-term** – finish the state-machine merger and alias table, then lock behaviour with pytest.
* **Medium-term** – populate aliases as new statements appear; adjust thresholds via config.
* **Long-term** – re-evaluate after ≥ 3 new banks show failure cases that can't be captured by rules + aliases; consider ML fine-tune.

---
*(appended without altering previous documentation)*
```

---

## 5  Roadmap Snapshot

| Horizon | Item | Owner | ETA |
|---------|------|-------|-----|
| Short-term | Forced rasterisation & adaptive DPI | OCR team | 30-Jun-25 |
| Short-term | Currency-token exclusion in `row_parser` | Parsing team | 30-Jun-25 |
| Medium | Pytest regression on all annotated PDFs | QA | 05-Jul-25 |
| Medium | Populate `description_aliases.yml` for YourBank edge cases | Data team | Rolling |
| Long | Evaluate ML token-classifier fallback | R&D | Q4-25 |

---

## 11  Source Files (PaddleOCR Stack)

| Path | Purpose |
|------|---------|
| `src/pdf_parser/paddleocr_extraction.py` | Core extractor – PDF pages ➜ PaddleOCR ➜ raw CSV (lines + bboxes + confidence) |
| `src/pdf_parser/postprocess/row_parser.py` | Rule-engine – raw OCR lines ➜ structured transactions table |
| `scripts/paddleocr_test.py` | Convenience script to run extractor on every PDF in `raw_pdfs/` and drop results into `Outputs/` |
| `scripts/evaluate_paddleocr.py` | Benchmarks parsed CSVs against `data/annotations/*_ground_truth.csv` (precision / recall) |
| `tests/test_row_parser_against_groundtruth.py` | Unit / regression test ensuring 100 % match on annotated PDFs (to be extended into full batch harness) |
| `rules/description_aliases.yml` | Alias table for normalising free-text descriptions (incremental learning) |
| `requirements.lock` (rows for `paddleocr`, `paddlex`) | Captures the exact PaddleOCR dependency tree |

All new or modified code related to PaddleOCR lives in the paths above; pull requests referencing **Task 9.x** touch only these files to keep the solution self-contained.

---

*End of Document* 