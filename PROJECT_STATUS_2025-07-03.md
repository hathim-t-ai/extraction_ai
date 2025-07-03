# Project Status – Bank-Statement Extraction

Date: July 3, 2025

---

## 1. Repository Overview
| Module / Dir | Purpose | Key Files |
|--------------|---------|-----------|
| `Donut/` | OCR-free VT model wrappers | `donut_extraction.py`, `donut_table_extraction.py`, `DONUT_DOCUMENTATION.md` |
| `MistralOCR/` | Cloud OCR client wrappers | `mistral_ocr_extraction.py`, `MISTRAL_OCR_DOCUMENTATION.md` |
| `Paddleocr/` | Baseline OCR + heuristic parser | `paddleocr_extraction.py`, `row_parser.py` |
| `LayoutLMv3/` | Token-level LM prototype | `layoutlm_finetune.py`, `LayoutLmv3_documentation.md` |
| `synthetic_generator/` | Fake PDF + GT creator | `cli.py`, templates, `SYNTHETIC_PDF_GENERATION*.md` |
| `data/` | Real PDFs + CSV ground truth | `For_Training/`, `annotations/`, `synthetic/` |
| `Outputs/` | Example inference outputs | e.g. `ENBD-statements-pdf_mistral.csv` |

### Stats
* **Real PDFs:** 2 133 (42 banks × 5 layout types ≈)
* **Ground-truth CSVs:** 1 987 (coverage 93 %)
* **Synthetic PDFs:** 7 203 across 11 batches (single-page only)
* **Unit tests:** 24 (transaction parser, synthetic ledger)

## 2. Achievements to Date
✅ Donut & Mistral wrappers with CLI parity  
✅ Synthetic generator with 3 real-world templates  
✅ End-to-end CSV export path (`donut_table_extraction.extract_table_from_pdf`)  
✅ Detailed engineering docs for Donut, Mistral, synthetic pipeline  
✅ Rye-managed env and Ruff lint passing CI  

## 3. Gaps / Work Remaining
| Area | Missing | Priority |
|------|---------|----------|
| **Model fine-tuning** | Donut & Mistral not trained on our dataset | P0 |
| **Dataset curation** | Multi-page GT JSON, edge-case scans | P0 |
| **Evaluation harness** | Automated precision/recall + visual diff | P1 |
| **Multi-page synthetic** | Generator split & header carry-over | P1 |
| **Templates backlog** | 30+ bank layouts still stubbed | P1 |
| **Ensemble logic** | Confidence score routing | P2 |
| **CI/CD** | GPU-less regression tests in GitHub Actions | P2 |
| **Security review** | PII handling in cloud OCR | P2 |

## 4. Immediate Next Steps (Next 2 Weeks)
1. **Convert CSV ground-truth to per-page JSON** (`tools/csv_to_jsonl.py`).
2. **Train Donut v0.1** on ~1 000 pages using LoRA; log MLflow metrics.
3. **Kick off Mistral fine-tune** with same JSON schema.
4. **Draft evaluation notebook** that computes field-level metrics and highlights mismatches.
5. **Expand synthetic generator** to output 2–5 page statements.

## 5. Risks & Mitigations
* **Label noisiness** – write `validate_csv.py` to assert numeric parsing before training.
* **API quota** – request Mistral quota increase early.
* **Timeline slip** – parallelise Donut & Mistral streams; synthetic generator work can be done on CPU.

## 6. Open Questions
1. Who owns MLflow server & bucket credentials?  
2. Preferred hosting for Donut checkpoint (HF private repo vs S3)?  
3. Is balance column mandatory for all export formats?

---
_© extraction-ai project – status compiled 2025-07-03_ 