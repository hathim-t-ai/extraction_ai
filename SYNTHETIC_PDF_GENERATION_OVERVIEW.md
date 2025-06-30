# Synthetic PDF Generator – Quick Overview

Date: June 30 2025

This note is a **reader-friendly cheat-sheet** for anyone who needs to pick up work on the synthetic bank-statement generator without digging through the full spec.  Use it to refresh your memory after a break or to brief a new teammate.

---

## 1.  What We Have Built So Far

1. **Command-line tool (`python -m synthetic_generator`)**  
   • Generates a batch of PDFs, plus perfectly aligned JSON & CSV ground-truth files.  
   • Rich progress bar and manifest with SHA-256 checksums.
2. **Data model**  
   • `Statement` (header info + list of `Transaction`s).  
   • Guarantees schema validity via Pydantic.
3. **Ledger faker**  
   • Random but realistic list of debit & credit rows with running balance.
4. **Dual rendering paths**  
   • **HTML → WeasyPrint** (easy to style, supports real CSS).  
   • **ReportLab** fallback (pure-Python, no native libs).
5. **Layout variety**  
   • Bank-specific templates: `Axis Bank`, `BISB` (bilingual banner), `Emirates NBD`.  
   • Generic template for all other banks.
6. **Variant engine (Phase 2)**  
   • `--schema` flag: 5-column (with Balance) or 4-column (no Balance).  
   • Random font family & header colour per document.
7. **Round-robin layout picker** – pass a list of layouts and the generator cycles through them.

### One-liner to try it out
```bash
rye run python -m synthetic_generator \
  --statements 6 \
  --layouts enbd_type_i,bisb_type_i,axisbank_type_i \
  --renderer html \
  --schema random
```
The batch lands in `data/synthetic/YYYYMMDD_HHMMSS_batch/`.

---

## 2.  Where We Stand Today (30 Jun 2025)

✅  Works end-to-end for single-page statements.  
✅  3 distinct bank headers + CSS variations.  
✅  Ground-truth files auto-generate.  
⚠️  No multi-page logic yet.  
⚠️  Noise / "scanned" look still a stub.  
⚠️  Only HTML renderer supports the new 4-column schema.  
⚠️  Need many more templates (we have >40 real layouts to mimic).

---

## 3.  Next Up on the Roadmap

1. **Multi-page support**  
   • Split after N rows, repeat header, add "Carried forward" rows.
2. **Noise pipeline**  
   • Blur, JPEG compression, slight rotation, speckle dots, optional watermark.
3. **Template backlog**  
   • Finish ENBD Type II–V, SBI Type III, HDFC Type I, etc.  
   • Re-use shared partials to keep code DRY.
4. **ReportLab parity**  
   • Port 4-column & future schemas to pure-python renderer.
5. **CI & tests**  
   • Pytest for new features; upload sample PDFs in GitHub Actions for visual diff.

---

## 4.  Useful CLI Flags (TL;DR)
```text
--statements 50                # how many PDFs to create
--layouts enbd_type_i,default  # choose templates
--schema five_column|four_no_balance|random
--renderer html|reportlab|auto # auto picks html if template exists
--seed 123                     # deterministic output
```

---

Happy generating!  For deep dives see `SYNTHETIC_PDF_GENERATION.md`. 