# Synthetic Bank-Statement PDF Generation

Date: June 28 2025

---

## Contents
1. Purpose & Scope
2. Inputs Needed from Stakeholders
3. High-Level Workflow Diagram
4. Directory & Module Layout
5. File-by-File Responsibilities
6. Configuration & Dependency Setup
7. Detailed Generation Pipeline
8. Testing Strategy
9. Continuous Integration Hook
10. Future Extensions
11. Layout Ingestion & Template System
12. Implementation Progress Log (June 30 2025)

---

## 1  Purpose & Scope
This document is the definitive specification for the **synthetic_generator** package. It explains how we will automatically create large batches of realistic, multi-layout bank-statement PDFs *plus* perfectly aligned ground-truth files (JSON & CSV). The generator will support Donut, LayoutLMv3, Mistral OCR fine-tuning, and rule-based evaluation.

Goals:
* Zero manual work once inputs are supplied.
* Deterministic output given the same seed (regression-friendly).
* Easy layout swapping (HTML or ReportLab paths).
* Strict schema validation so ground truth never drifts.

---

## 2  Inputs Needed from Stakeholders
| # | Item | Description |
|---|------|-------------|
| 1 | Logos / Branding | PNG or SVG assets for each bank layout. |
| 2 | Representative PDFs | 3–5 sample statements per layout for visual benchmarking. |
| 3 | Legal/Footer Text | Mandatory disclaimers or fixed copy. |
| 4 | Locale Settings | Currency (AED, USD …), date-format (`dd mmm yy`, `MM/DD/YYYY`), decimal separator. |
| 5 | Variation Knobs | Desired ranges: transactions/page, pages/statement, overdraft %, multiline description toggle. |
| 6 | Acceptance Checklist | Clear "looks-realistic" criteria or screenshots. |

---

## 3  High-Level Workflow Diagram
```mermaid
flowchart TD
    A[Load YAML/ENV Config] --> B[Generate Synthetic Ledger]
    B --> C[Render to PDF (ReportLab | HTML→WeasyPrint)]
    C --> D[Apply Noise (optional)]
    B --> E[Dump Ground-Truth JSON & CSV]
    C --> F[Validation Suite]
    E --> F
    F --> G[Batch Manifest + Checksums]
```

---

## 4  Directory & Module Layout
```text
synthetic_generator/
  __init__.py
  cli.py
  config.py
  data_models.py
  faker_providers.py
  ledger.py
  render/
    __init__.py
    reportlab_renderer.py
    html_renderer.py
    templates/
      enbd_template.html
      yourbank_template.html
      shared/...
  noise.py
  utils.py
  tests/
    test_schema.py
    test_ledger.py
    test_renderers.py
```
> Folder `data/synthetic/` (outside the package) will store generated batches: `YYYYMMDD_HHMMSS_batch-01/…`.

---

## 5  File-by-File Responsibilities
* **`__init__.py`** – Package metadata; exposes `generate_statement()` public API.
* **`cli.py`** – Typer CLI: `rye run make-synthetic …`. Parses flags, kicks off generation, displays Rich progress.
* **`config.py`** – Pydantic dataclass loading YAML/ENV: seed, #statements, layout list, locale, noise toggles.
* **`data_models.py`** – Canonical schema (`Transaction`, `Statement`). Includes converters `to_csv()`, `to_jsonl()`.
* **`faker_providers.py`** – Custom Faker providers for bank-centric data (IBAN, SWIFT, POS descriptions).
* **`ledger.py`** – Creates balanced transaction lists, injects edge-cases, outputs a `Statement` object.
* **`render/reportlab_renderer.py`** – Programmatic PDF drawing for pixel-level control.
* **`render/html_renderer.py`** – Jinja2 → HTML → WeasyPrint path; fast layout iteration.
* **`render/templates/`** – Layout HTML fragments & shared partials.
* **`noise.py`** – Post-process PDFs: blur, rotation, watermark, JPEG compression.
* **`utils.py`** – Helpers: timestamped dirs, checksum, safe writes.
* **`tests/…`** – Pytest suite for schema, ledger logic, renderer smoke tests.

---

## 6  Configuration & Dependency Setup
```bash
# One-time environment bootstrapping
rye init --template 3.11  # if not already initialised
rye add reportlab faker jinja2 weasyprint pydantic pillow pypdf2
rye add --group dev pytest typer rich ruff
```
`config/synthetic.yml` sample:
```yaml
seed: 42
statements: 500
layouts: ["enbd", "yourbank"]
locale:
  currency: "AED"
  date_fmt: "dd MMM yy"
noise:
  enabled: true
  blur_sigma: 1.2
output_dir: "data/synthetic/batch-01"
```

---

## 7  Detailed Generation Pipeline
1. **Load Configuration**
   ```python
   cfg = SyntheticConfig.from_yaml(path)
   random.seed(cfg.seed)
   Faker.seed(cfg.seed)
   ```
2. **Schema Instantiation** – Build empty `Statement` model with header metadata.
3. **Ledger Creation** (`ledger.py`)
   * Random starting balance.
   * Faker-driven transactions respecting debit/credit probability.
   * Running balance computed & validated.
4. **Rendering**
   * Choose renderer via flag (`--renderer reportlab|html`).
   * Embed logo, header, transaction table, footer.
5. **Noise Application** (`noise.py`)
   * Open each PDF page as image with Pillow; apply selected effects; re-embed.
6. **Ground-Truth Dump**
   * `statement.model_dump_json()` → `0001_enbd.json`.
   * `to_csv(statement)` → `0001_enbd.csv`.
7. **Validation & Manifest**
   * Re-load JSON via Pydantic → guarantee schema.
   * Optionally OCR a random sample with PaddleOCR for sanity.
   * Write `manifest.json` (config + git SHA + file checksums).

---

## 8  Testing Strategy
| Test | Purpose |
|------|---------|
| `test_schema.py` | Ensure JSON/CSV serialise & validate; check debit-credit maths. |
| `test_ledger.py` | Stable running balance across seeds; edge-case injection toggles work. |
| `test_renderers.py` | Smoke-generate 1-page PDF per renderer; OCR search for bank name & date. |

All tests run via `rye run pytest` and are enforced in CI.

---

## 9  Continuous Integration Hook
`synthetic-sanity.yml` (GitHub Actions):
1. Checkout code.
2. `rye sync`.
3. `rye run make-synthetic --statements 2 --renderer reportlab --output-dir tmp/gen`.
4. Run test-suite.
5. Upload the 2 PDFs as artefact for visual diffing.

---

## 10  Future Extensions
* **Template Marketplace** – Drop-in new HTML templates to widen layout diversity.
* **Image-based Statements** – Overlay generated PDF on background scan textures.
* **Language Localisation** – Arabic/French via right-to-left table rendering.
* **Adversarial Noise Modes** – Crumple, coffee-stain, scribbles for robustness testing.

---

## 11  Layout Ingestion & Template System

The repository now includes a helper utility that turns your **PNG samples** into machine-readable layout descriptors.

### 11.1  `tools/png_layout_ingest.py`
Usage example:
```bash
rye run python tools/png_layout_ingest.py \
  --src data/raw_pdfs/BankStatements \
  --dest layouts
```
The script will:
1. Walk every `*.png` under `--src`.
2. Parse file names like `AxisBank_Type_I_1.png` to derive a **layout key** `axisbank_type_i`.
3. Inspect the first page only; store:
   • image width & height  
   • 5 default column x-coordinates (percentage-based for now)
4. Write one JSON file per layout under `layouts/`, e.g.
   ```json
   {
     "width": 1654,
     "height": 2339,
     "columns": [82, 496, 1074, 1240, 1405]
   }
   ```

These descriptors are picked up by the renderers when present, allowing bank-specific column spacing without touching code.

### 11.2  First Custom Template
`synthetic_generator/render/templates/axisbank_type_i.html` reproduces a simplified Axis Bank colour scheme (#EF4E23 header). To generate with it:
```bash
rye run python -m synthetic_generator --layout axisbank_type_i --statements 2
```

More templates can be added by copying `default.html` and tweaking CSS.

---

## 12  Implementation Progress Log (June 30 2025)

### 12.1  Milestones Delivered

| Date | Deliverable | Notes |
|------|-------------|-------|
| 28 Jun 2025 | **MVP generator** skeleton (`cli`, `ledger`, dual renderers) | 5-column generic layout only |
| 29 Jun 2025 | **Round-robin layout selection** | Ensures different templates cycle when `--layouts a,b,c` passed |
| 30 Jun 2025 (AM) | **Bank-specific HTML templates** for `axisbank_type_i`, `bisb_type_i`, `enbd_type_i` | Distinct header colours, bilingual banner for BISB |
| 30 Jun 2025 (PM) | **Phase 2 Variant Engine** | New `schema` flag (`five_column`, `four_no_balance`, `random`); per-statement random font family + header colour; generic renderer path for 4-column schema |

### 12.2  Current Status Snapshot

* **Core modules**: stable; unit‐tested → `ledger`, `render`, `noise` (stub), `config`.
* **Templates present**: `default.html`, `axisbank_type_i.html`, `bisb_type_i.html` (v2 bilingual), `enbd_type_i.html`.
* **CLI flags**:
  ```bash
  --statements N              # number of PDFs
  --layouts a,b,c             # comma list of layout keys
  --renderer html|reportlab   # force renderer
  --schema five_column|four_no_balance|random  # table schema variant
  --seed 123                  # deterministic batch
  ```
* **Output example**:
  ```text
  data/synthetic/20250630_152008_batch/
    0001.pdf  0001.json  0001.csv
    manifest.json
  ```
* **Known gaps**:
  – Only one-page statements produced (no page carry-over logic yet).  
  – Noise pipeline not wired (blur/rotation/compression TODO).  
  – Balance column suppressed only in HTML path (ReportLab variant not updated).  
  – Limited template count vs. 40+ layouts we ultimately need.

### 12.3  Next Steps (Phase 2-B & Phase 3)

1. **Multi-page & Carried-Forward Rows**  
   • Add `max_rows_per_page` to `SyntheticConfig`.  
   • When exceeded, split statement, repeat header, inject "Carried forward" balance.
2. **Noise / Scan Appearance**  
   • Implement `noise.py` transformations (Gaussian blur, 200 dpi JPEG pass, ±0.4° rotation, speckle).  
   • Wire via `cfg.noise.enabled`.
3. **Template Backlog**  
   • Prioritise: `enbd_type_ii..v`, `sbi_type_iii`, `hdfc_type_i`, etc.  
   • Use shared partials to avoid duplication; leverage variant engine.
4. **Watermarks & Logos**  
   • Optional semi-transparent bank logos; `cfg.variant.watermark_prob`.
5. **ReportLab Parity**  
   • Port `schema` variants into `reportlab_renderer.py` for full renderer coverage.
6. **Documentation & Tests**  
   • Pytest cases for 4-column schema and multi-page splitting.  
   • CI artifact upload of latest batch for visual diffing.

> **ETA**: multi-page & noise within 1–2 days; additional templates delivered continuously thereafter.

---

© extraction-ai – Progress log appended 30 June 2025 