# Comparative Guide – Bank-Statement Transaction Extraction Engines

Date: June 28 2025

---

## Contents
1. Executive Summary
2. Side-by-Side Matrix
3. Engine Deep-Dives
   - 3.1 Donut
   - 3.2 LayoutLMv3
   - 3.3 Mistral OCR (Cloud)
   - 3.4 PaddleOCR + Rule Engine
4. Synthetic Bank-Statement PDF Generation (Free Workflow)
5. Recommendations & Next Steps

---

## 1  Executive Summary
Four extraction approaches have been prototyped in this repository. They differ in the way they consume documents, in annotation effort, in compute/cost profile, and in how well they handle messy layouts.

| Solution      | Core Idea                                   | Best For                              |
|---------------|---------------------------------------------|----------------------------------------|
| **Donut**     | OCR-free vision-encoder/decoder → JSON      | Small, quick fine-tunes; low annotation|
| **LayoutLMv3**| Multimodal encoder (text + layout + patches) | Highest accuracy on irregular layouts  |
| **Mistral OCR**| Managed REST API with vision + LLM         | Zero-infrastructure, fast bootstrapping|
| **PaddleOCR** | Classical OCR + deterministic Python rules  | Offline baseline, zero ML cost         |

---

## 2  Side-by-Side Matrix

| Dimension                 | Donut            | LayoutLMv3          | Mistral OCR (API) | PaddleOCR |
|---------------------------|------------------|---------------------|-------------------|-----------|
| Annotation effort         | low (page-level) | high (token-level)  | low-mid (page)    | none (rules)|
| Pages to start            | 10–20            | 250–1 000           | 30–200            | 0         |
| Fine-tune hardware        | 12 GB GPU        | 24 GB GPU           | Cloud-managed     | –         |
| Fine-tune wall-clock      | ~45 min          | ~2 h                | ~40 min           | –         |
| Inference speed (CPU)     | 2–3 s/page       | 4–5 s/page          | 1 HTTP call       | 1 s/page  |
| Inference unit cost       | free             | free               | ~$0.01/page       | free      |
| Integration complexity    | ⭐⭐              | ⭐⭐⭐                | ⭐                | ⭐⭐       |
| Complex-layout handling   | good             | excellent           | good              | rule-dependent |

(⭐ = easiest)

---

## 3  Engine Deep-Dives

### 3.1  Donut (Document Understanding Transformer)
**What it does** Vision encoder + BART-style decoder that emits arbitrary text (usually JSON) from a page image. No separate OCR step.

**Tech** Swin-Transformer encoder, prompt token to indicate schema, trained end-to-end with cross-entropy.

**Out-of-box output** The plain `donut-base` checkpoint returns an empty `text_sequence` on bank statements because it was never taught a bank-statement prompt.

**Fine-tune & annotation**
* Annotation unit: *page PNG + target JSON string*.
* 10–20 pages give usable accuracy.
* Example prompt:
  ```text
  <s_bank_statement>
  {"transactions":[{"date":"01 Jan 25","description":"Salary","amount":"10000","balance":"12000"}]}
  ```
* Label tools: JSON editor, or script that converts your CSV ground truth.
* Training: LoRA or full fine-tune; freeze encoder to fit in ≤12 GB VRAM.

**Timing** ~45 min for 1 k pages ×10 epochs on A100; 3 h on M-series Mac.

**Ease / complexity** Low annotation; single model call at inference; JSON→CSV.

**Cost** Open-source; GPU rental only (~$1–2 per run).

---

### 3.2  LayoutLMv3
**What it does** Fuses OCR tokens, 2-D coordinates and visual patches; predicts a label per token.

**Tech** BERT-style shared encoder, masked-language + masked-image pre-training.

**Out-of-box output** Random labels; unusable until fine-tuned.

**Fine-tune & annotation**
* Annotation unit: each word gets a BIO tag (`B-DATE`, `I-DESC` …) and a bounding-box.
* Requires OCR dump first (PyMuPDF/PaddleOCR) → annotate in **Label Studio, doccano, Kili, CVAT**.
* 250–1 000 pages recommended.
* HF `Trainer`; 3–5 epochs.

**Timing** ~2 h for 1 k pages on RTX 3090.

**Ease / complexity** Highest annotation cost; strongest accuracy on messy tables.

**Cost** GPU time ~$3–5; inference free but slower.

---

### 3.3  Mistral OCR (Managed Cloud)
**What it does** Vision encoder + LLM decoder behind REST. Returns Markdown or schema-constrained JSON.

**Tech** ViT encoder, proprietary instruction-tuned decoder; constrained decoding with user-provided JSON Schema.

**Out-of-box output** Good Markdown tables (~70-80 % correct) or JSON when schema supplied.

**Fine-tune & annotation**
* JSONL lines `{image:…, annotation:{transactions:[…]}}`.
* Upload, start fine-tune via API; no GPUs on your side.
* 30–200 pages enough due to strong base model.

**Timing** Managed; 2 GPU-hours ≈ 40 min.

**Ease / complexity** Easiest; zero infrastructure but vendor lock-in.

**Cost** Per-page (~$0.008–0.015). Sample fine-tune ≈ $33.

---

### 3.4  PaddleOCR + Rule Engine
**What it does** Classical OCR engine → raw lines; custom Python merges lines into transactions.

**Tech** DBNet detector, CRNN recogniser, optional angle classifier; deterministic clustering & balance validator.

**Out-of-box output** Raw lines; with your rules: 97 % recall/precision on ENBD, 14 % on YourBank (pending 3-line fix).

**Fine-tune & annotation** None. Improvement = tweak Python rules; add yaml aliases.

**Timing** Rule edit/test cycles in seconds.

**Ease / complexity** Easy to run; rule logic grows with layout diversity.

**Cost** Open-source; negligible compute.

---

## 4  Synthetic Bank-Statement PDF Generation (Free Workflow)

1. **Install libs**
   ```bash
   rye add reportlab faker jinja2 pypdf2 pillow
   ```
2. **Define canonical JSON schema** (same one you will train Donut/Mistral with).
3. **Data generator** – use `faker` to create random companies, dates, amounts; maintain running balance.
4. **Render PDFs**
   * **ReportLab** for programmatic drawing, or 
   * **HTML → PDF** via Jinja2 + WeasyPrint for rapid layout switching.
5. **Pair ground-truth** – save the JSON next to each PDF; optionally export CSV.
6. **Variations** – swap templates, fonts, add watermarks, Gaussian blur to mimic scans.
7. **Scale** – 1 000 pages ≈ 3 min CPU-only; disk <150 MB.
8. **Uses** – pre-train Donut/LayoutLMv3, active-learn Mistral, stress-test Paddle rules.

---

## 5  Recommendations & Next Steps

* **Quick win** – Fine-tune Donut on 15–20 real pages (auto-generate JSON from your CSV ground truth).  <br/>
* **Long-term accuracy** – Plan a 500-page BIO annotation sprint → LayoutLMv3 fine-tune.  <br/>
* **Infrastructure-free pilot** – Wrap Mistral OCR with the JSON Schema above; watch cost metrics.  <br/>
* **Offline fallback** – Continue hardening PaddleOCR rule-engine; keep it as safety net.  <br/>
* **Leverage synthetic PDFs** immediately for data augmentation and regression testing.

---

© extraction-ai project – Comparative guide authored 28 June 2025 