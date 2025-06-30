# LayoutLMv3 Documentation

_Date: June 25, 2025_

---

## 1. What Is LayoutLMv3?

LayoutLMv3 is a **multimodal Transformer** purpose-built for Visually Rich Document Understanding (VRDU).  It jointly models:

* **Text** – the Unicode tokens from OCR
* **Layout** – the 2-D coordinates of each word on the page
* **Vision** – the raw pixels around each text region

By embedding all three signals into the same Transformer encoder, the model learns context such as:

> "The word *Amount* at (x≈540, y≈110) likely relates to the right-aligned numbers beneath it."

---

### 1.1  Core Ingredients

| Component | Purpose |
|-----------|---------|
| **Tokeniser**          | WordPiece → token IDs |
| **2-D Positional Emb.**| Adds (x, y, w, h) info to each token |
| **Patch Extractor**    | Splits the page image into 14×14/16×16 patches   (à la Swin) |
| **Shared Encoder**     | 12-layer BERT-style Transformer handling both text & patch tokens |
| **Task Head**          | Linear layer per token (NER), or CLS head (QA, classification) |

---

### 1.2  Pre-Training Objectives

1. **Masked Language Modelling (MLM)** – identical to BERT.  
2. **Masked Image / Vision Modelling (MIM/MVM)** – hide 40 % of patches and reconstruct them (BEiT style).  
3. **Text–Image Alignment** – binary classification: *Does this text match this image ?*

The joint objectives teach the encoder to correlate language, layout, and visual cues without manual labels.

---

### 1.3  Fine-Tuning

For downstream extraction we bolt-on a tiny task-specific head:

* **Token-classification** (field/NER) – used in our demo.  
* **Sequence-classification** (document type, QA, etc.).

Only this head's weights are randomly initialised; the encoder re-uses the multimodal knowledge from pre-training.

---

## 2. Demo Workflow Recap (scripts/demo_layoutlm_extraction.py)

1. **PDF → Words + Boxes**  
   *PyMuPDF* extracted each word and its pixel-level bounding-box.  Coordinates were rescaled to the 0-1000 range so every page shares the same reference frame.

2. **Processor Step**  
   `LayoutLMv3Processor` built four tensors:
   * `input_ids` (text tokens)  
   * `bbox`      (box per token)  
   * `pixel_values` (page image)  
   * `attention_mask`

3. **Model Forward Pass**  
   * Text-embedding + 2-D position      
   * Patch-embedding + 2-D position     
   * 12-layer self-attention fusion     
   * Linear classification head → logits per token.

4. **Argmax & CSV Export**  
   The script mapped logits → label IDs (`LABEL_0`, `LABEL_1`, …) and wrote one CSV row per word:

   ```csv
   pdf,page,word,label,x0,y0,x1,y1
   ENBD-statements-pdf.pdf,1,Available,LABEL_9,309,134,365,147
   …
   ```

> **Note :** We used the *base* checkpoint `microsoft/layoutlmv3-base`.  The task head is therefore **untrained**, so label IDs are effectively random.  Accurate extraction requires fine-tuning on annotated bank-statement data.

---

## 3. How a Trained LayoutLMv3 Extracts Tables

1. Annotate tokens in ~250–1 000 sample statements: `DATE`, `DESCRIPTION`, `AMOUNT`, …  
2. Fine-tune for 3-5 epochs → the classification head learns those tags.  
3. Inference tags every word.  
4. Post-processing groups contiguous tokens with the same label into cells, then assembles rows.

Result: a structured, machine-readable transaction table with human-level accuracy on complex layouts.

---

### 3.1  Strengths

* Handles irregular layouts (multi-column, rotated headers).  
* Uses visual signals (rules, table lines) missing from pure text models.  
* Learns context (e.g., a right-aligned number below *BALANCE* is probably a balance field).

### 3.2  Limitations

* Requires high-quality bounding-boxes; OCR noise propagates.  
* Fine-tuning dataset must cover target layouts & fonts.  
* Large (~450 MB) and slower than pure-text NER models.

---

## 4. Next Steps for This Project

1. **Decrypt / repair** the YourBank PDF or use a clean copy (the version in repo is encrypted).  
2. **Create an annotation set** using the existing ground-truth CSVs.  
3. **Fine-tune** `layoutlmv3-base` (or a smaller layout-aware model) for token-classification.  
4. **Integrate** the inference pipeline into your 3-stage parser (replace current rule-based Stage 1).

---

## 5. End-to-End Fine-Tuning Pipeline (Bank-Statement Extraction)

> This section explains **how to move from the random-label demo to a fully-trained LayoutLMv3 extractor**.  It mirrors the structure of `PADDLEOCR_DOCUMENTATION.md` so both engines follow consistent documentation standards.

### 5.1  Directory/Script Overview

| Path | Purpose |
|------|---------|
| `data/raw_pdfs/` | Source PDFs (all banks & layouts) |
| `data/annotations/` | Human-labelled JSONL exports (`*.jsonl`) with token-level BIO tags |
| `scripts/examine_training_data.py` | Converts/inspects annotation exports and builds Hugging Face `Dataset` splits |
| `src/models/layoutlm_finetune.py` | Fine-tunes `microsoft/layoutlmv3-base` → `models/layoutlm_finetuned/` |
| `scripts/demo_layoutlm_extraction.py` | Batch inference: PDF → CSV/JSON table using the fine-tuned checkpoint |
| `models/layoutlm_finetuned/` | Saved model + processor after training |
| `docs/LayoutLmv3_finetuning_guide.md` | (this file) Expanded training hyper-parameters, tips & FAQs |

> **Note:** All three components now exist.  Dataset preparation lives in `scripts/examine_training_data.py`, fine-tuning in `src/models/layoutlm_finetune.py`, and batch inference in `scripts/demo_layoutlm_extraction.py`.

### 5.2  Data Preparation Steps

1. **OCR + Boxes**  
   Use your existing PyMuPDF extractor to dump `words`, `bbox`, and page `image` for *every* page.
2. **Manual Annotation**  
   Import those JSONs into Label Studio (or similar) and tag tokens with the BIO schema: `B-DATE`, `I-DESCRIPTION`, `B-AMOUNT`, …
3. **Export & Convert**  
   `scripts/examine_training_data.py` merges the annotation JSON and image into the HF `Dataset` format with fields:
   ```json
   {
     "id": "page-001",
     "words": ["28-Jun-25", "POS", "STARBUCKS", "-18.50"],
     "bbox":  [[43,91,103,105], …],
     "labels": [3, 4, 5, 6],           # int ids per token
     "image":  "page_001.png"
   }
   ```
4. **Dataset Splits** – 80 % train, 10 % validation, 10 % test saved under `data/training/layoutlm/`.

### 5.3  Training Loop (src/models/layoutlm_finetune.py)

```python
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    id2label=id2label,
    label2id=label2id,
)
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
...
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="models/layoutlm_finetuned",
        learning_rate=5e-5,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        fp16=True,
        evaluation_strategy="steps",
        save_steps=500,
    ),
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    tokenizer=processor,
    compute_metrics=seqeval_metrics_fn,
)
trainer.train()
```
*Expected training time:* ≈ 2 h on a single RTX 3090 (24 GB) for 1 000 pages.

### 5.4  Inference Pipeline (scripts/demo_layoutlm_extraction.py)

1. **Load model/processor** from `models/layoutlm_finetuned/`  
2. **Extract OCR words + boxes** for new PDF.  
3. **processor(..., boxes=bbox)** → tensors.  
4. **model(**tensors)** → logits → BIO labels.  
5. **Group spans** into structured rows: `date, description, debit, credit, balance`.  
6. *Optional:* confidence filtering; fallback to rule-based if confidences < 0.5.

### 5.5  QA & Continuous Improvement

* **Metric targets**: per-token F1 ≥ 0.95, per-row accuracy ≥ 0.90 on test set.  
* **Error analysis**: add mis-predicted pages back to annotation pool and re-train monthly.  
* **Model versioning**: save checkpoints with semantic version tags (`v1.0.0`, `v1.1.0`, …) in `models/` and record evaluation numbers in `Outputs/model_evaluation_report.json`.

---

## 6. Required External Dependencies

| Package | Purpose | Already in `pyproject.toml`? |
|---------|---------|------------------------------|
| `transformers>=4.52` | Model + Trainer | ✅ |
| `datasets>=3.6`      | HF Dataset          | ✅ |
| `seqeval`            | BIO evaluation      | ❌ (add via Rye) |
| `label-studio`       | Annotation UI       | (optional) |

Add missing libs via:
```bash
rye add seqeval --dev
```

---

_This section appended on June 25 2025._

_© PDF-Parser Project – LayoutLMv3 deep-dive documentation_ 