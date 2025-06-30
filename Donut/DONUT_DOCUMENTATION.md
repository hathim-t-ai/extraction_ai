# Donut LLM – Bank-Statement Extraction Primer

Date: June 27, 2025

---

## 1. What *is* Donut?

**Donut** (Document Understanding Transformer) is an **OCR-free** vision-encoder–decoder model for document AI.

| Component | Architecture | Role |
|-----------|--------------|------|
| **Encoder** | Swin-Transformer | Converts an RGB page image into a sequence of visual patch embeddings. |
| **Decoder** | BART-style transformer | Autoregressively generates a *text sequence* (typically JSON). |

Key characteristics:

* **No bounding-boxes needed** – the model implicitly learns to recognise characters *and* layout.
* **Prompt-driven** – a special token (e.g. `<s_docvqa>`, `<s_receipt>`) tells the decoder *what* to output.
* **Flexible output** – key-value pairs, tables, free-form answers… anything tokenisable.

> **Heads-up** The Donut "base" checkpoint (`naver-clova-ix/donut-base`) is *just pre-trained*. It is **not** an out-of-the-box OCR engine – that's why our quick test produced an empty `text_sequence`.

---

## 2. Public Hugging Face checkpoints

| Checkpoint | Task it was fine-tuned for |
|------------|---------------------------|
| `donut-base` | **None** – pre-training only (needs fine-tuning). |
| `donut-base-finetuned-docvqa` | Visual question answering (DocVQA). |
| `donut-base-finetuned-docbank` | Generic OCR token extraction. |
| `donut-base-finetuned-rvlcdip` | Document classification. |
| `donut-base-finetuned-over-sea` | Receipt key-value extraction (CORD). |
| `donut-base-finetuned-cord-v2` | Korean receipts, key-values. |
| `donut-base-finetuned-cord-v2-lineitem` | **Line-item table extraction** – closest to bank-statement rows. |
| `donut-base-finetuned-iam-handwriting` | Hand-written text transcription. |

Choose a checkpoint that matches your task *or* fine-tune your own as described below.

---

## 3. Why the base model returns an empty string

Our demo called `DonutExtractor` with the *base* model + the default prompt `<s>`. Because the model has **never been trained** to map that prompt to a target sequence, it terminates early and returns an empty JSON:

```json
{"text_sequence": ""}
```

To get structured fields (date, description, amount, balance), we must fine-tune the model or switch to a checkpoint that already outputs those fields.

---

## 4. Fine-tuning workflow ("a dozen annotated PDFs")

> **A dozen annotated PDFs** simply means ~12 PDF documents for which you have ground-truth data – i.e. the correct transactions table expressed as JSON.

### 4.1 Data preparation
1. **Render pages** to images (e.g. 300 DPI with PyMuPDF).
2. **Define a new prompt token** and output schema, e.g.

   ```text
   <s_bank_statement>
   { "transactions": [
       { "date": "01 Jan 2025", "description": "Salary", "amount": "10000", "balance": "12000" },
       ...
   ] }
   ```
3. For every page, save `(image, target_string)` pairs in a HF `datasets` structure or JSONL.

### 4.2 Training command (simplified)

```bash
python donut_finetune.py \
  --train_dir data/finetune_jsonl \
  --pretrained_model naver-clova-ix/donut-base \
  --prompt_token "<s_bank_statement>" \
  --output_dir checkpoints/donut-bank-statement \
  --epochs 10 --batch 2 --lr 1e-5
```
* LoRA-style or partial-freeze training is common for small datasets.

### 4.3 Inference & CSV export

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
from Donut.donut_table_extraction import parse_transactions_from_donut
import pandas as pd, json, fitz, PIL.Image as Image

ckpt = "checkpoints/donut-bank-statement"
processor = DonutProcessor.from_pretrained(ckpt)
model = VisionEncoderDecoderModel.from_pretrained(ckpt).eval()

# render a page…
img = Image.open("page_1.png")
pixel = processor(img, return_tensors="pt").pixel_values
prompt = processor.tokenizer("<s_bank_statement>", add_special_tokens=False, return_tensors="pt").input_ids
out = model.generate(pixel, decoder_input_ids=prompt)
json_str = processor.batch_decode(out, skip_special_tokens=True)[0]
rows_df = pd.DataFrame(json.loads(json_str)["transactions"])  # -> to_csv / to_excel
```

---

## 5. Key takeaways

* Donut is **OCR-free** but *not* generic – you must match the checkpoint to your task.
* A custom prompt token + a handful of annotated PDFs is usually enough for good results.
* Once fine-tuned, converting the model output to `.csv`/`.xlsx` is just `json.loads` → `pandas`.

---

## 6. Why the CORD-V2 model produced almost no text on our PDFs

During a quick test we used `naver-clova-ix/donut-base-finetuned-cord-v2` with the **default prompt** `<s>`.

1. **Prompt token matters** – Every Donut checkpoint learns a *task-specific start token* that must precede decoding.  For CORD-V2 that token is
   ```text
   <s_cord-v2>
   ```
   If the model receives a different prompt it usually terminates immediately, yielding an empty `text_sequence` as you observed.

2. **Domain mismatch** – Even with the correct prompt this checkpoint is tuned on Korean/English retail **receipts**. Its JSON schema contains keys like `menu`, `total_price`, etc.  Multi-page bank statements do not match that schema, so the extraction will still be unusable for our tables.

In practice you have two options:
• Provide the right prompt just to see non-empty receipt-style JSON.
• (Recommended) Fine-tune a new checkpoint with a bank-statement prompt such as `<s_bank_statement>` and an output schema that lists transactions.

*Added: June 25 2025 – explanation following initial testing mishap.*

---

*Last updated: June 25, 2025*

---

## 7. End-to-end roadmap for a fine-tuned "bank-statement Donut"

Below is a condensed but complete blueprint – from blank slate to a production-ready checkpoint that emits a JSON table of transactions and can be exported to CSV/XLSX.

### 7.1 Scope & prerequisites
* **Goal** Generate
  ```json
  { "transactions": [ {"date":..., "description":..., "amount":..., "balance":...} ] }
  ```
* **Hardware** Any single modern GPU (A100 fastest). CPU/M-series possible (see § 8).
* **Code base** Current repo already hosts Donut wrappers in `Donut/`.

### 7.2 Prompt token & schema
```
<s_bank_statement>
```
Keep keys lower-case, stable order, minimal spaces; Donut memorises exact characters.

### 7.3 Data labelling (≈ 10-20 statements)
1. Render each PDF page → 300 DPI PNG.
2. Create canonical JSON per page following the schema (you can script from existing CSV ground truth).
3. Build `metadata.jsonl` per split:
   ```jsonl
   {"file_name":"0001.png", "ground_truth":"{\"gt_parse\": {…}}"}
   ```
   Hugging Face `datasets` can read this directly (see § 5 for loading snippet).

### 7.4 Sanity checks
Small Python script: load JSON, assert 4 keys, check token length < 1024.

### 7.5 Training config (LoRA example)
| Param | Value |
|-------|-------|
| base model | `naver-clova-ix/donut-base` |
| prompt token | `<s_bank_statement>` (added to tokenizer) |
| encoder | frozen |
| decoder | LoRA rank 8 or full FT |
| res  | 1920 × 2560 |
| batch | 4 (fp16) |
| epochs | 10–15 |
| lr | 1e-4 → 1e-5 cosine |
| loss | token CE |

### 7.6 Launch training
```bash
rye run python -m Donut.train \
  --config config/train_bank.yaml \
  --pretrained_model_name_or_path naver-clova-ix/donut-base \
  --output_dir checkpoints/donut-bank-v1
```
Monitor edit-distance on val set; < 0.15 is usually sufficient.

### 7.7 Evaluate
Convert predictions → `pandas.DataFrame`, compute recall/precision of dates & amounts, manually inspect worst pages.

### 7.8 Package & version
```bash
cp -r checkpoints/donut-bank-v1 Donut/bank_statement_ckpt
huggingface-cli upload checkpoints/donut-bank-v1 --repo bank-statement-donut   # optional
```

### 7.9 Integration
Update `donut_table_extraction.py` to load the new ckpt and pass `<s_bank_statement>` prompt.

### 7.10 Timeline
* Data prep 1–2 days  
* Train on single A100 < 1 h per 1 k pages  
* Eval & polish 1 day  
Total ≈ 3 working days for first pass.

---

## 8. Training when you only have an M-series Mac

Apple-silicon lacks CUDA but you still have three paths:

### 8.1 Train locally on **MPS** backend
```python
import torch
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
```
* Install universal wheels:  
  `pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cpu`
* Freeze encoder + LoRA → fits in 16 GB RAM.
* Speed: ~0.4 images/s → 500 pages × 10 epochs ≈ 3 hours.

### 8.2 Rent a cloud GPU briefly
| Provider | Example | $/h |
|----------|---------|-----|
| Google Colab Pro | T4 / A100 | 0–1 |
| Paperspace | A100-40 GB | 1.2 |
| AWS Spot | `g5.xlarge` (A10G) | 0.35 |
Export ckpt and copy back to your Mac for inference.

### 8.3 Managed training services
* **Hugging Face AutoTrain Advanced**: upload dataset, pay per GPU-hour.  
* **Google Vertex AI Training**: Docker submit; GPU auto-provisioned.

### 8.4 Decision matrix
| Dataset size | Turn-around | Recommended |
|--------------|------------|-------------|
| ≤ 300 pages | Overnight OK | local MPS |
| 300–2 000 | < 3 h | rented A10G / Colab |
| > 2 000 | asap | A100 / managed |

Once fine-tuned, inference is CPU-friendly, so deployment/testing remains fully local on your Mac.

### 8.5 Reality check for an 8 GB MacBook Air (M1, 2020)

Your exact machine (fan-less Air, 8 GB unified RAM) can handle **inference** but hits two limits when fine-tuning:

1. **Memory ceiling** – Full Donut + activations ≈ 10 GB even with fp16 & LoRA. macOS will swap → training hangs or crashes.
2. **Thermals / speed** – Sustained back-prop throttles the M1 Air; each step becomes several seconds.

What *does* work locally:
• Prompt-tuning only (train just the new `<s_bank_statement>` embedding) – fits well within 8 GB, but convergence is limited.
• All extraction / evaluation scripts.

Recommended production workflow:
1. Prepare PNGs + `metadata.jsonl` on the Air.
2. Upload to cloud storage.
3. Spin up a cheap GPU (Colab/Paperspace/AWS Spot) for < $2 and run the training command from § 7.6.
4. Download the resulting checkpoint; inference thereafter is CPU-only and runs fine on the Air.

*Appended: June 25 2025 – hardware footnote for 8 GB M1 users*

*Appended: June 25 2025*

## 9. Clarifications & Deep-Dives (Requested: June 28 2025)

### 9.1 Train locally on **MPS** backend – follow-up questions

**a) What does `torch` do?**  
PyTorch is the deep-learning framework Donut is built on. It provides:
- N-dimensional `Tensor` objects + highly optimised math kernels.
- Automatic differentiation (autograd) so gradients flow through the model.
- `nn.Module` abstractions for neural networks.
- Device management (`cpu`, `cuda`, `mps`, …) so a single model can run unmodified on Apple Silicon.

In § 8.1 we import `torch` solely to detect the *Metal Performance Shaders* (MPS) backend:
```python
import torch
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
```

**b) Why install the *universal wheels*?**  
A *wheel* (`*.whl`) is Python's binary package format. The *universal CPU* wheels published at
```
https://download.pytorch.org/whl/cpu
```
contain *architecture-agnostic* (x86 + arm64) **CPU-only** builds. On macOS this avoids pulling an incompatible CUDA build and exposes the `mps` backend automatically:
```bash
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cpu
```

**c) What does *Freeze encoder + LoRA* mean?**  
Donut = Swin-Transformer *encoder* + BART-style *decoder*.
- *Freezing the encoder* sets `requires_grad=False` so no gradients nor weight updates are computed → ~50 % less memory.
- *LoRA* (Low-Rank Adaptation) injects two tiny rank-`r` matrices into every trainable weight and trains only those (`r≈8-16`). Quality ≈ full fine-tune but memory/FLOPs scale with `r`, keeping the whole model inside 16 GB of unified RAM.
- At inference time the adapters are merged – **zero runtime penalty**.

---

### 9.2 Cloud training workflows: Google Colab Pro vs Hugging Face AutoTrain Advanced

| Step | Colab Pro (Notebook) | HF AutoTrain Adv (UI/API) |
|------|----------------------|---------------------------|
| 1 | Create GPU runtime (T4/A100). | Create new *Vision + Text* project. |
| 2 | Upload or mount `dataset.zip`. | Upload same archive or link HF dataset. |
| 3 | `rye install` deps, clone repo. | Pick **Base model** → `naver-clova-ix/donut-base`. |
| 4 | Run training script:<br>`rye run python Donut/train.py ...` | Configure hyper-params (epochs, LR, *LoRA*, eval metric). |
| 5 | Checkpoints save to Drive. | AutoTrain trains & evaluates automatically. |
| 6 | `zip` checkpoint → download. | Click **Download model** / **Push to Hub**. |
| 7 | Integrate with `donut_table_extraction.py`. | Same integration step. |

Linear checklist (works for both services):
1. **Dataset** Every sample = PNG + JSON string (**including** prompt):
   ```json
   {"file_name":"0001.png", "ground_truth":"{\"transactions\":[…]}", "prompt":"<s_bank_statement>"}
   ```
2. **Package** Place PNGs in `images/`, zip with `train.jsonl`, `val.jsonl`:
   ```bash
   zip dataset.zip images/*.png train.jsonl val.jsonl
   ```
3. **Train** Colab → notebook; AutoTrain → web UI.
4. **Retrieve checkpoint** Zip & download or *Push to Hub*.
5. **Verify locally (CPU)**
   ```python
   from Donut.donut_table_extraction import parse_transactions_from_donut
   parse_transactions_from_donut("donut_ckpt", "<s_bank_statement>", "page_1.png")
   ```
6. **Document** run metadata in `DEVELOPMENT_DECISIONS.md` (GPU type, hours, cost).

---

### 9.3 "Once fine-tuned, inference is CPU-friendly"
Training requires back-prop ⇒ lots of activations + gradients ⇒ GPU/MPS recommended. Inference is just a forward pass (~1 GFLOP per page). A modern laptop CPU (M-series ≈ >200 GFLOPS) finishes < 1 s/page, RAM ≈ 0.5 GB. Therefore deployment can be **GPU-free**, reducing costs and simplifying ops.

---

### 9.4 Preparing PNGs + `metadata.jsonl` on a MacBook Air

Install tools:
```bash
rye add pymupdf pillow pandas tqdm
```

**A. Render PDFs → PNG (300 DPI)**
```python
import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm

pdf_dir = Path("data/raw_pdfs")
png_dir = Path("data/png_pages")
png_dir.mkdir(exist_ok=True)

for pdf_path in pdf_dir.glob("*.pdf"):
    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(tqdm(doc, desc=pdf_path.name)):
            pix = page.get_pixmap(dpi=300, colorspace=fitz.csRGB)
            out_name = f"{pdf_path.stem}_{page_idx:03}.png"
            pix.save(png_dir / out_name)
```

**B. Convert CSV ground truth → Donut JSON strings**
```python
import pandas as pd, json, re
from collections import defaultdict
from pathlib import Path

gt_root = Path("data/annotations")
page_jsons = defaultdict(list)

for csv_path in gt_root.glob("*_ground_truth.csv"):
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        page_tag = re.sub(r"\.pdf$", "", row["source_pdf"]) + f"_{row['page']:03}"
        tx = {
            "date": row["date"],
            "description": row["description"],
            "amount": str(row["amount"]),
            "balance": str(row["balance"]),
        }
        page_jsons[page_tag].append(tx)

donut_targets = {tag: json.dumps({"transactions": txs}, ensure_ascii=False)
                 for tag, txs in page_jsons.items()}
```

**C. Build `train.jsonl` / `val.jsonl`**
```python
import json, random
from pathlib import Path

all_tags = list(donut_targets.keys())
random.shuffle(all_tags)
cut = int(0.8 * len(all_tags))
splits = {"train": all_tags[:cut], "val": all_tags[cut:]}

for split, tags in splits.items():
    with open(f"{split}.jsonl", "w", encoding="utf-8") as f:
        for tag in tags:
            f.write(json.dumps({
                "file_name": f"{tag}.png",
                "ground_truth": donut_targets[tag],
                "prompt": "<s_bank_statement>"
            }, ensure_ascii=False) + "\n")
```

**D. Sanity check**
```python
from datasets import load_dataset
import json

ds = load_dataset("json", data_files={"train": "train.jsonl", "val": "val.jsonl"})
assert "transactions" in json.loads(ds["train"][0]["ground_truth"])
```

**E. Zip for upload**
```bash
zip -r dataset.zip data/png_pages train.jsonl val.jsonl
```

Your `dataset.zip` is now ready for Colab or AutoTrain.

---

*Appended: June 28 2025 – detailed clarifications following user Q&A* 