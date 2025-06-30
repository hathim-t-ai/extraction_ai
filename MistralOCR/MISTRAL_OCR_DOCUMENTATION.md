# Mistral OCR – Engineering Notes

Date: June 25 2025

---

## 1. End-to-end Processing Pipeline

1. **Rasterisation** – Each PDF page is rendered to RGB at 200 DPI via PyMuPDF (see `_pdf_pages_to_images`).
2. **Base-64 packaging** – The PIL image is encoded as inline `data:image/png;base64,…`.
3. **HTTPS call** – `client.ocr.process(model="mistral-ocr-latest", document={…})` hits `POST /v1/ocr`.
4. **Vision + LLM inference (Cloud)**
   * ViT encoder → layout detector → LLM decoder.
   * Output: Markdown + `OCRPageObject` JSON.
5. **Local normalisation** – `MistralOCRExtractor` stores one record per page in a `pandas.DataFrame` with columns:
   `pdf`, `page`, `markdown`, `ocr_json`.

### Why Markdown ➜ HTML ➜ DataFrame?
Markdown tables are free-text; converting to HTML with `markdown` gives parseable `<tr>/<td>` nodes, which BeautifulSoup turns into a tidy matrix. The matrix → `DataFrame` allows numeric type casting & CSV/XLSX export.

```python
html = markdown.markdown(table_md, extensions=["tables"])
rows = [[td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
        for tr in BeautifulSoup(html, "html.parser").find_all("tr")]
df = pd.DataFrame(rows[1:], columns=rows[0])
```

---

## 2. Schema-Driven Extraction

The API can return **structured JSON** directly. Pass a JSON-Schema under `document_annotation_format`.

```python
schema = {
  "type": "object",
  "properties": {
    "transactions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "date":        {"type": "string"},
          "description": {"type": "string"},
          "out":         {"type": "number"},
          "in":          {"type": "number"},
          "balance":     {"type": "number"}
        },
        "required": ["date", "description", "balance"]
      }
    }
  },
  "required": ["transactions"]
}

resp = client.ocr.process(
    model="mistral-ocr-latest",
    document=doc_ref,
    document_annotation_format={"type": "json_schema", "schema": schema}
)
```

`resp.pages[0].document_annotation["transactions"]` already contains structured rows – minimal post-processing needed.

---

## 3. Fine-Tuning Workflow

| Step | Action |
|------|--------|
| **Dataset** | Each *page* is a *document* (image/URL) + **annotation JSON** matching your schema. |
| **Upload** | `file_id = client.files.upload("train.jsonl", purpose="fine-tune").id` |
| **Create job** | `job = client.ocr.fine_tunes.create(base_model="mistral-ocr-latest", training_file=file_id, n_epochs=3, suffix="bankstatements")` |
| **Monitor** | `client.ocr.fine_tunes.retrieve(job.id).status` → queued → running → succeeded |
| **Use model** | `model="mistral-ocr-ft:bankstatements:2025-06-25"` in subsequent `ocr.process` calls |

*Compute & cost:* managed GPUs; ~2 GPU-hours for 2 k pages ⇒ ≈ $8 compute + $25 orchestration ≈ $33 total.

*Active-learning loop:* start with 30 real pages → fine-tune → correct predictions → append to dataset → repeat.

---

## 4. Label-Creation Options

| Tool | Notes |
|------|-------|
| **Label Studio** | Free; JSON schema tasks; 150 pages/day speed with hotkeys. |
| **doccano** | Simple text tables; CSV/JSON export. |
| **Kili-lite** | Hierarchical JSON; good UI. |
| **CVAT** | Needed only for bbox-level labels. |

*Synthetic augmentation* – Generate fake statements via ReportLab/Faker, save the JSON used to render as ground truth.

---

## 5. Privacy & Data Residency

*   Transport: TLS1.3.  
*   Storage: encrypted bucket, auto-purged ≤30 days.  
*   Option `x-mistral-no-retention: true` removes even that window (Enterprise).  
*   Region pinning header `x-mistral-region: me-central-1` keeps data in UAE.

---

## 6. Quick-Start Cheat-Sheet

```bash
# Extract Markdown & JSON
python -m MistralOCR.mistral_ocr_extraction data/raw_pdfs/YourBank.pdf Outputs/YourBank_mistral.jsonl

# Convert to transactions DataFrame
python tools/markdown_to_tx.py Outputs/YourBank_mistral.jsonl Outputs/YourBank_tx.csv
```

> For schema output simply pass the schema dict and skip the conversion step.

---

© extraction-ai project – Mistral OCR notes 