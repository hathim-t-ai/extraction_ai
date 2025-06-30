from __future__ import annotations

"""donut_table_extraction.py

Run DonutExtractor on a PDF and post-process the generated **text_sequence**
into a simple transactions table.

The heuristic parser is not template-specific; it scans the Donut output text
line-by-line looking for:
  • **Date** token at the line start              – e.g. ``1 Jan 2025`` or ``01 Feb``
  • Subsequent numeric tokens                     – candidate *amount* and *balance*
  • 3-letter currency code                       – optional

Once ≥2 numeric tokens are seen, the last one is treated as *balance* and the
preceding one as *amount*; everything between the date and the first number is
joined into the *description*.

The resulting DataFrame columns are: ``date, description, currency, amount, balance``.

Limitations
-----------
Works best on statements where each transaction appears on one physical line
in the PDF (or Donut output).  Multi-line wrapped descriptions may require
further logic.
"""

from pathlib import Path
import re
from typing import List, Any

import pandas as pd

from .donut_extraction import DonutExtractor

# ---------------------------------------------------------------------------
# Regex patterns (adapted from PaddleOCR row_parser)
# ---------------------------------------------------------------------------
_DATE_RE = re.compile(r"^(\d{1,2}) ([A-Za-z]{3,9})(?:,)?(?: (\d{4}))?$", re.I)
_AMT_TOKEN = re.compile(r"^-?(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d+[.,]\d{1,2})$", re.ASCII)
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_number(tok: str) -> float:
    tok = tok.rstrip(",.;").replace(" ", "").replace(",", "")
    if tok.count(".") > 1:  # thousand separator dots
        parts = tok.split(".")
        tok = "".join(parts[:-1]) + "." + parts[-1]
    return float(tok)


def _tokenise(line: str) -> List[str]:
    return [t.strip() for t in re.split(r"\s+", line.strip()) if t]


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_transactions_from_donut(df_pred: pd.DataFrame) -> pd.DataFrame:
    """Convert Donut prediction DataFrame into a transactions DataFrame."""
    all_lines: List[str] = []
    for txt in df_pred["prediction"]:
        try:
            # predictions are JSON-encoded strings – unwrap
            import json
            obj = json.loads(txt)
            seq = obj.get("text_sequence", "") if isinstance(obj, dict) else str(obj)
        except Exception:
            seq = str(txt)
        # split by newline and filter empty
        lines = [ln.strip() for ln in seq.split("\n") if ln.strip()]
        all_lines.extend(lines)

    records: List[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush():
        if not current:
            return
        if current.get("amount") is not None:
            desc = " ".join(current["desc"]).strip()
            currency = current.get("currency", "")
            records.append(
                {
                    "date": current["date"],
                    "description": desc,
                    "currency": currency,
                    "amount": current["amount"],
                    "balance": current.get("balance"),
                }
            )

    for line in all_lines:
        toks = _tokenise(line)
        if not toks:
            continue

        # detect line starting with date pattern (2 or 3 tokens)
        date_tok = ""
        if len(toks) >= 2 and _DATE_RE.match(" ".join(toks[:2])):
            date_tok = " ".join(toks[:2])
        elif len(toks) >= 3 and _DATE_RE.match(" ".join(toks[:3])):
            date_tok = " ".join(toks[:3])

        if date_tok:
            flush()
            current = {
                "date": date_tok,
                "desc": [line],
                "nums": [],
            }
            cur_curr = next((t for t in toks if _CURRENCY_RE.match(t)), "")
            if cur_curr:
                current["currency"] = cur_curr
        elif current is not None:
            current["desc"].append(line)
        else:
            continue  # skip lines before first date

        # collect numeric tokens in line, ignoring 4-digit years
        nums = [t for t in toks if _AMT_TOKEN.match(t) and not _YEAR_RE.fullmatch(t)]
        if current is not None:
            current.setdefault("nums", []).extend(nums)

        if current and len(current["nums"]) >= 2:
            try:
                bal = _clean_number(current["nums"][-1])
                amt = _clean_number(current["nums"][-2])
                current["amount"] = amt
                current["balance"] = bal
                flush()
                current = None
            except ValueError:
                pass

    flush()
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def extract_table_from_pdf(pdf_path: Path | str, output_path: Path | str) -> pd.DataFrame:
    """Run Donut over *pdf_path* and save transactions table to *output_path*.

    The output format depends on the *output_path* suffix – ``.csv`` or
    ``.xlsx`` are supported.
    """
    extractor = DonutExtractor()
    pred_df = extractor.extract_from_pdf(pdf_path)
    tx_df = parse_transactions_from_donut(pred_df)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.suffix.lower() == ".csv":
        tx_df.to_csv(out_p, index=False)
    elif out_p.suffix.lower() in {".xlsx", ".xls"}:
        tx_df.to_excel(out_p, index=False)
    else:
        raise ValueError("Unsupported output extension. Use .csv or .xlsx")
    return tx_df


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def cli(argv: List[str] | None = None) -> None:  # pragma: no cover
    import argparse, textwrap

    parser = argparse.ArgumentParser(
        prog="donut_table_extract",
        description=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument("pdf", type=Path, help="Input PDF path")
    parser.add_argument(
        "output",
        type=Path,
        help="Output file (.csv or .xlsx) for the transactions table",
    )
    args = parser.parse_args(args=argv)

    df = extract_table_from_pdf(args.pdf, args.output)
    print(df.head())


if __name__ == "__main__":  # pragma: no cover
    cli() 