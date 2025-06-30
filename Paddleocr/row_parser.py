from __future__ import annotations
"""Rule-engine post-processor: convert PaddleOCR line CSV into a table of
(bank) transactions.

The algorithm is template-free and works page-by-page:
1. Cluster OCR lines by *y* centre (merge wrapped descriptions).
2. Within a cluster join the partial strings left→right.
3. A lightweight state-machine walks the clusters:
   • whenever it sees a date token (``DD Mon`` or ``DD Mon YYYY``)
     it opens a new *buffer*; subsequent clusters are appended.
   • once ≥2 numeric tokens are seen → treat last as balance, previous as amount.
4. Everything between date and amount becomes the description, run through an
   optional alias table (``rules/description_aliases.yml``).

Down-stream validation can still correct sign errors by enforcing the running
balance arithmetic.
"""

from pathlib import Path
import ast
import re
from typing import Any, List

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
_DATE_RE = re.compile(r"^(\d{1,2}) ([A-Za-z]{3,9})(?:,)?(?: (\d{4}))?$", re.I)
_AMT_TOKEN = re.compile(
    r"^-?(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d+[.,]\d{1,2})$",
    re.ASCII,
)
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")

__all__ = ["parse_transactions"]

# ---------------------------------------------------------------------------
# Alias table (optional)
# ---------------------------------------------------------------------------
_ALIAS_PATH = Path(__file__).resolve().parents[2] / "rules" / "description_aliases.yml"
if _ALIAS_PATH.is_file():
    with open(_ALIAS_PATH, "r", encoding="utf-8") as _f:
        _ALIASES: dict[str, str] = yaml.safe_load(_f) or {}
    _ALIASES = {k.lower(): v for k, v in _ALIASES.items()}
else:
    _ALIASES = {}


def _tokenise(s: str) -> List[str]:
    toks = re.split(r"\s+", s.strip())
    return [t.strip() for t in toks if t and t not in {"-", "–"}]


def _clean_number(tok: str) -> float:
    tok = tok.rstrip(",.;").replace(" ", "").replace(",", "")
    if tok.count(".") > 1:  # dots as thousand sep
        parts = tok.split(".")
        tok = "".join(parts[:-1]) + "." + parts[-1]
    return float(tok)


def _merge_lines(df_grp: pd.DataFrame) -> str:
    df_grp = df_grp.copy()
    df_grp["x"] = df_grp["bbox"].apply(lambda b: (b[0][0] + b[3][0]) / 2)
    return " ".join(df_grp.sort_values("x")["text"].tolist()).strip()


def _apply_alias(desc: str) -> str:
    key = desc.lower().strip()
    best = None
    for k in _ALIASES:
        if k in key and (best is None or len(k) > len(best)):
            best = k
    return _ALIASES[best] if best else desc


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_transactions(ocr_csv: Path | str, y_threshold: float = 8.0) -> pd.DataFrame:
    ocr_df = pd.read_csv(ocr_csv)
    if ocr_df["bbox"].dtype == object:
        ocr_df["bbox"] = ocr_df["bbox"].apply(ast.literal_eval)

    # derive y-centre and sort
    ocr_df["y"] = ocr_df["bbox"].apply(lambda b: (b[0][1] + b[1][1]) / 2)
    ocr_df = ocr_df.sort_values(["page", "y"]).reset_index(drop=True)

    records: list[dict[str, Any]] = []

    for _, page_df in ocr_df.groupby("page"):
        # cluster by y
        clusters: list[list[int]] = []
        bucket: list[int] = []
        last_y = None
        for idx, row in page_df.iterrows():
            if last_y is None or abs(row["y"] - last_y) <= y_threshold:
                bucket.append(idx)
            else:
                clusters.append(bucket)
                bucket = [idx]
            last_y = row["y"]
        if bucket:
            clusters.append(bucket)

        current: dict[str, Any] | None = None

        YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

        def _ensure_year(date_str: str, desc_parts: list[str]) -> str:
            """Append the first 4-digit year found in *desc_parts* if *date_str* doesn't have one."""
            if re.search(r"\d{4}$", date_str):
                return date_str  # already has year
            for seg in desc_parts:
                m = YEAR_RE.search(seg)
                if m:
                    return f"{date_str} {m.group(0)}"
            return date_str

        def flush():
            if current and current.get("amount") is not None:
                date_val = _ensure_year(current["date"], current["desc"])
                records.append({
                    "date": date_val,
                    "description": _apply_alias(" ".join(current["desc"])).strip(),
                    "currency": current.get("currency", ""),
                    "amount": current["amount"],
                    "balance": current.get("balance"),
                })

        for cl in clusters:
            merged = _merge_lines(page_df.loc[cl])
            tokens = _tokenise(merged)
            if not tokens:
                continue

            # detect date (2- or 3-token)
            date_tok = ""
            for i in range(len(tokens) - 1):
                cand2 = tokens[i] + " " + tokens[i + 1]
                if _DATE_RE.match(cand2):
                    date_tok = cand2.rstrip(',')
                    break
                if i + 2 < len(tokens):
                    cand3 = cand2 + " " + tokens[i + 2]
                    if _DATE_RE.match(cand3):
                        date_tok = cand3.rstrip(',')
                        break
            has_date = bool(date_tok)

            # Pick numeric tokens that look like monetary values; explicitly
            # ignore 4-digit *year* numbers to avoid treating them as amounts.
            num_idx = [
                i
                for i, t in enumerate(tokens)
                if _AMT_TOKEN.match(t) and not 1900 <= _safe_int(t) <= 2100
            ]

            if has_date:
                flush()
                current = {
                    "date": date_tok,
                    "desc": [merged],
                    "nums": [tokens[i] for i in num_idx],
                }
                cur_tok = next((t for t in tokens if _CURRENCY_RE.match(t)), "")
                if cur_tok:
                    current["currency"] = cur_tok
            elif current is not None:
                current["desc"].append(merged)
                current["nums"].extend([tokens[i] for i in num_idx])

            # resolution once ≥2 numeric tokens
            if current and len(current["nums"]) >= 2:
                try:
                    bal = _clean_number(current["nums"][-1])
                    amt = None
                    for tok in reversed(current["nums"][:-1]):
                        val = _clean_number(tok)
                        if abs(val) < 1.5 * abs(bal):
                            amt = val
                            break
                    if amt is None:
                        amt = _clean_number(current["nums"][-2])
                    current["amount"] = amt
                    current["balance"] = bal
                    flush()
                    current = None
                except ValueError:
                    pass

    tx_df = pd.DataFrame.from_records(records).drop_duplicates(subset=["date", "amount", "balance"], keep="first")
    return tx_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(tok: str) -> int | float:
    """Return *tok* as int if lossless, otherwise float('inf')."""
    try:
        if tok.isdigit():
            return int(tok)
    except Exception:  # pragma: no cover – ultra-defensive
        pass
    return float("inf") 