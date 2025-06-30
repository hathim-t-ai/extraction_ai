import pandas as pd
from pathlib import Path
import sys, os

# Ensure src is on PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pdf_parser.postprocess.row_parser import parse_transactions

# List of test cases: (raw_ocr_csv, ground_truth_csv)
TEST_CASES = [
    (
        Path("Outputs/YourBank-Bank Statement Example Final_paddleocr.csv"),
        Path("data/annotations/YourBank-Bank Statement Example Final_ground_truth.csv"),
    ),
    (
        Path("Outputs/ENBD-statements-pdf_paddleocr.csv"),
        Path("data/annotations/ENBD-statements-pdf_ground_truth.csv"),
    ),
]


def _recall(gt: pd.DataFrame, pred: pd.DataFrame) -> float:
    """Compute recall by (date, amount) exact match."""
    common = (
        gt.assign(_key=lambda d: d["date"].astype(str).str.strip())
        .merge(
            pred.assign(_key=lambda d: d["date"].astype(str).str.strip()),
            on=["_key", "amount"],
            how="inner",
            suffixes=("_gt", "_pred"),
        )
    )
    return len(common) / len(gt)


def _precision(gt: pd.DataFrame, pred: pd.DataFrame) -> float:
    common = (
        pred.assign(_key=lambda d: d["date"].astype(str).str.strip())
        .merge(
            gt.assign(_key=lambda d: d["date"].astype(str).str.strip()),
            on=["_key", "amount"],
            how="inner",
            suffixes=("_pred", "_gt"),
        )
    )
    return len(common) / len(pred)


import pytest

@pytest.mark.parametrize("ocr_csv,gt_csv", TEST_CASES)
def test_row_parser_recall_precision(ocr_csv: Path, gt_csv: Path):
    assert ocr_csv.is_file(), f"Missing OCR CSV {ocr_csv}"
    assert gt_csv.is_file(), f"Missing GT CSV {gt_csv}"

    parsed_df = parse_transactions(ocr_csv)
    gt_df = pd.read_csv(gt_csv)

    rec = _recall(gt_df, parsed_df)
    prec = _precision(gt_df, parsed_df)

    # expectations: at least 0.9 recall & 0.9 precision
    assert rec >= 0.9, f"Recall too low: {rec:.2%} for {ocr_csv.name}"
    assert prec >= 0.9, f"Precision too low: {prec:.2%} for {ocr_csv.name}" 