from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys


def convert_csv_to_json(csv_path: Path, json_path: Path) -> None:
    """Convert a single CSV file to the training-ready JSON format.

    The JSON will have the shape::
        {
            "transactions": [ {<row1>}, {<row2>}, ... ]
        }

    Column names from the CSV header row are preserved exactly, with **no**
    normalisation.
    """
    with csv_path.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    json_data = {"transactions": rows}

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(json_data, fp, ensure_ascii=False, indent=2)



def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    csv_dir = repo_root / "data" / "For_Training" / "BS_CSV"
    json_dir = repo_root / "data" / "For_Training" / "BS_JSON"

    if not csv_dir.exists():
        print(f"CSV directory not found: {csv_dir}", file=sys.stderr)
        sys.exit(1)

    csv_files = sorted(p for p in csv_dir.glob("*.csv") if p.is_file())
    if not csv_files:
        print("No CSV files found – nothing to convert.")
        return

    for csv_path in csv_files:
        json_path = json_dir / f"{csv_path.stem}.json"
        convert_csv_to_json(csv_path, json_path)
        print(f"✓ {csv_path.name} → {json_path.relative_to(repo_root)} ({json_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main() 