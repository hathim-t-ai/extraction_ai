from __future__ import annotations

"""Simple utility to analyse sample statement PNG files and create layout JSON descriptors.

Usage:
    rye run python tools/png_layout_ingest.py --src data/raw_pdfs/BankStatements --dest layouts

For now, detection is naive: we record image width/height and use generic column ratios.
This still allows renderer to size columns realistically, and we can refine later.
"""

import json
import re
from pathlib import Path

from PIL import Image
import typer

app = typer.Typer()


BANK_RE = re.compile(r"(?P<bank>[A-Za-z]+)_(?:Bank)?_?Type_(?P<type>[IVX]+)_")


def analyse_image(path: Path):
    img = Image.open(path)
    if img is None:
        raise ValueError(f"Failed to load {path}")
    w, h = img.size
    # baseline column ratios as percentages of width
    ratios = [0.05, 0.30, 0.65, 0.75, 0.85]
    cols = [int(r * w) for r in ratios]
    return {"width": w, "height": h, "columns": cols}


@app.command()
def main(
    src: Path = typer.Option(..., exists=True, help="Folder with statement PNG files"),
    dest: Path = typer.Option(Path("layouts"), help="Destination folder for JSON descriptors"),
):
    layouts: dict[str, dict] = {}
    for file in src.rglob("*.png"):
        m = BANK_RE.match(file.name)
        if not m:
            typer.echo(f"Skipping {file.name} (pattern mismatch)")
            continue
        layout_key = f"{m.group('bank').lower()}_type_{m.group('type').lower()}"
        if layout_key in layouts:
            continue  # already analysed first page
        metrics = analyse_image(file)
        layouts[layout_key] = metrics
        typer.echo(f"Analysed {layout_key}: {metrics}")

    dest.mkdir(parents=True, exist_ok=True)
    for key, metrics in layouts.items():
        out = dest / f"{key}.json"
        out.write_text(json.dumps(metrics, indent=2))
    typer.echo(f"Wrote {len(layouts)} layout descriptors to {dest}")


if __name__ == "__main__":
    app() 