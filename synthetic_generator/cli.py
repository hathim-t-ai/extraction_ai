from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional, List
import re

import typer
from rich.console import Console
from rich.progress import Progress

from .config import SyntheticConfig
from .ledger import build_statement
from .render import render
from .noise import apply_noise
from .utils import timestamped_dir, sha256sum

app = typer.Typer(help="Synthetic bank-statement generator")
console = Console()


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    config_path: Optional[Path] = typer.Option(None, help="Path to YAML config"),
    output_dir: Optional[Path] = typer.Option(None, help="Override output directory"),
    statements: Optional[int] = typer.Option(None, help="Number of statements to create"),
    layouts: Optional[str] = typer.Option(None, help="Comma-separated layout keys"),
    renderer: Optional[str] = typer.Option(None, help="Renderer: auto | reportlab | html"),
    schema: Optional[str] = typer.Option(None, help="Table schema: five_column | four_no_balance | random"),
):
    """Generate synthetic bank statements.

    You can supply options directly or via the **generate** sub-command, e.g.:
    
    `python -m synthetic_generator --statements 5`
    or
    `python -m synthetic_generator generate --statements 5`
    """
    # If a sub-command was invoked do nothing here.
    if ctx.invoked_subcommand is not None:
        return
    _run_generation(config_path, output_dir, statements, layouts, renderer, schema)


@app.command("generate")
def generate_cmd(
    config_path: Optional[Path] = typer.Option(None, help="Path to YAML config"),
    output_dir: Optional[Path] = typer.Option(None, help="Override output directory"),
    statements: Optional[int] = typer.Option(None, help="Number of statements to create"),
    layouts: Optional[str] = typer.Option(None, help="Comma-separated layout keys"),
    renderer: Optional[str] = typer.Option(None, help="Renderer: auto | reportlab | html"),
    schema: Optional[str] = typer.Option(None, help="Table schema: five_column | four_no_balance | random"),
):
    """Explicit sub-command wrapper around generator."""
    _run_generation(config_path, output_dir, statements, layouts, renderer, schema)


# internal helper shared by root & sub-command

def _run_generation(
    config_path: Optional[Path],
    output_dir: Optional[Path],
    statements: Optional[int],
    layouts: Optional[str],
    renderer: Optional[str],
    schema: Optional[str],
):
    cfg = SyntheticConfig.from_yaml(config_path) if config_path else SyntheticConfig()
    if statements:
        cfg.statements = statements
    if output_dir:
        cfg.output_dir = output_dir
    if layouts:
        cfg.layouts = [s.strip() for s in re.split(r",|\s+", layouts) if s.strip()]
    if renderer:
        cfg.renderer = renderer
    if schema:
        cfg.schema = schema
    generate_statements(cfg)


def generate_statements(cfg: SyntheticConfig) -> None:
    random.seed(cfg.seed)
    cfg.output_dir = timestamped_dir(cfg.output_dir)

    manifest = {}
    layouts_list = cfg.layouts.copy() if cfg.layouts else []
    with Progress() as progress:
        task = progress.add_task("Generating", total=cfg.statements)
        for idx in range(1, cfg.statements + 1):
            # Select layout round-robin if multiple specified
            if layouts_list:
                current = layouts_list[(idx - 1) % len(layouts_list)]
                cfg.layouts = [current]

            stmt = build_statement(cfg)
            pdf_path = cfg.output_dir / f"{idx:04d}.pdf"
            render(stmt, cfg, pdf_path)
            apply_noise(pdf_path, cfg)

            # Write JSON & CSV
            json_path = pdf_path.with_suffix(".json")
            csv_path = pdf_path.with_suffix(".csv")
            json_path.write_text(stmt.model_dump_json(by_alias=True, indent=2))
            csv_path.write_text(stmt.to_csv())

            manifest[str(pdf_path.name)] = {
                "sha256": sha256sum(pdf_path),
                "json": json_path.name,
                "csv": csv_path.name,
            }

            progress.update(task, advance=1)

    # Save manifest
    (cfg.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    console.print(f"[green]Generated batch at {cfg.output_dir}[/green]")


def _main():  # entry-point when `python -m synthetic_generator` executed
    app()


if __name__ == "__main__":
    _main() 