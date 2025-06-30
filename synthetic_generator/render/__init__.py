from __future__ import annotations

from pathlib import Path

from ..config import SyntheticConfig
from ..data_models import Statement


def render(statement: Statement, cfg: SyntheticConfig, output_path: Path) -> None:
    if cfg.renderer == "auto":
        # prefer HTML if template exists
        from pathlib import Path as _P
        tmpl = _P(__file__).parent / "templates" / f"{cfg.layouts[0]}.html"
        use_html = tmpl.exists()
        renderer_name = "html" if use_html else "reportlab"
    else:
        renderer_name = cfg.renderer

    if renderer_name == "reportlab":
        from .reportlab_renderer import render_pdf as _render
    else:
        try:
            from .html_renderer import render_pdf as _render  # noqa: WPS433
        except Exception as exc:  # pragma: no cover
            # Any failure (missing WeasyPrint native libs, etc.) falls back.
            print(
                f"[synthetic-generator] HTML renderer unavailable ({exc}). "
                "Falling back to ReportLab.",
            )
            from .reportlab_renderer import render_pdf as _render

    _render(statement, cfg, output_path) 