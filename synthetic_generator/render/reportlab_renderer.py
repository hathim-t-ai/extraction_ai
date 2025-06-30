from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import re

from ..config import SyntheticConfig
from ..data_models import Statement


# Simple constants for table layout
LEFT_MARGIN = 20 * mm
TOP_MARGIN = 25 * mm
LINE_HEIGHT = 6 * mm
COLS = [LEFT_MARGIN, 50 * mm, 120 * mm, 150 * mm, 180 * mm]


def render_pdf(statement: Statement, cfg: SyntheticConfig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4

    # Determine bank name from layout key
    layout_key = cfg.layouts[0]
    bank_match = re.match(r"([a-z]+)_", layout_key)
    bank_name = bank_match.group(1).title() + " Bank" if bank_match else "Generic Bank"

    # Header with optional color for Axis Bank
    c.setFont("Helvetica-Bold", 14)
    if "axisbank" in layout_key:
        c.setFillColor(colors.HexColor("#ef4e23"))
    c.drawString(LEFT_MARGIN, height - TOP_MARGIN, f"{bank_name} – {statement.currency} Account")
    c.setFillColor(colors.black)

    c.setFont("Helvetica", 8)
    c.drawString(LEFT_MARGIN, height - TOP_MARGIN - 10, f"Statement Period: {statement.periodStart} – {statement.periodEnd}")
    c.drawString(LEFT_MARGIN, height - TOP_MARGIN - 20, f"Account #: {statement.accountNumber}")

    # Table header
    y = height - TOP_MARGIN - 40
    headers = ["Date", "Description", "Debit", "Credit", "Balance"]
    for i, h in enumerate(headers):
        c.drawString(COLS[i], y, h)
    y -= LINE_HEIGHT

    c.setFont("Helvetica", 7)
    for txn in statement.transactions:
        if y < 40 * mm:
            c.showPage()
            y = height - TOP_MARGIN
        c.drawString(COLS[0], y, txn.date.strftime("%d %b %y"))
        c.drawString(COLS[1], y, txn.description.split("\n")[0][:40])
        c.drawRightString(COLS[2] + 20 * mm, y, str(txn.debit or ""))
        c.drawRightString(COLS[3] + 20 * mm, y, str(txn.credit or ""))
        c.drawRightString(COLS[4] + 20 * mm, y, str(txn.balance))
        y -= LINE_HEIGHT

    # Footer
    c.setFont("Helvetica", 6)
    c.drawString(LEFT_MARGIN, 15 * mm, "This is a system-generated statement. For queries contact support@bank.com")

    c.showPage()
    c.save() 