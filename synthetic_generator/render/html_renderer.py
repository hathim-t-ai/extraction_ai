from pathlib import Path
import random
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from ..config import SyntheticConfig
from ..data_models import Statement

TEMPLATE_DIR = Path(__file__).parent / "templates"

env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html"]),
)

def _render_generic_html(statement: Statement, bank_name: str, schema: str) -> str:
    """Return an HTML string for schemas that don't rely on bank-specific templates."""
    # basic style with slight randomness
    font_family = random.choice(["Helvetica", "Arial", "Times", "Calibri", "Roboto"])
    header_color = random.choice(["#cc0000", "#ef4e23", "#006699", "#004080", "#222"])

    # Build columns per schema
    if schema == "four_no_balance":
        headers = ["Date", "Description", "Debit", "Credit"]
    else:  # fallback five_column
        headers = ["Date", "Description", "Debit", "Credit", "Balance"]

    def make_cell(value: str, is_num: bool = False) -> str:
        align = "right" if is_num else "left"
        return f'<td style="text-align:{align};">{value}</td>'

    rows_html = ""  # construct each <tr>
    for txn in statement.transactions:
        date_str = txn.date.strftime("%d-%m-%Y")
        desc = txn.description.replace("\n", "<br/>")
        cells = [make_cell(date_str), make_cell(desc)]
        # debit / credit
        cells.append(make_cell(txn.debit or "", is_num=True))
        cells.append(make_cell(txn.credit or "", is_num=True))
        if schema == "five_column":
            cells.append(make_cell(txn.balance, is_num=True))
        rows_html += "<tr>" + "".join(cells) + "</tr>\n"

    thead_html = "<tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr>"

    html = f"""
<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'/>
  <style>
    body {{ font-family: '{font_family}', sans-serif; margin: 40px; }}
    h1 {{ color:{header_color}; text-align:center; margin-bottom:8px; }}
    table {{ width:100%; border-collapse:collapse; font-size:9pt; }}
    th, td {{ border:1px solid #999; padding:4px; }}
    th {{ background:#eee; text-align:left; }}
  </style>
</head>
<body>
  <h1>{bank_name} – {statement.currency} Account</h1>
  <p>Statement Period: {statement.periodStart} – {statement.periodEnd}</p>
  <p>Account #: {statement.accountNumber}</p>

  <table>
    <thead>{thead_html}</thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</body>
</html>
"""
    return html

def render_pdf(statement: Statement, cfg: SyntheticConfig, output_path: Path) -> None:
    # Determine effective schema for this statement
    schema = cfg.schema
    if schema == "random":
        schema = random.choice(["five_column", "four_no_balance"])

    # If schema is five_column we try existing templates first
    use_generic = schema != "five_column"

    bank_key = cfg.layouts[0]
    bank_name = bank_key.replace("_", " ").title()

    if not use_generic:
        template_name = f"{bank_key}.html" if (TEMPLATE_DIR / f"{bank_key}.html").exists() else "default.html"
        template = env.get_template(template_name)
        html_str = template.render(statement=statement)
    else:
        html_str = _render_generic_html(statement, bank_name, schema)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_str).write_pdf(str(output_path)) 