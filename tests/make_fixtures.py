"""Generate the canonical invoice fixtures used across all three test levels.

    python tests/make_fixtures.py

Produces (in tests/fixtures/):
  purchase_digital.pdf     digital PDF, company is the BUYER  → purchase
  sales_digital.pdf        digital PDF, company is the SELLER → sales
  interstate_igst.pdf      inter-state invoice using IGST
  unknown_party.pdf        supplier GSTIN not in the party master
  unknown_items.pdf        line items not in the product master
  clean_scan.png           rasterised purchase invoice (clean 200-DPI scan)
  low_quality_photo.png    same, downscaled + blurred + rotated 3°

Company GSTIN in the fixtures is 24AAACT2727Q1ZW (Gujarat, state code 24) —
set the same value in Company Configuration to exercise direction detection.
"""
import pathlib
import subprocess
import sys

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

COMPANY = dict(name="Jayhind Traders Pvt Ltd", gstin="24AAACT2727Q1ZW", state="Gujarat",
               address="14 Ashram Road, Ahmedabad, Gujarat 380009")
SUPPLIER = dict(name="Shree Steel Supplies", gstin="24AABCS1429B1ZX", state="Gujarat",
                address="Plot 22, GIDC Estate, Vadodara, Gujarat 390010")
CUSTOMER = dict(name="Kiran Enterprises", gstin="24AADCK4567L1Z9", state="Gujarat",
                address="9 MG Road, Surat, Gujarat 395003")
MAHARASHTRA = dict(name="Deccan Metals Ltd", gstin="27AAGCD8899M1Z4", state="Maharashtra",
                   address="88 Andheri East, Mumbai, Maharashtra 400069")
UNKNOWN = dict(name="Nova Packaging LLP", gstin="24AAJCN7788K1ZQ", state="Gujarat",
               address="Survey 41, Sanand, Gujarat 382110")

STEEL_LINES = [
    ("MS Steel Rod 12mm", "7214", 100, "PCS", 250.00, 18),
    ("Galvanised Sheet 2mm", "7210", 40, "PCS", 780.00, 18),
]
UNKNOWN_LINES = [
    ("Corrugated Carton 18x12x10", "4819", 500, "PCS", 34.50, 12),
    ("Bubble Wrap Roll 1m x 100m", "3923", 15, "ROL", 890.00, 18),
]


def _money(value: float) -> str:
    return f"{value:,.2f}"


def _build_html(seller, buyer, number, date, lines, interstate=False):
    rows, taxable_total, tax_total = [], 0.0, 0.0
    for desc, hsn, qty, unit, rate, gst in lines:
        taxable = qty * rate
        tax = taxable * gst / 100
        taxable_total += taxable
        tax_total += tax
        rows.append(
            f"<tr><td>{desc}</td><td>{hsn}</td><td class='r'>{qty}</td><td>{unit}</td>"
            f"<td class='r'>{_money(rate)}</td><td class='r'>{_money(taxable)}</td>"
            f"<td class='r'>{gst}%</td><td class='r'>{_money(tax)}</td></tr>"
        )

    grand = taxable_total + tax_total
    round_off = round(grand) - grand
    grand_rounded = grand + round_off

    if interstate:
        tax_block = f"<tr><td>IGST</td><td class='r'>{_money(tax_total)}</td></tr>"
    else:
        half = tax_total / 2
        tax_block = (
            f"<tr><td>CGST</td><td class='r'>{_money(half)}</td></tr>"
            f"<tr><td>SGST</td><td class='r'>{_money(half)}</td></tr>"
        )

    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
      body {{ font-family: DejaVu Sans, Arial, sans-serif; font-size: 11px; margin: 28px; }}
      h1 {{ font-size: 17px; letter-spacing: 2px; text-align: center; margin: 0 0 14px; }}
      .party {{ width: 100%; margin-bottom: 12px; }}
      .party td {{ vertical-align: top; width: 50%; }}
      table.items {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
      table.items th, table.items td {{ border: 1px solid #333; padding: 4px 6px; }}
      table.items th {{ background: #eee; }}
      .r {{ text-align: right; }}
      .totals {{ margin-top: 10px; width: 45%; margin-left: auto; border-collapse: collapse; }}
      .totals td {{ border: 1px solid #333; padding: 3px 6px; }}
      .grand td {{ font-weight: bold; }}
    </style></head><body>
      <h1>TAX INVOICE</h1>
      <table class="party"><tr>
        <td><b>{seller['name']}</b><br>{seller['address']}<br>
            GSTIN: {seller['gstin']}<br>State: {seller['state']}</td>
        <td><b>Bill To: {buyer['name']}</b><br>{buyer['address']}<br>
            GSTIN: {buyer['gstin']}<br>State: {buyer['state']}</td>
      </tr></table>
      <div>Invoice No: {number} &nbsp;&nbsp;&nbsp; Invoice Date: {date}</div>
      <table class="items">
        <tr><th>Description</th><th>HSN</th><th>Qty</th><th>Unit</th>
            <th>Rate</th><th>Taxable</th><th>GST</th><th>Tax</th></tr>
        {''.join(rows)}
      </table>
      <table class="totals">
        <tr><td>Taxable Value</td><td class="r">{_money(taxable_total)}</td></tr>
        {tax_block}
        <tr><td>Round Off</td><td class="r">{round_off:.2f}</td></tr>
        <tr class="grand"><td>Grand Total</td><td class="r">{_money(grand_rounded)}</td></tr>
      </table>
    </body></html>"""


def _write_pdf(html: str, path: pathlib.Path) -> None:
    """Render HTML → PDF. Prefers weasyprint; falls back to a plain-text PDF."""
    try:
        from weasyprint import HTML

        HTML(string=html).write_pdf(str(path))
        return
    except Exception:
        pass
    # Fallback: fpdf2 (text-only, still a real digital text layer).
    try:
        import re

        from fpdf import FPDF

        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=9)
        pdf.multi_cell(0, 5, text)
        pdf.output(str(path))
        return
    except Exception as exc:
        sys.exit(f"Install weasyprint or fpdf2 to build PDF fixtures ({exc})")


def _rasterise(pdf_path: pathlib.Path, png_path: pathlib.Path, degrade: bool = False) -> None:
    import fitz
    from PIL import Image, ImageFilter

    with fitz.open(pdf_path) as doc:
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72))
        pix.save(png_path)

    if degrade:  # simulate a phone photo: smaller, blurred, slightly rotated
        with Image.open(png_path) as img:
            img = img.convert("RGB")
            img = img.resize((int(img.width * 0.55), int(img.height * 0.55)), Image.LANCZOS)
            img = img.filter(ImageFilter.GaussianBlur(0.8))
            img = img.rotate(3, expand=True, fillcolor=(255, 255, 255))
            img.save(png_path, quality=60)


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    specs = [
        ("purchase_digital.pdf", SUPPLIER, COMPANY, "SS/2026/0412", "05-07-2026", STEEL_LINES, False),
        ("sales_digital.pdf", COMPANY, CUSTOMER, "JT/2026/1188", "06-07-2026", STEEL_LINES, False),
        ("interstate_igst.pdf", MAHARASHTRA, COMPANY, "DM-2026-771", "04-07-2026", STEEL_LINES, True),
        ("unknown_party.pdf", UNKNOWN, COMPANY, "NP/26-27/019", "07-07-2026", STEEL_LINES, False),
        ("unknown_items.pdf", SUPPLIER, COMPANY, "SS/2026/0433", "07-07-2026", UNKNOWN_LINES, False),
    ]
    for name, seller, buyer, number, date, lines, interstate in specs:
        path = FIXTURES / name
        _write_pdf(_build_html(seller, buyer, number, date, lines, interstate), path)
        print(f"  wrote {path.name}")

    try:
        _rasterise(FIXTURES / "purchase_digital.pdf", FIXTURES / "clean_scan.png")
        print("  wrote clean_scan.png")
        _rasterise(FIXTURES / "purchase_digital.pdf", FIXTURES / "low_quality_photo.png", degrade=True)
        print("  wrote low_quality_photo.png")
    except Exception as exc:
        print(f"  (skipped image fixtures: {exc})")

    print(f"Fixtures written to {FIXTURES}")


if __name__ == "__main__":
    main()
