"""Generate the LAYOUT-accuracy fixture set + golden expectations.

    python tests/make_layout_fixtures.py

These are distinct from make_fixtures.py (which mirrors the seeded dev DB for the
Node matching harness). This set reproduces the *layout families* a user hits in
the wild — the ones the old text-line parser read wrongly — so the structuring
engine's accuracy can be scored field-by-field against `layout_golden.json` by
`tests/accuracy_report.py`.

Each family below is modelled on a real sample invoice: three-column header,
From:/To:, banner with only a Bill-To block, sidebar menu, timeline strip,
side-by-side with GSTINs, literal-HTML leak, well-formed GST, POS receipt and a
minimal corporate grid. Terse column headers ("Item", "Grand", "Taxable", "Amt")
are used on purpose — they are exactly what broke the narrow keyword lists.
"""
import html as _html
import json
import pathlib

from weasyprint import HTML

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

# ── Shared item sets ─────────────────────────────────────────────────────────
# "Acme/Metro" set: terse integer money, terse headers (Item/Grand/Taxable).
ACME_ITEMS = [
    ("Industrial Pump", 2, 15000, 30000),
    ("PVC Pipe", 25, 480, 12000),
    ("Valve", 10, 950, 9500),
]
ACME_TOTALS = dict(taxable=51500, cgst=4635, sgst=4635, grand=60770)

# "Prime/Blue Star" set: HSN column, decimal money.
PRIME_ITEMS = [
    ("Industrial Motor", "8501", 2, "12,500.00", "25,000.00"),
    ("Control Panel", "8537", 1, "18,000.00", "18,000.00"),
    ("Copper Cable", "8544", 50, "240.00", "12,000.00"),
]
PRIME_TOTALS = dict(taxable="55,000.00", cgst="4,950.00", sgst="4,950.00", grand="64,900.00")

BASE_CSS = """
  body { font-family: DejaVu Sans, Arial, sans-serif; font-size: 11px; margin: 28px; }
  h1 { font-size: 18px; text-align: center; margin: 0 0 14px; }
  table { border-collapse: collapse; }
  .items { width: 100%; margin-top: 8px; }
  .items th, .items td { border: 1px solid #333; padding: 4px 6px; }
  .items th { background: #eee; }
  .r { text-align: right; }
  .party { width: 100%; margin-bottom: 12px; }
  .party td { vertical-align: top; width: 50%; padding: 4px; }
  td.g { border: 1px solid #999; padding: 4px 8px; }
"""


def _doc(body: str, css: str = "") -> str:
    return f'<!doctype html><html><head><meta charset="utf-8"><style>{BASE_CSS}{css}</style></head><body>{body}</body></html>'


def _acme_table() -> str:
    rows = "".join(
        f"<tr><td>{d}</td><td class=r>{q}</td><td class=r>{r}</td><td class=r>18%</td><td class=r>{a}</td></tr>"
        for d, q, r, a in ACME_ITEMS
    )
    t = ACME_TOTALS
    return f"""<table class=items>
      <tr><th>Item</th><th>Qty</th><th>Rate</th><th>GST</th><th>Amount</th></tr>
      {rows}
      <tr><td colspan=4 class=r>Taxable</td><td class=r>{t['taxable']}</td></tr>
      <tr><td colspan=4 class=r>CGST</td><td class=r>{t['cgst']}</td></tr>
      <tr><td colspan=4 class=r>SGST</td><td class=r>{t['sgst']}</td></tr>
      <tr><td colspan=4 class=r>Grand</td><td class=r>{t['grand']}</td></tr>
    </table>
    <p>Bank: HDFC Bank | IFSC: HDFC0001234 | Authorized Signatory</p>"""


def _prime_table() -> str:
    rows = "".join(
        f"<tr><td>{i}</td><td>{d}</td><td>{h}</td><td class=r>{q}</td><td class=r>{r}</td><td class=r>18%</td><td class=r>{a}</td></tr>"
        for i, (d, h, q, r, a) in enumerate(PRIME_ITEMS, start=1)
    )
    t = PRIME_TOTALS
    return f"""<table class=items>
      <tr><th>Sr</th><th>Description</th><th>HSN</th><th>Qty</th><th>Rate</th><th>GST</th><th>Amount</th></tr>
      {rows}
      <tr><td colspan=6 class=r>Taxable</td><td class=r>{t['taxable']}</td></tr>
      <tr><td colspan=6 class=r>CGST 9%</td><td class=r>{t['cgst']}</td></tr>
      <tr><td colspan=6 class=r>SGST 9%</td><td class=r>{t['sgst']}</td></tr>
      <tr><td colspan=6 class=r>Grand Total</td><td class=r>{t['grand']}</td></tr>
    </table>
    <p>Bank: HDFC Bank | A/C 50100123456789 | IFSC HDFC0001234</p>
    <p>Authorized Signatory _________________________</p>"""


# ── Layout builders ──────────────────────────────────────────────────────────

def build() -> list:
    """Return [(filename, html, golden_dict)] for every layout family."""
    specs = []

    # 1. Three-column header: labels row, values row (Modern).
    specs.append(("layout_3col.pdf", _doc(f"""<h1>Invoice</h1>
      <table class=party><tr><td>Supplier</td><td>Invoice</td><td>Buyer</td></tr>
      <tr><td>Acme Pumps Pvt Ltd</td><td>INV2208</td><td>Metro Infra Ltd</td></tr></table>
      {_acme_table()}"""),
      dict(seller_name="Acme Pumps Pvt Ltd", buyer_name="Metro Infra Ltd", invoice_no="INV2208",
           line_count=3, taxable=51500, grand=60770)))

    # 2. Card sections: banner + label/value column pairs (Card).
    specs.append(("layout_card.pdf", _doc(f"""
      <div style="background:navy;color:white;text-align:center;padding:4px">CARD SECTIONS</div>
      <table class=party><tr><td>Supplier</td><td>Acme Pumps</td></tr>
      <tr><td>Invoice</td><td>INV2202</td></tr>
      <tr><td>Buyer</td><td>Metro Infra</td></tr>
      <tr><td>Due</td><td>09-08-26</td></tr></table>{_acme_table()}"""),
      dict(seller_name="Acme Pumps", buyer_name="Metro Infra", invoice_no="INV2202",
           line_count=3, taxable=51500, grand=60770)))

    # 3. From:/To: inline (Compact).
    specs.append(("layout_fromto.pdf", _doc(f"""<h1>Invoice INV2207</h1>
      <p>From: Acme Pumps Pvt Ltd</p><p>To: Metro Infra Ltd</p>{_acme_table()}"""),
      dict(seller_name="Acme Pumps Pvt Ltd", buyer_name="Metro Infra Ltd", invoice_no="INV2207",
           line_count=3, taxable=51500, grand=60770)))

    # 4. Banner + only a Bill To / Ship To block, no supplier block (Banner).
    specs.append(("layout_banner.pdf", _doc(f"""
      <div style="background:green;color:white;text-align:center;padding:6px">TOP BANNER</div>
      <table class=party><tr><td>Bill To</td><td>Ship To</td></tr>
      <tr><td>Metro Infra</td><td>Metro Warehouse</td></tr></table>{_acme_table()}"""),
      dict(seller_name="", buyer_name="Metro Infra", invoice_no="",
           line_count=3, taxable=51500, grand=60770)))

    # 5. Sidebar menu column + inline labelled parties (Sidebar).
    specs.append(("layout_sidebar.pdf", _doc(f"""
      <table class=party style="width:100%"><tr>
      <td style="width:20%">MENU<br>Supplier<br>Buyer<br>Bank<br>Terms</td>
      <td>INVOICE INV-2201<br>Supplier: Acme Pumps Pvt Ltd<br>Buyer: Metro Infra Ltd<br>GSTIN:24AACCA1234F1Z5</td>
      </tr></table>{_acme_table()}"""),
      dict(seller_name="Acme Pumps Pvt Ltd", buyer_name="Metro Infra Ltd", invoice_no="INV-2201",
           line_count=3, taxable=51500, grand=60770)))

    # 6. Timeline strip + inline pipe-separated parties (Timeline).
    specs.append(("layout_timeline.pdf", _doc(f"""<h1>Quotation &#8594; PO &#8594; Dispatch &#8594; Invoice</h1>
      <p>Invoice #: INV2203</p><p>Supplier: Acme Pumps | Buyer: Metro Infra</p>{_acme_table()}"""),
      dict(seller_name="Acme Pumps", buyer_name="Metro Infra", invoice_no="INV2203",
           line_count=3, taxable=51500, grand=60770)))

    # 7. Side by side with GSTINs (Newspaper).
    specs.append(("layout_sidebyside.pdf", _doc(f"""<h1>NEWSPAPER STYLE</h1>
      <table class=party><tr><td>SUPPLIER</td><td>BUYER</td></tr>
      <tr><td>Acme Pumps<br>GSTIN:24AACCA1234F1Z5<br>21 GIDC Ahmedabad</td>
      <td>Metro Infra<br>GSTIN:24AACCM9999L1Z2<br>Mumbai</td></tr></table>{_acme_table()}"""),
      dict(seller_name="Acme Pumps", buyer_name="Metro Infra", invoice_no="",
           seller_gstin="24AACCA1234F1Z5", buyer_gstin="24AACCM9999L1Z2",
           line_count=3, taxable=51500, grand=60770)))

    # 8. Literal-HTML leak: <b>/<br/> appear as visible text (HeaderBlocks/Sectioned).
    leak = _html.escape("<b>Supplier</b><br/>Acme Industrial Pvt Ltd<br/>21 GIDC Estate<br/>Ahmedabad<br/>GSTIN:24AACCA1234F1Z5")
    leak2 = _html.escape("<b>Buyer</b><br/>Blue Star Engineering<br/>88 Ring Road<br/>Surat<br/>GSTIN:24AACCB5678L1Z8")
    specs.append(("layout_htmlleak.pdf", _doc(f"""<h1>Header Block Layout</h1>
      <table class=party><tr><td>FROM</td><td>BILL TO</td></tr>
      <tr><td>{leak}</td><td>{leak2}</td></tr></table>{_prime_table()}"""),
      dict(seller_name="Acme Industrial Pvt Ltd", buyer_name="Blue Star Engineering", invoice_no="",
           seller_gstin="24AACCA1234F1Z5", buyer_gstin="24AACCB5678L1Z8",
           line_count=3, taxable=55000, grand=64900)))

    # 9. Well-formed GST invoice (Wholesale/Purchase — the "good" layout).
    specs.append(("layout_wellformed.pdf", _doc(f"""<h1>WHOLESALE GST INVOICE</h1>
      <table class=party><tr><td>Supplier</td><td>Buyer</td></tr>
      <tr><td><b>Prime Industrial Supplies Pvt. Ltd.</b><br>Unit-7, Naroda GIDC, Ahmedabad, Gujarat - 382330<br>
      GSTIN: 24AABCP3456M1Z7<br>PAN: AABCP3456M<br>State: Gujarat (24)</td>
      <td><b>Zenith Retail LLP</b><br>45 Ring Road, Surat, Gujarat<br>GSTIN: 24AAAFZ5678L1Z8<br>State: Gujarat (24)</td>
      </tr></table>
      <p>Invoice No INV-2026-1101 &nbsp; Invoice Date 10-07-2026 &nbsp; PO No PO-5601</p>{_prime_table()}"""),
      dict(seller_name="Prime Industrial Supplies Pvt. Ltd.", buyer_name="Zenith Retail LLP",
           invoice_no="INV-2026-1101", seller_gstin="24AABCP3456M1Z7", buyer_gstin="24AAAFZ5678L1Z8",
           line_count=3, taxable=55000, grand=64900)))

    # 10. POS thermal receipt: seller header only, "Item/Amt", "Total" (SUPER MART).
    specs.append(("layout_pos.pdf", _doc("""
      <h1>SUPER MART</h1><p>GSTIN:24ABCDE1234F1Z1<br>Receipt #45821</p>
      <table class=items style="width:60%">
      <tr><th>Item</th><th>Amt</th></tr>
      <tr><td>Milk</td><td class=r>62</td></tr>
      <tr><td>Bread</td><td class=r>45</td></tr>
      <tr><td>Eggs</td><td class=r>78</td></tr>
      <tr><td>Total</td><td class=r>185</td></tr></table>""",
      css="@page { size: 80mm 200mm; }"),
      dict(seller_name="SUPER MART", buyer_name="", invoice_no="45821",
           seller_gstin="24ABCDE1234F1Z1", line_count=3, taxable=None, grand=185)))

    # 11. Minimal corporate grid: label/value pairs across, "Total" footer (Corporate).
    specs.append(("layout_corporate.pdf", _doc(f"""<h1>CORPORATE TAX INVOICE</h1>
      <table class=party><tr><td>Supplier</td><td>ABC Manufacturing Pvt Ltd</td><td>Invoice</td><td>INV-1001</td></tr>
      <tr><td>Buyer</td><td>XYZ Retail LLP</td><td>Date</td><td>10-07-2026</td></tr>
      <tr><td>GSTIN</td><td>24ABCDE1234F1Z1</td><td>PO</td><td>PO-8891</td></tr></table>
      <table class=items>
      <tr><th>Item</th><th>Qty</th><th>Rate</th><th>GST</th><th>Amount</th></tr>
      <tr><td>Monitor</td><td class=r>2</td><td class=r>9200</td><td class=r>18%</td><td class=r>18400</td></tr>
      <tr><td>Keyboard</td><td class=r>5</td><td class=r>850</td><td class=r>18%</td><td class=r>4250</td></tr>
      <tr><td colspan=4 class=r>Total</td><td class=r>22650</td></tr></table>"""),
      dict(seller_name="ABC Manufacturing Pvt Ltd", buyer_name="XYZ Retail LLP", invoice_no="INV-1001",
           seller_gstin="24ABCDE1234F1Z1", line_count=2, taxable=None, grand=22650)))

    # 12. Letterhead over a lone Bill To / Ship To box (user sample invoice-new-1):
    # unlabelled full-width seller header, bordered info grid (Order No / Due Date /
    # Place of Supply / Payment Terms), CGST+SGST+Total item columns, truncated
    # decimals ("6,372.0"), amount-in-words, bank block.
    specs.append(("layout_letterhead.pdf", _doc(f"""<h1>TAX INVOICE</h1>
      <div><b>ABC Traders Pvt. Ltd.</b><br>GSTIN: 24ABCDE1234F1Z5<br>
      123 Business Park, Vadodara, Gujarat 390001<br>
      Phone: +91 98765 43210 | Email: sales@abctraders.com</div>
      <table style="width:88%;margin:10px auto">
      <tr><td class=g>Invoice No.</td><td class=g>INV-2026-0001</td><td class=g>Invoice Date</td><td class=g>12-Jul-2026</td></tr>
      <tr><td class=g>Order No.</td><td class=g>SO-2026-0101</td><td class=g>Due Date</td><td class=g>19-Jul-2026</td></tr>
      <tr><td class=g>Place of Supply</td><td class=g>Gujarat</td><td class=g>Payment Terms</td><td class=g>7 Days</td></tr></table>
      <table class=party><tr><th style="text-align:left;border:1px solid #999">Bill To</th>
      <th style="text-align:left;border:1px solid #999">Ship To</th></tr>
      <tr><td style="border:1px solid #999">XYZ Enterprises<br>GSTIN:24AACCX1234A1Z2<br>Vadodara, Gujarat</td>
      <td style="border:1px solid #999">XYZ Warehouse<br>Vadodara, Gujarat</td></tr></table>
      <table class=items>
      <tr><th>HSN</th><th>Product</th><th>Qty</th><th>Unit</th><th>Rate</th><th>Taxable</th><th>GST %</th><th>CGST</th><th>SGST</th><th>Total</th></tr>
      <tr><td>8471</td><td>Wireless Keyboard</td><td class=r>2</td><td>Nos</td><td class=r>1,500.00</td><td class=r>3,000.00</td><td class=r>18%</td><td class=r>270.00</td><td class=r>270.00</td><td class=r>3,540.0</td></tr>
      <tr><td>8473</td><td>Wireless Mouse</td><td class=r>3</td><td>Nos</td><td class=r>800.00</td><td class=r>2,400.00</td><td class=r>18%</td><td class=r>216.00</td><td class=r>216.00</td><td class=r>2,832.0</td></tr>
      <tr><td colspan=5></td><td class=r>Subtotal</td><td class=r>5,400.00</td><td></td><td></td><td></td></tr>
      <tr><td colspan=6></td><td class=r>CGST</td><td class=r>486.00</td><td></td><td></td></tr>
      <tr><td colspan=6></td><td class=r>SGST</td><td class=r>486.00</td><td></td><td></td></tr>
      <tr><td colspan=8></td><td class=r>Grand Total</td><td class=r>6,372.0</td></tr></table>
      <p><b>Amount in Words:</b> Indian Rupees Six Thousand Three Hundred Seventy-Two Only</p>
      <p><b>Bank Details:</b><br>Bank: HDFC Bank<br>A/C No: 12345678901234<br>IFSC: HDFC0001234</p>
      <p>Terms: 1. Goods once sold will not be taken back. 2. Interest @18% p.a. on overdue invoices.</p>"""),
      dict(seller_name="ABC Traders Pvt. Ltd.", buyer_name="XYZ Enterprises",
           invoice_no="INV-2026-0001", invoice_date="2026-07-12",
           seller_gstin="24ABCDE1234F1Z5", buyer_gstin="24AACCX1234A1Z2",
           line_count=2, taxable=5400, grand=6372)))

    return specs


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    golden = []
    for name, doc, expected in build():
        HTML(string=doc).write_pdf(str(FIXTURES / name))
        golden.append({"file": name, **expected})
        print(f"  wrote {name}")
    (FIXTURES / "layout_golden.json").write_text(json.dumps(golden, indent=2))
    print(f"  wrote layout_golden.json ({len(golden)} fixtures)")


if __name__ == "__main__":
    main()
