"""Score the structuring engine field-by-field against the golden layout set.

    python tests/accuracy_report.py

Reads every fixture named in `fixtures/layout_golden.json`, runs the real
extraction + structuring pipeline on it, and prints a pass/fail scorecard with an
overall percentage. This is the objective measure of "is the OCR getting the
proper name and all things" that the accuracy work is judged by (baseline in
ACCURACY_BASELINE.md).
"""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.extractor import extract  # noqa: E402
from app.structuring import parse_invoice  # noqa: E402

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _norm(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _name_ok(got: str, exp: str) -> bool:
    g, e = _norm(got), _norm(exp)
    if e == "":            # e.g. banner: must NOT invent a name
        return g == ""
    if not g:
        return False
    # A polluted name ("XYZ Retail LLP Date 10-07-2026") must NOT pass — allow
    # only a few extra characters around the expected name.
    return g == e or g in e or (e in g and len(g) <= len(e) + 6)


def _num_ok(got, exp) -> bool:
    if exp is None:
        return True
    return abs((got or 0) - exp) <= 1


def _mark(passed: bool) -> str:
    return "\033[32m✓\033[0m" if passed else "\033[31m✗\033[0m"


def main() -> int:
    golden = json.loads((FIXTURES / "layout_golden.json").read_text())
    total = ok = 0
    for g in golden:
        data = (FIXTURES / g["file"]).read_bytes()
        ocr = extract(data, "application/pdf")
        inv = parse_invoice(ocr)
        seller, buyer = inv["seller"], inv["buyer"]
        checks = [
            ("seller", _name_ok(seller["name"], g["seller_name"])),
            ("buyer", _name_ok(buyer["name"], g["buyer_name"])),
            ("inv#", _norm(inv["invoice"]["number"]) == _norm(g["invoice_no"]) if g["invoice_no"] else True),
            ("lines", len(inv["lineItems"]) == g["line_count"]),
            ("taxable", _num_ok(inv["totals"]["taxableTotal"], g.get("taxable"))),
            ("grand", _num_ok(inv["totals"]["grandTotal"], g.get("grand"))),
        ]
        # Stricter fields scored only where the golden pins them.
        if g.get("seller_gstin"):
            checks.append(("s.gst", seller["gstin"] == g["seller_gstin"]))
        if g.get("buyer_gstin"):
            checks.append(("b.gst", buyer["gstin"] == g["buyer_gstin"]))
        if g.get("invoice_date"):
            checks.append(("date", inv["invoice"]["date"] == g["invoice_date"]))
        for _, passed in checks:
            total += 1
            ok += passed
        detail = " ".join(f"{n}:{_mark(p)}" for n, p in checks)
        print(f"{g['file']:<24} {detail}   seller={seller['name']!r} buyer={buyer['name']!r} "
              f"lines={len(inv['lineItems'])} grand={inv['totals']['grandTotal']}")

    pct = 100 * ok / total if total else 0
    print(f"\nAFTER (geometry-first Python engine): {ok}/{total} field checks pass ({pct:.0f}%)")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
