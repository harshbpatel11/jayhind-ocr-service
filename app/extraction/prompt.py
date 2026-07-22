"""Prompt construction + JSON schema for the extraction LLM.

The system prompt fixes the model's role; the user prompt carries the exact output
schema, the GST rules, and — crucially — the *grounding hints* read verbatim from
the page (GSTINs, invoice number/date, grand total). Grounding sharply reduces
hallucination: the model is told the real GSTINs on the document and instructed to
copy, not invent.

``build_json_schema`` returns a JSON Schema the llama.cpp backend can compile to a
GBNF grammar, so generation is *structurally* constrained to the contract — the
model literally cannot emit prose or the wrong keys.
"""

from __future__ import annotations

from app.domain.pipeline_types import RuleHints

# The trailing "/no_think" disables Qwen3's chain-of-thought so the model emits
# the JSON object directly (thinking tokens would otherwise fight the JSON grammar).
SYSTEM_PROMPT = (
    "You are an expert at reading Indian GST tax invoices and returning STRICT "
    "JSON. You output ONLY a single JSON object — no markdown, no code fences, no "
    "commentary. You never invent values you cannot read; unknown fields are null. "
    "/no_think"
)

_SCHEMA_TEXT = """\
Return ONE JSON object with EXACTLY these keys (no extra keys, no comments):
{
 "seller": {"name","address","gstin","phone","email","pincode"},
 "buyer":  {"name","address","gstin","phone","email","pincode"},
 "invoice": {"number","date"},
 "lineItems": [{"description","hsnSac","quantity","unit","rate","discount",
                "taxableAmount","gstRate","cgstAmount","sgstAmount","igstAmount","lineTotal"}],
 "taxSummary": [{"rate","taxableAmount","cgst","sgst","igst"}],
 "totals": {"subTotal","discountTotal","taxableTotal","taxTotal","roundOff","grandTotal","amountInWords"}
}
Rules:
- seller = the SUPPLIER / issuer (From / Sold By / seller GSTIN). buyer = the recipient (Bill To / Ship To).
- Money = plain numbers with 2 decimals (no currency symbol, no commas). Percentages = plain numbers (e.g. 18).
- date = "YYYY-MM-DD" (Indian invoices are day-first, e.g. 03/04/2026 = 2026-04-03) or null. Anything unknown = null.
- hsnSac = the line's HSN or SAC code — a 4/6/8-digit number, usually the column right after the description. Copy it for EVERY line that shows one; never leave it out.
- taxableAmount = the taxable value of the line AFTER any discount. gstRate = the total GST percent on the line.
- Intra-state (seller & buyer GSTIN state codes equal) -> fill cgstAmount + sgstAmount, set igstAmount null.
- Inter-state (state codes differ, or the buyer has no GSTIN) -> fill igstAmount, set cgstAmount + sgstAmount null.
- totals.taxableTotal is AFTER discount; grandTotal = taxableTotal + taxTotal + roundOff.
- One lineItems entry per product row. Never merge rows. Never invent rows.
- Copy GSTIN / invoice number / grand total EXACTLY as printed. Output the JSON object only.\
"""


def _hint_block(hints: RuleHints) -> str:
    """A short 'facts read from the page' section that grounds the model."""
    lines: list[str] = []
    if hints.gstins:
        lines.append(f"- GSTINs on the document: {', '.join(hints.gstins)}")
    if hints.invoice_number:
        lines.append(f"- Invoice number likely: {hints.invoice_number}")
    if hints.invoice_date:
        lines.append(f"- Invoice date likely: {hints.invoice_date}")
    if hints.hsn_codes:
        lines.append(f"- HSN/SAC codes on the document: {', '.join(hints.hsn_codes)}")
    if hints.grand_total:
        lines.append(f"- Grand total likely: {hints.grand_total}")
    if not lines:
        return ""
    return "Facts read directly from the page (use them; correct only if clearly wrong):\n" + "\n".join(lines)


def build_user_prompt(markdown: str, hints: RuleHints, char_budget: int) -> str:
    """Assemble the user turn: schema + grounding hints + the document text."""
    body = markdown[:char_budget]
    hint_block = _hint_block(hints)
    parts = [_SCHEMA_TEXT]
    if hint_block:
        parts.append(hint_block)
    parts.append("INVOICE (markdown/text):\n" + body)
    return "\n\n".join(parts)


def _party_schema() -> dict:
    props = {k: {"type": ["string", "null"]} for k in ("name", "address", "gstin", "phone", "email", "pincode")}
    return {"type": "object", "properties": props, "required": ["name"]}


def build_json_schema() -> dict:
    """JSON Schema for the response (compiled to a GBNF grammar by llama.cpp)."""
    number_or_null = {"type": ["number", "null"]}
    line_item = {
        "type": "object",
        "properties": {
            "description": {"type": ["string", "null"]},
            "hsnSac": {"type": ["string", "null"]},
            "quantity": number_or_null,
            "unit": {"type": ["string", "null"]},
            "rate": number_or_null,
            "discount": number_or_null,
            "taxableAmount": number_or_null,
            "gstRate": number_or_null,
            "cgstAmount": number_or_null,
            "sgstAmount": number_or_null,
            "igstAmount": number_or_null,
            "lineTotal": number_or_null,
        },
        "required": ["description", "quantity", "rate", "taxableAmount"],
    }
    tax_slab = {
        "type": "object",
        "properties": {
            "rate": number_or_null,
            "taxableAmount": number_or_null,
            "cgst": number_or_null,
            "sgst": number_or_null,
            "igst": number_or_null,
        },
    }
    totals = {
        "type": "object",
        "properties": {
            "subTotal": number_or_null,
            "discountTotal": number_or_null,
            "taxableTotal": number_or_null,
            "taxTotal": number_or_null,
            "roundOff": number_or_null,
            "grandTotal": number_or_null,
            "amountInWords": {"type": ["string", "null"]},
        },
    }
    return {
        "type": "object",
        "properties": {
            "seller": _party_schema(),
            "buyer": _party_schema(),
            "invoice": {
                "type": "object",
                "properties": {"number": {"type": ["string", "null"]}, "date": {"type": ["string", "null"]}},
            },
            "lineItems": {"type": "array", "items": line_item},
            "taxSummary": {"type": "array", "items": tax_slab},
            "totals": totals,
        },
        "required": ["seller", "buyer", "invoice", "lineItems", "totals"],
    }
