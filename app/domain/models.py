"""API-contract models: the exact ``ExtractedInvoice`` / ``ParseResult`` shape.

These Pydantic models are the single source of truth for the JSON the service
returns from ``POST /parse``. They MUST stay byte-compatible with the ERP's
TypeScript contract in
``jayhind-client-back/src/const/invoice-scan-contract.ts`` — the child ERP maps
these exact keys into the scan-review screen and the draft voucher.

Fields are declared in idiomatic ``snake_case`` and serialised to the contract's
``camelCase`` via a Pydantic alias generator, so the Python stays PEP 8 while the
wire format matches the ERP. Always dump with ``by_alias=True`` (the API layer
does this through ``model_dump(by_alias=True)``).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

#: Structured-invoice schema version. Bump only on a breaking contract change;
#: the ERP pins ``EXTRACTED_SCHEMA_VERSION = 1``.
SCHEMA_VERSION: int = 1


class _Contract(BaseModel):
    """Base for every wire model: snake_case in Python, camelCase on the wire."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        # Numbers arrive from the LLM as strings/None routinely; validation and
        # coercion happen in the pipeline, so keep the models permissive here.
        extra="ignore",
    )


class PartyBlock(_Contract):
    """A supplier/recipient block exactly as printed on the invoice."""

    name: str = ""
    address: str = ""
    gstin: str | None = None
    state_code: str | None = None
    state_name: str | None = None
    pan: str | None = None
    phone: str | None = None
    email: str | None = None
    pincode: str | None = None


class LineItem(_Contract):
    """One invoice line. Money is a plain number, GST split by supply type."""

    description: str = ""
    hsn_sac: str | None = None
    quantity: float = 0.0
    unit: str | None = None
    rate: float = 0.0
    discount: float | None = None
    taxable_amount: float = 0.0
    gst_rate: float | None = None
    cgst_amount: float | None = None
    sgst_amount: float | None = None
    igst_amount: float | None = None
    line_total: float | None = None
    confidence: float = 0.0


class TaxSlab(_Contract):
    """One row of the invoice's tax summary, grouped by GST rate."""

    rate: float = 0.0
    taxable_amount: float = 0.0
    cgst: float = 0.0
    sgst: float = 0.0
    igst: float = 0.0


class Totals(_Contract):
    """Invoice totals. ``taxableTotal`` is AFTER any whole-bill discount."""

    sub_total: float = 0.0
    discount_total: float = 0.0
    taxable_total: float = 0.0
    tax_total: float = 0.0
    round_off: float = 0.0
    grand_total: float = 0.0
    amount_in_words: str | None = None


class InvoiceMeta(_Contract):
    """Document-level identity."""

    number: str = ""
    date: str | None = None
    #: Explicit payment-due date if printed ("YYYY-MM-DD"), else derived from
    #: ``date + paymentTermsDays`` when both are known, else null.
    due_date: str | None = None
    #: Credit-period length in days read from text like "Net 30" / "Payment Due: 30 Days".
    payment_terms_days: int | None = None


class ExtractedInvoice(_Contract):
    """The structured invoice returned inside every ``ParseResult``."""

    schema_version: int = SCHEMA_VERSION
    seller: PartyBlock = PartyBlock()
    buyer: PartyBlock = PartyBlock()
    invoice: InvoiceMeta = InvoiceMeta()
    line_items: list[LineItem] = []
    tax_summary: list[TaxSlab] = []
    totals: Totals = Totals()
    #: Per-field score 0..1 keyed by JSON path (e.g. ``"seller.gstin"``) — drives
    #: the amber highlights in the ERP's scan-review screen.
    field_confidence: dict[str, float] = {}
    #: One headline 0..1 score (≈50 % header, 50 % line items).
    overall_confidence: float = 0.0


class ParseResult(_Contract):
    """The full ``POST /parse`` response: provenance + the structured invoice."""

    #: ``pdf-text`` when the digital PDF text layer was used, else ``ocr``.
    method: str = "ocr"
    #: Provenance of the structuring. The ERP's union type only admits ``rules``;
    #: the LLM is an internal implementation detail of that structuring.
    structuring_method: str = "rules"
    page_count: int = 0
    duration_ms: int = 0
    #: Reading-order text of the whole document (stored as the scan's rawText).
    text: str = ""
    invoice: ExtractedInvoice = ExtractedInvoice()
