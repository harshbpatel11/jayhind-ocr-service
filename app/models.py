"""Response contract shared with the NestJS backend (`InvoiceExtractionClient`)."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

ExtractionMethod = Literal["pdf-text", "ocr"]


class Token(BaseModel):
    text: str
    #: [x1, y1, x2, y2] in page coordinates (points for PDFs, pixels for images).
    bbox: List[float]
    #: 1.0 on the pdf-text path (characters are exact, not recognised).
    confidence: float = 1.0


class Table(BaseModel):
    rows: List[List[Optional[str]]]


class Page(BaseModel):
    index: int
    width: float
    height: float
    text: str
    tokens: List[Token] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)


class ExtractResponse(BaseModel):
    method: ExtractionMethod
    pageCount: int
    durationMs: int
    #: Reading-order text of the whole document (pages joined by a blank line).
    text: str
    pages: List[Page]


# ── Structured invoice (mirror of the TS `ExtractedInvoice` contract) ─────────

class PartyBlock(BaseModel):
    name: str
    address: str
    gstin: Optional[str] = None
    stateCode: Optional[str] = None
    stateName: Optional[str] = None
    pan: Optional[str] = None
    #: Printed on the invoice but absent from the GSTIN and the GST registry —
    #: the document is the only source, so the party form falls back to these.
    phone: Optional[str] = None
    email: Optional[str] = None
    pincode: Optional[str] = None


class LineItem(BaseModel):
    description: str
    hsnSac: Optional[str] = None
    quantity: float
    unit: Optional[str] = None
    rate: float
    discount: Optional[float] = None
    taxableAmount: float
    gstRate: Optional[float] = None
    cgstAmount: Optional[float] = None
    sgstAmount: Optional[float] = None
    igstAmount: Optional[float] = None
    lineTotal: Optional[float] = None
    confidence: float


class TaxSlab(BaseModel):
    rate: float
    taxableAmount: float
    cgst: float
    sgst: float
    igst: float


class Totals(BaseModel):
    taxableTotal: float
    taxTotal: float
    roundOff: float
    grandTotal: float
    amountInWords: Optional[str] = None


class InvoiceMeta(BaseModel):
    number: str
    date: Optional[str] = None


class ExtractedInvoice(BaseModel):
    schemaVersion: int
    seller: PartyBlock
    buyer: PartyBlock
    invoice: InvoiceMeta
    lineItems: List[LineItem] = Field(default_factory=list)
    taxSummary: List[TaxSlab] = Field(default_factory=list)
    totals: Totals
    fieldConfidence: dict = Field(default_factory=dict)
    #: One headline 0..1 score for the whole extraction — 50% header fields,
    #: 50% line items. The Master Hub archives documents below its threshold.
    overallConfidence: float = 0.0


class ParseResponse(BaseModel):
    method: ExtractionMethod
    #: Only the offline rule-based structurer exists; recorded for provenance.
    structuringMethod: Literal["rules"] = "rules"
    pageCount: int
    durationMs: int
    #: Reading-order text of the whole document (stored as the scan's rawText).
    text: str
    invoice: ExtractedInvoice


class HealthResponse(BaseModel):
    status: Literal["ok"]
    ocr_available: bool
    gpu: bool
