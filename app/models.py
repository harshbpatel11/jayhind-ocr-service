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


class HealthResponse(BaseModel):
    status: Literal["ok"]
    ocr_available: bool
    gpu: bool
