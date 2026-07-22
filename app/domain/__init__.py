"""Domain layer: the wire contract, internal pipeline types, and stage interfaces."""

from app.domain.models import (
    SCHEMA_VERSION,
    ExtractedInvoice,
    InvoiceMeta,
    LineItem,
    ParseResult,
    PartyBlock,
    TaxSlab,
    Totals,
)
from app.domain.pipeline_types import (
    BlockType,
    ExtractionContext,
    LayoutBlock,
    LayoutTable,
    LoadedDocument,
    PageImage,
    PageLayout,
    ReaderOutput,
    RuleHints,
    SourceKind,
    TableCell,
)

__all__ = [
    "SCHEMA_VERSION",
    "ExtractedInvoice",
    "InvoiceMeta",
    "LineItem",
    "ParseResult",
    "PartyBlock",
    "TaxSlab",
    "Totals",
    "BlockType",
    "ExtractionContext",
    "LayoutBlock",
    "LayoutTable",
    "LoadedDocument",
    "PageImage",
    "PageLayout",
    "ReaderOutput",
    "RuleHints",
    "SourceKind",
    "TableCell",
]
