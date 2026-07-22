"""Jayhind Invoice-OCR — a local, CPU-only, accuracy-first extraction service.

Pipeline (all local, no GPU, no network):

    upload → ingest → preprocess → PaddleOCR-VL 1.6 → layout analysis
           → rule-based processing → Qwen3-8B (local CPU) → business-rule
           → confidence scoring → validated ExtractedInvoice JSON
"""

__version__ = "2.0.0"
