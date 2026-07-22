"""Latency benchmark for the OCR pipeline (in-process, real configured engines).

Measures wall-clock parse time against the project's targets:

    1 page  < 4 s        20 pages < 60 s        50 pages < 2 min

It runs the *actual* configured pipeline (reader + extractor from env), so with
the default Qwen3-8B on CPU expect times far above these targets — the project is
tuned for accuracy, not speed (see the README). Use ``OCR_EXTRACTOR_ENGINE=rules``
and/or ``OCR_READER_ENGINE=null`` to benchmark the pipeline plumbing alone.

Usage:
    python scripts/benchmark.py --file invoice.pdf --repeat 3
    python scripts/benchmark.py --synthetic 1,20,50
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.container import Container  # noqa: E402

TARGETS_S = {1: 4.0, 20: 60.0, 50: 120.0}


def _synthetic_pdf(pages: int) -> bytes:
    """A multi-page digital-PDF invoice for timing (has a text layer)."""
    import fitz

    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text(
            (40, 60),
            f"TAX INVOICE (page {i + 1})\n"
            "Sold By: ACME STEELS PRIVATE LIMITED  GSTIN: 24AJGPP6816J1ZY\n"
            "Bill To: JAYHIND ENTERPRISES  GSTIN: 27AABCU9603R1ZX\n"
            "Invoice No: INV-2026-0042  Date: 03/04/2026\n"
            "| Description | HSN | Qty | Rate | Taxable | GST % |\n"
            "| Steel Coil | 7208 | 10 | 5000 | 50000 | 18 |\n"
            "Grand Total: 59000.00",
            fontsize=10,
        )
    data = doc.tobytes()
    doc.close()
    return data


def _run(container: Container, data: bytes, name: str, mime: str, repeat: int) -> list[float]:
    times: list[float] = []
    for _ in range(repeat):
        start = time.monotonic()
        container.pipeline.run_sync(data, mime, name)
        times.append(time.monotonic() - start)
    return times


def _report(label: str, pages: int, times: list[float]) -> None:
    best, avg = min(times), statistics.mean(times)
    target = TARGETS_S.get(pages)
    verdict = ""
    if target is not None:
        verdict = "  PASS" if best <= target else f"  OVER (target {target:.0f}s)"
    print(f"{label:<22} pages={pages:<3} best={best:6.2f}s avg={avg:6.2f}s{verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR pipeline latency benchmark")
    parser.add_argument("--file", help="an invoice (PDF/PNG/JPG/TIFF) to benchmark")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--synthetic", default="1,20,50", help="comma page counts for synthetic PDFs")
    args = parser.parse_args()

    container = Container()
    print(f"reader={container.reader.engine}  extractor={container.extractor.name}\n")

    if args.file:
        with open(args.file, "rb") as handle:
            data = handle.read()
        result = container.pipeline.run_sync(data, "", args.file)
        times = _run(container, data, args.file, "", args.repeat)
        _report(args.file, result.page_count, times)
        return

    for token in args.synthetic.split(","):
        pages = int(token)
        data = _synthetic_pdf(pages)
        _report(f"synthetic-{pages}pg", pages, _run(container, data, f"{pages}p.pdf", "application/pdf", args.repeat))


if __name__ == "__main__":
    main()
