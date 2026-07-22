"""Concurrency / throughput load test against a RUNNING server.

Fires N total requests at ``/parse`` with a configurable concurrency and reports
throughput, latency percentiles and the error rate — the numbers you need before
pointing thousands of invoices at the box.

Start the server first (``scripts/serve.sh``), then:

    python scripts/loadtest.py --file invoice.pdf --requests 50 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


async def _one(client: httpx.AsyncClient, url: str, data: bytes, name: str, headers: dict) -> tuple[bool, float]:
    start = time.monotonic()
    try:
        files = {"file": (name, data, "application/octet-stream")}
        resp = await client.post(url, files=files, headers=headers, timeout=None)
        return resp.status_code == 200, time.monotonic() - start
    except Exception:
        return False, time.monotonic() - start


async def _worker(queue: asyncio.Queue, client, url, data, name, headers, results) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        results.append(await _one(client, url, data, name, headers))
        queue.task_done()


async def run(args) -> None:
    with open(args.file, "rb") as handle:
        data = handle.read()
    headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}
    url = f"{args.base_url.rstrip('/')}/parse"

    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(args.requests):
        queue.put_nowait(1)

    results: list[tuple[bool, float]] = []
    start = time.monotonic()
    async with httpx.AsyncClient() as client:
        workers = [
            asyncio.create_task(_worker(queue, client, url, data, args.file, headers, results))
            for _ in range(args.concurrency)
        ]
        await asyncio.gather(*workers)
    wall = time.monotonic() - start

    latencies = sorted(t for _, t in results)
    ok = sum(1 for success, _ in results if success)
    print(f"requests={len(results)} ok={ok} errors={len(results) - ok}")
    print(f"wall={wall:.1f}s throughput={len(results) / wall:.2f} req/s")
    if latencies:
        print(f"latency  p50={_pct(latencies, 50):.2f}s  p95={_pct(latencies, 95):.2f}s  max={latencies[-1]:.2f}s")


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, int(round(pct / 100.0 * (len(values) - 1))))
    return values[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR /parse load test")
    parser.add_argument("--file", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--key", default="", help="OCR_API_KEY if the server requires one")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
