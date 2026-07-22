"""API tests via FastAPI TestClient with a fake-backed container."""

from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import Container
from app.extraction.rules_extractor import RulesExtractor
from app.main import create_app
from tests.conftest import SAMPLE_MARKDOWN_INTRASTATE, FakeReader


def _png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.full((120, 200, 3), 255, dtype=np.uint8)).save(buffer, "PNG")
    return buffer.getvalue()


def _client(*, api_key: str = "", markdown: str = SAMPLE_MARKDOWN_INTRASTATE) -> TestClient:
    container = Container(Settings(api_key=api_key, reader_engine="null", extractor_engine="rules", preprocess_enabled=False))
    container.__dict__["reader"] = FakeReader(markdown)
    container.__dict__["extractor"] = RulesExtractor()
    return TestClient(create_app(container))


def test_health():
    with _client() as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["engine"] == "fake"
        assert body["gpu"] is False


def test_parse_returns_contract():
    with _client() as client:
        resp = client.post("/parse", files={"file": ("scan.png", _png_bytes(), "image/png")})
        assert resp.status_code == 200
        body = resp.json()
        assert body["structuringMethod"] == "rules"
        inv = body["invoice"]
        assert inv["schemaVersion"] == 1
        assert len(inv["lineItems"]) == 2
        assert inv["seller"]["stateName"] == "Gujarat"
        assert set(["seller", "buyer", "invoice", "lineItems", "taxSummary", "totals",
                    "fieldConfidence", "overallConfidence"]).issubset(inv.keys())


def test_parse_empty_file_is_400():
    with _client() as client:
        resp = client.post("/parse", files={"file": ("empty.png", b"", "image/png")})
        assert resp.status_code == 400


def test_parse_unreadable_is_terminal_4xx():
    with _client() as client:
        resp = client.post("/parse", files={"file": ("x.bin", b"garbage-not-a-doc", "application/octet-stream")})
        assert 400 <= resp.status_code < 500


def test_auth_enforced_when_key_set():
    with _client(api_key="secret") as client:
        # missing key
        resp = client.post("/parse", files={"file": ("scan.png", _png_bytes(), "image/png")})
        assert resp.status_code == 401
        # correct key
        resp = client.post(
            "/parse",
            files={"file": ("scan.png", _png_bytes(), "image/png")},
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 200
