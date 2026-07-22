"""Unit tests for robust LLM-JSON parsing/repair."""

from __future__ import annotations

import pytest

from app.extraction.json_utils import parse_json_object


def test_plain_object():
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_code_fence():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_trailing_commentary():
    text = 'Here is the invoice:\n{"a": 1, "b": [1,2]}\nThanks!'
    assert parse_json_object(text) == {"a": 1, "b": [1, 2]}


def test_trailing_comma_repair():
    assert parse_json_object('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_nested_braces_and_strings():
    text = '{"note": "value with } brace", "inner": {"x": 1}}'
    assert parse_json_object(text) == {"note": "value with } brace", "inner": {"x": 1}}


@pytest.mark.parametrize("bad", ["", "no json here", "[1,2,3]"])
def test_invalid_raises(bad):
    with pytest.raises(ValueError):
        parse_json_object(bad)
