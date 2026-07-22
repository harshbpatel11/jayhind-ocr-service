"""Robust JSON extraction/repair for LLM output.

Even a grammar-constrained model occasionally wraps its answer in a ``` fence,
adds a trailing comma, or emits a stray token. These helpers recover the JSON
object without ever executing model text, so the extractor never crashes the
pipeline on a formatting hiccup.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    stripped = _FENCE_RE.sub("", stripped)
    return stripped.strip()


def _slice_outermost_object(text: str) -> str:
    """Return the substring from the first ``{`` to its matching ``}``.

    Brace-counting (not a greedy first/last slice) so trailing model chatter or a
    second object doesn't corrupt the parse.
    """
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in ``text``, repairing common LLM defects.

    Raises ``ValueError`` if no object can be recovered — the caller treats that
    as a retryable extraction error.
    """
    if not text or not text.strip():
        raise ValueError("empty model output")
    candidate = _slice_outermost_object(_strip_fences(text))
    for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("model returned JSON that was not an object")
    raise ValueError("could not parse a JSON object from model output")
