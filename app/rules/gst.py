"""Indian GST domain rules — pure, deterministic, fully unit-tested.

Everything here is a pure function of its inputs (no I/O), so the GST maths is
verifiable in isolation and reused identically by the rule processor, the
validator, and the confidence scorer.

Key facts encoded:
  * A GSTIN is 15 chars: ``SS`` state code + 10-char PAN + entity/Z/check.
  * The place-of-supply origin is the GSTIN state prefix.
  * Intra-state supply ⇒ CGST + SGST; inter-state ⇒ IGST.
"""

from __future__ import annotations

import re

#: GST state code (first two digits of a GSTIN) → state name.
GST_STATE: dict[str, str] = {
    "01": "Jammu and Kashmir", "02": "Himachal Pradesh", "03": "Punjab",
    "04": "Chandigarh", "05": "Uttarakhand", "06": "Haryana", "07": "Delhi",
    "08": "Rajasthan", "09": "Uttar Pradesh", "10": "Bihar", "11": "Sikkim",
    "12": "Arunachal Pradesh", "13": "Nagaland", "14": "Manipur", "15": "Mizoram",
    "16": "Tripura", "17": "Meghalaya", "18": "Assam", "19": "West Bengal",
    "20": "Jharkhand", "21": "Odisha", "22": "Chhattisgarh", "23": "Madhya Pradesh",
    "24": "Gujarat", "25": "Daman and Diu",
    "26": "Dadra and Nagar Haveli and Daman and Diu", "27": "Maharashtra",
    "28": "Andhra Pradesh", "29": "Karnataka", "30": "Goa", "31": "Lakshadweep",
    "32": "Kerala", "33": "Tamil Nadu", "34": "Puducherry",
    "35": "Andaman and Nicobar Islands", "36": "Telangana", "37": "Andhra Pradesh",
    "38": "Ladakh", "97": "Other Territory", "99": "Centre Jurisdiction",
}

#: Structural GSTIN pattern: 2-digit state, 5 letters (PAN entity), 4 digits,
#: 1 letter (PAN check), 1 entity digit/letter, ``Z``, 1 check char.
GSTIN_RE = re.compile(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]\b")

#: Standard Indian GST rate slabs (percent).
GST_RATE_SLABS: tuple[float, ...] = (0.0, 0.25, 3.0, 5.0, 12.0, 18.0, 28.0)


def normalize_gstin(value: str | None) -> str | None:
    """Upper-case, strip spaces, and return a GSTIN only if it is 15 chars."""
    if not value:
        return None
    cleaned = re.sub(r"\s+", "", str(value)).upper()
    return cleaned if len(cleaned) == 15 else None


def is_valid_gstin(value: str | None) -> bool:
    """Structural validity check (state prefix + PAN shape + ``Z``)."""
    cleaned = normalize_gstin(value)
    return bool(cleaned and GSTIN_RE.fullmatch(cleaned) and cleaned[:2] in GST_STATE)


def find_gstins(text: str) -> list[str]:
    """Every distinct GSTIN appearing in free text, in order of appearance."""
    seen: list[str] = []
    for match in GSTIN_RE.findall((text or "").upper()):
        if match[:2] in GST_STATE and match not in seen:
            seen.append(match)
    return seen


def state_code_of(gstin: str | None) -> str | None:
    """The 2-digit GST state code embedded in a GSTIN."""
    cleaned = normalize_gstin(gstin)
    if cleaned and cleaned[:2] in GST_STATE:
        return cleaned[:2]
    return None


def state_name_of(gstin: str | None) -> str | None:
    """The state name for a GSTIN's prefix."""
    code = state_code_of(gstin)
    return GST_STATE.get(code) if code else None


def pan_of(gstin: str | None) -> str | None:
    """The 10-char PAN embedded in a GSTIN (chars 3–12)."""
    cleaned = normalize_gstin(gstin)
    return cleaned[2:12] if cleaned else None


def is_inter_state(seller_gstin: str | None, buyer_gstin: str | None) -> bool | None:
    """True if IGST applies (different states), False for CGST+SGST.

    Returns ``None`` when it cannot be decided (a party GSTIN is missing) — the
    caller then trusts whichever tax columns the document actually printed.
    """
    seller = state_code_of(seller_gstin)
    buyer = state_code_of(buyer_gstin)
    if not seller or not buyer:
        return None
    return seller != buyer


def nearest_rate_slab(rate: float | None, tolerance: float = 0.6) -> float | None:
    """Snap a noisy GST percent to the nearest standard slab, if close enough."""
    if rate is None:
        return None
    best = min(GST_RATE_SLABS, key=lambda slab: abs(slab - rate))
    return best if abs(best - rate) <= tolerance else round(rate, 2)
