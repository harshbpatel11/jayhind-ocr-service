"""GSTIN rules for invoice structuring — port of `gstin.const.ts`.

A GSTIN is 15 chars: <2-digit state><10-char PAN><entity char><'Z'><checksum>.
Only the state code and PAN are encoded inside; the name/address are not.

The checksum is used to *disambiguate* an OCR repair, never to reject a GSTIN the
document states plainly — real sample/legacy GSTINs routinely fail it.
"""
import re
from typing import Dict, List, Optional

GSTIN_REGEX = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
#: Non-anchored form for scanning free text.
GSTIN_SCAN_REGEX = re.compile(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]")
#: Any standalone 15-char alphanumeric run — a GSTIN *candidate* to repair.
_GSTIN_CANDIDATE_REGEX = re.compile(r"\b[0-9A-Z]{15}\b")

GST_STATE_NAME_BY_CODE: Dict[str, str] = {
    "01": "Jammu and Kashmir", "02": "Himachal Pradesh", "03": "Punjab", "04": "Chandigarh",
    "05": "Uttarakhand", "06": "Haryana", "07": "Delhi", "08": "Rajasthan", "09": "Uttar Pradesh",
    "10": "Bihar", "11": "Sikkim", "12": "Arunachal Pradesh", "13": "Nagaland", "14": "Manipur",
    "15": "Mizoram", "16": "Tripura", "17": "Meghalaya", "18": "Assam", "19": "West Bengal",
    "20": "Jharkhand", "21": "Odisha", "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat",
    "25": "Dadra and Nagar Haveli and Daman and Diu", "26": "Dadra and Nagar Haveli and Daman and Diu",
    "27": "Maharashtra", "28": "Andhra Pradesh", "29": "Karnataka", "30": "Goa", "31": "Lakshadweep",
    "32": "Kerala", "33": "Tamil Nadu", "34": "Puducherry", "35": "Andaman and Nicobar Islands",
    "36": "Telangana", "37": "Andhra Pradesh", "38": "Ladakh",
}

_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gstin_checksum(gstin: str) -> Optional[str]:
    body = (gstin or "").upper()
    if len(body) < 14:
        return None
    total = 0
    for index in range(14):
        value = _ALPHABET.find(body[index])
        if value < 0:
            return None
        product = value * (1 if index % 2 == 0 else 2)
        total += product // 36 + product % 36
    return _ALPHABET[(36 - (total % 36)) % 36]


def is_checksum_valid(gstin: str) -> bool:
    value = (gstin or "").upper()
    return bool(GSTIN_REGEX.match(value)) and gstin_checksum(value) == value[14]


_DIGIT_FOR_LETTER = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "E": "3",
                     "A": "4", "S": "5", "G": "6", "C": "6", "T": "7", "Y": "7", "B": "8",
                     "R": "8", "P": "9"}
_LETTERS_FOR_DIGIT = {"0": ["O", "D", "Q"], "1": ["I", "L"], "2": ["Z"], "3": ["E", "J", "B"],
                      "4": ["A"], "5": ["S"], "6": ["G", "C"], "7": ["T", "Y"], "8": ["B", "R"],
                      "9": ["G", "P", "Q"]}
_MAX_REPAIRS = 3
_DIGIT_SLOTS = [0, 1, 7, 8, 9, 10]
_LETTER_SLOTS = [2, 3, 4, 5, 6, 11]


def _expand(chars: List[str], ambiguous) -> List[str]:
    candidates = [chars]
    for index, options in ambiguous:
        candidates = [[*c[:index], opt, *c[index + 1:]] for c in candidates for opt in options]
    return ["".join(c) for c in candidates]


def repair_gstin(raw: str) -> Optional[str]:
    """Repair a 15-char run whose chars the OCR confused with look-alikes → a
    valid GSTIN or None. A well-formed run is returned untouched (checksum never
    consulted). Ambiguity the checksum can't settle returns None rather than a
    confident wrong answer — a wrong GSTIN matches the wrong party."""
    candidate = re.sub(r"[^0-9A-Z]", "", (raw or "").upper())
    if len(candidate) != 15:
        return None
    if GSTIN_REGEX.match(candidate):
        return candidate

    chars = list(candidate)
    ambiguous = []
    repairs = 0
    for index in _DIGIT_SLOTS:
        if chars[index].isdigit():
            continue
        fixed = _DIGIT_FOR_LETTER.get(chars[index])
        repairs += 1
        if not fixed or repairs > _MAX_REPAIRS:
            return None
        chars[index] = fixed
    for index in _LETTER_SLOTS:
        if chars[index].isalpha():
            continue
        options = _LETTERS_FOR_DIGIT.get(chars[index])
        repairs += 1
        if not options or repairs > _MAX_REPAIRS:
            return None
        if len(options) == 1:
            chars[index] = options[0]
        else:
            ambiguous.append((index, options))
    if chars[12] == "0":
        repairs += 1
        if repairs > _MAX_REPAIRS:
            return None
        chars[12] = "O"
    if chars[13] != "Z":
        repairs += 1
        if chars[13] != "2" or repairs > _MAX_REPAIRS:
            return None
        chars[13] = "Z"

    well_formed = [g for g in _expand(chars, ambiguous) if GSTIN_REGEX.match(g)]
    if not well_formed:
        return None
    if len(well_formed) == 1:
        return well_formed[0]
    confirmed = [g for g in well_formed if is_checksum_valid(g)]
    return confirmed[0] if len(confirmed) == 1 else None


def derive_basics(raw: str) -> dict:
    """Offline-knowable pieces of a GSTIN (state + PAN)."""
    gst_no = (raw or "").strip().upper()
    if not GSTIN_REGEX.match(gst_no):
        return {"valid": False, "gstNo": gst_no, "stateCode": None, "stateName": None, "panNo": None}
    state_code = gst_no[:2]
    return {
        "valid": True, "gstNo": gst_no, "stateCode": state_code,
        "stateName": GST_STATE_NAME_BY_CODE.get(state_code), "panNo": gst_no[2:12],
    }


def find_gstins(text: str) -> List[str]:
    """Every GSTIN in the text, uppercased, de-duplicated in order of appearance.
    Verbatim matches first, then 15-char runs repaired against the grammar."""
    upper = re.sub(r"\s+", " ", (text or "").upper())
    found: List[str] = list(GSTIN_SCAN_REGEX.findall(upper))
    for candidate in _GSTIN_CANDIDATE_REGEX.findall(upper):
        if GSTIN_REGEX.match(candidate):
            continue
        repaired = repair_gstin(candidate)
        if repaired:
            found.append(repaired)
    seen = set()
    unique = []
    for g in found:
        if g not in seen:
            seen.add(g)
            unique.append(g)
    return unique
