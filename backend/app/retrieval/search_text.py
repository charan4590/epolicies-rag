"""Extra keyword-search terms derived from a chunk's text (the stored text itself is unchanged).

PostgreSQL full-text search keeps some formats as one token or cannot know what a range contains:

* ``Rawlings/Machinify``, ``OB/GYN``, ``member/patient`` -> also ``Rawlings Machinify``, ``OB GYN`` …
* ``RU486`` -> also ``RU 486``;  ``RU-486`` -> also ``RU486`` (so both spellings match)
* code ranges ``99202 - 99205`` -> also ``99203 99204`` (5-digit codes, at most 50 codes per range)

URLs are ignored, so link paths (``claim_call/ePolicies/content``) do not add noise.
"""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://[^\s)\]>]+")
_SLASH_WORDS_RE = re.compile(r"(?<![\w/])([A-Za-z][A-Za-z'’-]*(?:/[A-Za-z][A-Za-z'’-]*)+)(?![\w/])")
_LETTERS_DIGITS_RE = re.compile(r"\b([A-Za-z]{2,})(\d{2,})\b")
_HYPHEN_CODE_RE = re.compile(r"\b([A-Za-z]{2,})-(\d{2,})\b")
_CODE_RANGE_RE = re.compile(r"(?<![\d./-])(\d{5})\s*[-–]\s*(\d{5})(?![\d./]|-\d)")
MAX_RANGE_CODES = 50


def extra_search_terms(text: str) -> list[str]:
    """Additional terms to index for ``text`` (empty when nothing applies)."""
    clean = _URL_RE.sub(" ", text)
    extras: list[str] = []
    for match in _SLASH_WORDS_RE.finditer(clean):
        extras.append(match.group(1).replace("/", " "))
    for letters, digits in _LETTERS_DIGITS_RE.findall(clean):
        extras.append(f"{letters} {digits}")
    for letters, digits in _HYPHEN_CODE_RE.findall(clean):
        extras.append(f"{letters}{digits}")
    for low, high in _CODE_RANGE_RE.findall(clean):
        start, end = int(low), int(high)
        if start < end and end - start <= MAX_RANGE_CODES:
            extras.extend(str(code) for code in range(start + 1, end))
    return list(dict.fromkeys(extras))


def search_text(text: str) -> str:
    """The body text that goes into the full-text index: the chunk text plus extra terms."""
    extras = extra_search_terms(text)
    return f"{text}\n{' '.join(extras)}" if extras else text
