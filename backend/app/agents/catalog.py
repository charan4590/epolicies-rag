"""Questions about the knowledge base itself: "how many policies are there?", "list all policies".

Retrieval only returns a handful of passages, and no passage states how many documents exist, so
these questions are answered from the document index in PostgreSQL instead (no Gemini call).
An optional title filter is supported ("list all policies about Medicare"). Anything that looks
like a question about policy *content* ("how many policies cover abortion?") returns ``None`` so
the normal RAG workflow handles it.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Literal

_DOC = r"(?:policies|policy\s+documents|policy\s+files|documents|docs|files|policy\s+names|policy\s+titles)"
_COUNT = re.compile(
    rf"\b(?:how\s+many|(?:the\s+)?(?:total\s+)?(?:number|count)\s+of|total)\s+(?:the\s+)?(?:indexed\s+)?{_DOC}\b",
    re.IGNORECASE,
)
_LIST = re.compile(
    rf"\b(?:list|name|show|give(?:\s+me)?|display|enumerate)\s+(?:me\s+)?(?:all\s+)?(?:of\s+)?(?:the\s+)?"
    rf"(?:available\s+|indexed\s+)?{_DOC}\b"
    rf"|\b(?:which|what)\s+{_DOC}\s+(?:do\s+you\s+have|are\s+(?:there|available|indexed|loaded))\b"
    rf"|\b(?:which|what)\s+{_DOC}(?=\s+(?:mention|mentions|contain|contains|include|includes|are\s+about|"
    rf"relate\s+to|are\s+related\s+to)\b)",
    re.IGNORECASE,
)
_FILTER_START = re.compile(
    r"^(?:about|on|for|in|within|under|regarding|related\s+to|relate\s+to|are\s+related\s+to|are\s+about|"
    r"containing|contain|contains|with|mentioning|mention|mentions|include|includes|named|titled|called|"
    r"that\s+(?:mention|contain|include)|whose\s+(?:title|name)\s+(?:contains|includes))\b",
    re.IGNORECASE,
)
_FILLER = {
    "a", "all", "an", "any", "are", "assistant", "available", "base", "can", "could", "currently", "database",
    "do", "does", "epolicies", "exist", "exists", "for", "give", "have", "here", "in", "indexed", "is", "it",
    "knowledge", "list", "loaded", "me", "name", "names", "now", "of", "our", "please", "policies", "policy",
    "stored", "system", "tell", "the", "their", "there", "this", "title", "titles", "total", "us", "we", "what",
    "which", "you", "your",
}
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'&()\-/]*")
MAX_FILTERED_LIST = 100
# Small typos in the key words are accepted: "how mny polices are there"
_KEY_WORDS = ["many", "policies", "policy", "documents", "number", "total"]
_NOT_TYPOS = _FILLER | {"any", "man", "mane", "nay", "police", "polite"}


def _fix_key_word_typos(text: str) -> str:
    def fix(match: re.Match[str]) -> str:
        word = match.group(0)
        if len(word) < 3 or word.lower() in _NOT_TYPOS or word.lower() in _KEY_WORDS:
            return word
        close = difflib.get_close_matches(word.lower(), _KEY_WORDS, n=1, cutoff=0.8)
        return close[0] if close else word
    return re.sub(r"[A-Za-z]+", fix, text)


def _strip_leading_filler(text: str) -> str:
    """'are there in medicare' -> 'in medicare' (a filter start is kept even if it is a filler word)."""
    while text and not _FILTER_START.match(text):
        first, _, rest = text.partition(" ")
        if first.lower() not in _FILLER:
            break
        text = rest.strip()
    return text


@dataclass
class CatalogQuestion:
    kind: Literal["count", "list"]
    filter_terms: list[str] = field(default_factory=list)


def parse_catalog_question(question: str) -> CatalogQuestion | None:
    """Recognise a question about the document collection; ``None`` for anything else."""
    text = _fix_key_word_typos(question.strip().rstrip("?.! "))
    match = _COUNT.search(text) or _LIST.search(text)
    if not match:
        return None
    kind: Literal["count", "list"] = "count" if _COUNT.search(text) else "list"
    before, after = text[: match.start()], _strip_leading_filler(text[match.end():].strip(" ,:"))
    if [w for w in _WORD.findall(before) if w.lower() not in _FILLER]:
        return None  # e.g. "under COB, how many policies ..." – a content question
    filter_terms: list[str] = []
    start = _FILTER_START.match(after)
    if start:
        filter_terms = [w for w in _WORD.findall(after[start.end():]) if w.lower() not in _FILLER]
        if not filter_terms:
            return None
    elif [w for w in _WORD.findall(after) if w.lower() not in _FILLER]:
        return None  # e.g. "how many policies cover abortion" – needs the policy text
    return CatalogQuestion(kind=kind, filter_terms=filter_terms)


def build_catalog_answer(parsed: CatalogQuestion, documents: list[tuple[str, str]],
                         sections: list[tuple[str, str]] | None = None) -> str | None:
    """Markdown answer from the indexed documents' (title, relative_path); ``None`` when a filter matches nothing.

    Every document is counted; when two documents share a title, their file paths tell them apart. When no
    title matches the filter, section headings ``sections`` (title, heading) are searched instead.
    """
    documents = sorted({(t.strip() or p, p) for t, p in documents}, key=lambda d: (d[0].lower(), d[1]))
    repeated = {t for t, _ in documents if sum(1 for other, _ in documents if other == t) > 1}
    titles = [f"{t} ({p})" if t in repeated else t for t, p in documents]
    if parsed.filter_terms:
        terms = [t.lower() for t in parsed.filter_terms]
        titles = [t for t in titles if all(term in t.lower() for term in terms)]
        topic = " ".join(parsed.filter_terms)
        if not titles:
            return _sections_answer(topic, terms, sections or [])
        lines = [f"**{len(titles)}** indexed policy document{'s' if len(titles) != 1 else ''} "
                 f"{'have' if len(titles) != 1 else 'has'} “{topic}” in the title:"]
        shown = titles[:MAX_FILTERED_LIST]
    else:
        lines = [f"There are **{len(titles)}** policy documents in the knowledge base."]
        if parsed.kind == "count":
            lines.append("\nAsk “list all policies” to see their names, or open the **Knowledge base** page.")
            return "\n".join(lines)
        lines[0] += " Here they are:"
        shown = titles
    lines.append("")
    lines.extend(f"{i}. {title}" for i, title in enumerate(shown, start=1))
    if len(shown) < len(titles):
        lines.append(f"\n…and {len(titles) - len(shown)} more. See the **Knowledge base** page for the full list.")
    return "\n".join(lines)


def _sections_answer(topic: str, terms: list[str], sections: list[tuple[str, str]]) -> str | None:
    """No policy title matches: list the sections whose heading matches, grouped by policy."""
    matches = sorted({(t, h) for t, h in sections if all(term in h.lower() for term in terms)},
                     key=lambda th: (th[0].lower(), th[1].lower()))
    if not matches:
        return None
    policies = sorted({t for t, _ in matches}, key=str.lower)
    lines = [f"No policy title contains “{topic}”, but **{len(matches)}** section"
             f"{'s' if len(matches) != 1 else ''} in **{len(policies)}** polic{'ies' if len(policies) != 1 else 'y'} "
             f"{'have' if len(matches) != 1 else 'has'} it in the heading:", ""]
    lines.extend(f"{i}. {t} › {h}" for i, (t, h) in enumerate(matches[:MAX_FILTERED_LIST], start=1))
    if len(matches) > MAX_FILTERED_LIST:
        lines.append(f"\n…and {len(matches) - MAX_FILTERED_LIST} more.")
    return "\n".join(lines)
