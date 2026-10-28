"""Structure-aware chunking.

Strategy
--------
1. Blocks are grouped into *sections* at heading boundaries; every chunk records
   the heading path of the section it came from.
2. Each block becomes one or more *units* no larger than ``chunk_size`` tokens:
   tables are split by row groups with the header (and delimiter) row repeated;
   lists are split between top-level items (no overlap); a top-level item that is itself too
   long (ePolicies step trees: Step → If… → Then…) is split between lines and each
   later part starts with its parent steps, so conditions are never lost; long
   prose is split at sentence boundaries (falling back to word boundaries).
3. Units are packed greedily into chunks within a section. When a section spans
   several chunks, the tail of the previous prose unit (≤ ``chunk_overlap``
   tokens, whole sentences only) is repeated at the start of the next chunk.
   Tables are never used as overlap; their split parts repeat the header instead,
   and continued step lists repeat their parent steps instead.
4. Chunks smaller than ``min_chunk_tokens`` (e.g. a heading followed straight
   by a sub-heading) are merged into a neighbour when the result still fits.

Token counts are an *approximation* (≈4 characters per token, see
:func:`estimate_tokens`) because no offline Gemini tokenizer is available.
``chunk_size`` is therefore a target, not an exact model-token limit; the
configured maximum (1800) stays well under gemini-embedding-001's 2048-token
input limit.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

from app.ingestion.markdown_parser import Block, ParsedDocument

CHUNKER_VERSION = "v4"  # v2: deep lists keep later headings; v3: split step lists repeat parent steps,
#                           overlap uses whole sentences only; v4: no URLs in the embedded text,
#                           extra keyword terms (slash words, code ranges)
_BULLET_RE = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s")
MAX_CONTEXT_LINE_CHARS = 200
MAX_CONTEXT_SHARE = 0.4  # parent steps may use at most this share of a chunk
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)\s]+\)")
_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9\"'(\[*_-])|\n{2,}|\n(?=\s*[-*+] |\s*\d+[.)] )")


def estimate_tokens(text: str) -> int:
    """Approximate token count (≈4 chars/token, floor of 1.3 tokens per word)."""
    if not text:
        return 0
    words = len(text.split())
    return max(1, math.ceil(max(len(text) / 4.0, words * 1.3)))


@dataclass
class Unit:
    text: str
    kind: str
    links: list[str]
    line_start: int
    line_end: int
    table_part: tuple[int, int] | None = None
    continued: bool = False  # starts with repeated parent steps of a split list item


    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass
class Section:
    heading_path: tuple[str, ...]
    heading: Block | None
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Chunk:
    document_id: str
    chunk_index: int
    text: str
    heading_path: list[str]
    source_urls: list[str]
    metadata: dict
    embedding_input: str
    content_hash: str

    @property
    def chunk_id(self) -> str:
        return make_chunk_id(self.document_id, self.chunk_index, self.content_hash)

    @property
    def token_count(self) -> int:
        return estimate_tokens(self.text)

    @property
    def char_count(self) -> int:
        return len(self.text)


def make_chunk_id(document_id: str, chunk_index: int, content_hash: str) -> str:
    digest = hashlib.sha256(f"{document_id}|{chunk_index}|{content_hash}".encode()).hexdigest()
    return "chk_" + digest[:24]


_BARE_URL_RE = re.compile(r"https?://[^\s)\]>|]+")


def text_for_embedding(text: str) -> str:
    """Chunk text without URLs: links keep their text ("[Travel and Lodging - Procedures](https://…)" ->
    "Travel and Lodging - Procedures"). URLs were ~17% of the text – cost without meaning."""
    without_links = _MD_LINK_RE.sub(lambda m: m.group(1) if not _BARE_URL_RE.fullmatch(m.group(1)) else "", text)
    return re.sub(r"[ \t]{2,}", " ", _BARE_URL_RE.sub("", without_links))


def build_embedding_input(title: str, heading_path: list[str], text: str) -> str:
    """The exact text embedded for a chunk: document/section context + content (URLs removed; the chunk
    text and its source URLs are stored unchanged for display)."""
    section = " > ".join(heading_path) if heading_path else "(document start)"
    return f"Document: {title}\nSection: {section}\n\n{text_for_embedding(text)}"


# --------------------------------------------------------------------------- splitting
def _split_prose(text: str, max_tokens: int) -> list[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]
    pieces = [p for p in _SENTENCE_RE.split(text) if p and p.strip()]
    out: list[str] = []
    current = ""
    for piece in pieces:
        if estimate_tokens(piece) > max_tokens:
            if current:
                out.append(current)
                current = ""
            out.extend(_split_words(piece, max_tokens))
            continue
        candidate = f"{current} {piece}".strip() if current else piece
        if estimate_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            out.append(current)
            current = piece
    if current:
        out.append(current)
    return out


def _split_words(text: str, max_tokens: int) -> list[str]:
    words = text.split(" ")
    out: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and estimate_tokens(candidate) > max_tokens:
            out.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        out.append(" ".join(current))
    # Last resort for pathological tokens without spaces (e.g. a giant URL).
    final: list[str] = []
    max_chars = max_tokens * 4
    for item in out:
        while len(item) > max_chars * 1.5:
            final.append(item[:max_chars])
            item = item[max_chars:]
        final.append(item)
    return final


def _table_units(block: Block, max_tokens: int) -> list[Unit]:
    assert block.table is not None
    header = "\n".join(block.table.header_lines)
    rows = block.table.row_lines
    budget = max(1, max_tokens - estimate_tokens(header))
    groups: list[list[str]] = []
    current: list[str] = []
    for row in rows:
        if current and estimate_tokens("\n".join(current + [row])) > budget:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    total = len(groups)
    return [
        Unit(
            text=header + "\n" + "\n".join(group),
            kind="table",
            links=block.links,
            line_start=block.line_start,
            line_end=block.line_end,
            table_part=(n + 1, total),
        )
        for n, group in enumerate(groups)
    ]


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parent_steps(lines: list[str], start: int, max_tokens: int) -> list[str]:
    """Bullet lines above ``lines[start]`` that contain it (each less indented than the next), outermost
    first – e.g. "- **Step:** 1", "  - **If…** Yes", "    - **Then…** Is the member in a vendor state?"."""
    limit = _indent(lines[start]) if lines[start].strip() else 10**6
    chain: list[str] = []
    for line in reversed(lines[:start]):
        if not line.strip() or not _BULLET_RE.match(line):
            continue
        if _indent(line) < limit:
            short = _MD_LINK_RE.sub(r"\1", line)  # link text only: URLs would crowd out the conditions
            if len(short) > MAX_CONTEXT_LINE_CHARS:
                short = short[:MAX_CONTEXT_LINE_CHARS].rstrip() + " …"
            chain.insert(0, short)
            limit = _indent(line)
            if limit == 0:
                break
    # Keep the outermost step and the closest parents when the chain is long.
    while len(chain) > 2 and estimate_tokens("\n".join(chain)) > max_tokens * MAX_CONTEXT_SHARE:
        del chain[1]
    return chain


def _split_nested_item(item: str, max_tokens: int) -> list[tuple[str, bool]]:
    """Split one oversized list item between lines; every part after the first starts with the
    parent steps of its first line. Returns (text, has_parent_steps) pairs."""
    lines = item.split("\n")
    parts: list[tuple[str, bool]] = []
    context: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if any(line.strip() for line in current):
            parts.append(("\n".join(context + current), bool(context)))

    for index, line in enumerate(lines):
        if current and estimate_tokens("\n".join(context + current + [line])) > max_tokens:
            flush()
            context, current = _parent_steps(lines, index, max_tokens), []
        room = max_tokens - estimate_tokens("\n".join(context))
        if estimate_tokens(line) > room:  # one very long line: split it, keeping the same parents
            flush()
            current = []
            for piece in _split_prose(line, max(32, room)):
                parts.append(("\n".join(context + [piece]), bool(context)))
            context = _parent_steps(lines, index + 1, max_tokens) if index + 1 < len(lines) else context
            continue
        current.append(line)
    flush()
    return parts


def _list_units(block: Block, max_tokens: int) -> list[Unit]:
    items = block.list_items or [block.text]
    units: list[Unit] = []
    current: list[str] = []
    for item in items:
        if estimate_tokens(item) > max_tokens:
            if current:
                units.append(Unit("\n".join(current), "list", block.links, block.line_start, block.line_end))
                current = []
            units.extend(
                Unit(piece, "list", block.links, block.line_start, block.line_end, continued=continued)
                for piece, continued in _split_nested_item(item, max_tokens)
            )
            continue
        if current and estimate_tokens("\n".join(current + [item])) > max_tokens:
            units.append(Unit("\n".join(current), "list", block.links, block.line_start, block.line_end))
            current = []
        current.append(item)
    if current:
        units.append(Unit("\n".join(current), "list", block.links, block.line_start, block.line_end))
    return units


def block_to_units(block: Block, max_tokens: int) -> list[Unit]:
    if estimate_tokens(block.text) <= max_tokens:
        return [Unit(block.text, block.kind, block.links, block.line_start, block.line_end)]
    if block.kind == "table" and block.table and block.table.row_lines:
        return _table_units(block, max_tokens)
    if block.kind == "list":
        return _list_units(block, max_tokens)
    return [
        Unit(piece, block.kind, block.links, block.line_start, block.line_end)
        for piece in _split_prose(block.text, max_tokens)
    ]


def _overlap_tail(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    sentences = [s for s in _SENTENCE_RE.split(text) if s and s.strip()]
    tail: list[str] = []
    for sentence in reversed(sentences):
        if estimate_tokens(" ".join([sentence] + tail)) > overlap_tokens:
            break
        tail.insert(0, sentence)
    # No partial sentences: "Some states require…" cut to "states require…" changes the meaning.
    # When the last sentence alone is longer than the overlap budget, there is no overlap.
    return " ".join(tail)


# --------------------------------------------------------------------------- chunking
@dataclass
class _Draft:
    heading_path: tuple[str, ...]
    parts: list[str]
    units: list[Unit]
    section_paths: list[tuple[str, ...]]
    overlap_tokens: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(p for p in self.parts if p.strip())

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


def _sections(parsed: ParsedDocument) -> list[Section]:
    sections: list[Section] = [Section(heading_path=(), heading=None)]
    for block in parsed.blocks:
        if block.kind == "heading":
            sections.append(Section(heading_path=block.heading_path, heading=block))
        else:
            sections[-1].blocks.append(block)
    return [s for s in sections if s.heading is not None or s.blocks]


def _merged_path(earlier: "_Draft", later: "_Draft") -> tuple[str, ...]:
    """Heading path for two merged drafts: the deeper path when one section
    contains the other; otherwise the path of the draft with more content.
    All contributing paths are kept in ``metadata.section_paths``."""
    pa, pb = earlier.heading_path, later.heading_path
    if pb[: len(pa)] == pa:
        return pb
    if pa[: len(pb)] == pb:
        return pa
    if not earlier.units:
        return pb
    if not later.units:
        return pa
    return pa if earlier.tokens > later.tokens else pb


def chunk_document(
    parsed: ParsedDocument,
    document_id: str,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chunk_tokens: int = 25,
) -> list[Chunk]:
    drafts: list[_Draft] = []
    for section in _sections(parsed):
        heading_text = section.heading.text if section.heading else ""
        heading_tokens = estimate_tokens(heading_text)
        # Leave room for the repeated heading and any overlap so chunks stay within chunk_size.
        budget = max(32, chunk_size - heading_tokens - chunk_overlap)
        units = [u for block in section.blocks for u in block_to_units(block, budget)]
        if not units:
            drafts.append(_Draft(section.heading_path, [heading_text], [], [section.heading_path]))
            continue
        current = _Draft(section.heading_path, [heading_text] if heading_text else [], [], [section.heading_path])
        for unit in units:
            if current.units and estimate_tokens(current.text + "\n\n" + unit.text) > chunk_size:
                drafts.append(current)
                previous = current.units[-1]
                overlap = ""
                # Overlap is for prose only: a list part either starts a new self-contained item or
                # (when continued) repeats its parent steps instead.
                if chunk_overlap and previous.kind not in ("table", "code", "list") and unit.kind != "table":
                    overlap = _overlap_tail(previous.text, chunk_overlap)
                parts = [heading_text] if heading_text else []
                if overlap:
                    parts.append(overlap)
                current = _Draft(section.heading_path, parts, [], [section.heading_path],
                                 overlap_tokens=estimate_tokens(overlap))
            current.parts.append(unit.text)
            current.units.append(unit)
        drafts.append(current)

    drafts = _merge_small(drafts, chunk_size, min_chunk_tokens)

    chunks: list[Chunk] = []
    for draft in drafts:
        text = draft.text.strip()
        if not text:
            continue
        index = len(chunks)
        heading_path = list(draft.heading_path)
        embedding_input = build_embedding_input(parsed.title, heading_path, text)
        content_hash = hashlib.sha256(embedding_input.encode("utf-8")).hexdigest()
        links = [link for u in draft.units for link in u.links]
        external = list(dict.fromkeys(link for link in links if link.lower().startswith(("http://", "https://"))))
        internal = list(dict.fromkeys(link for link in links if link not in external))
        table_parts = [list(u.table_part) for u in draft.units if u.table_part and u.table_part[1] > 1]
        metadata = {
            "block_kinds": sorted({u.kind for u in draft.units} or {"heading"}),
            "line_start": min((u.line_start for u in draft.units), default=None),
            "line_end": max((u.line_end for u in draft.units), default=None),
            "section_paths": [list(p) for p in dict.fromkeys(draft.section_paths)],
            "has_table": any(u.kind == "table" for u in draft.units),
            "overlap_tokens": draft.overlap_tokens,
            "chunker_version": CHUNKER_VERSION,
        }
        if table_parts:
            metadata["table_parts"] = table_parts
        if internal:
            metadata["internal_links"] = internal
        chunks.append(
            Chunk(
                document_id=document_id,
                chunk_index=index,
                text=text,
                heading_path=heading_path,
                source_urls=external,
                metadata=metadata,
                embedding_input=embedding_input,
                content_hash=content_hash,
            )
        )
    return chunks


def _merge_small(drafts: list[_Draft], chunk_size: int, min_tokens: int) -> list[_Draft]:
    """Merge undersized drafts (e.g. heading-only sections) into the next draft,
    or into the previous one when they are last."""
    if not drafts:
        return drafts
    merged: list[_Draft] = []
    pending: _Draft | None = None
    for draft in drafts:
        if pending is not None:
            combined_tokens = estimate_tokens(pending.text + "\n\n" + draft.text)
            if combined_tokens <= chunk_size or not pending.units:
                draft = _Draft(
                    heading_path=_merged_path(pending, draft),
                    parts=pending.parts + draft.parts,
                    units=pending.units + draft.units,
                    section_paths=pending.section_paths + draft.section_paths,
                    overlap_tokens=pending.overlap_tokens,
                )
            else:
                merged.append(pending)
            pending = None
        if draft.tokens < min_tokens or not draft.units:
            pending = draft
        else:
            merged.append(draft)
    if pending is not None:
        if merged and (estimate_tokens(merged[-1].text + "\n\n" + pending.text) <= chunk_size or not pending.units):
            last = merged[-1]
            merged[-1] = _Draft(
                heading_path=_merged_path(last, pending),
                parts=last.parts + pending.parts,
                units=last.units + pending.units,
                section_paths=last.section_paths + pending.section_paths,
                overlap_tokens=last.overlap_tokens,
            )
        elif pending.units:
            merged.append(pending)
    return merged
