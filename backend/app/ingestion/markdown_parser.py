"""Structure-preserving Markdown parsing.

Uses markdown-it-py to identify top-level blocks, then slices the *original
source lines* for each block so tables, lists and links reach the chunker (and
the LLM) exactly as written.  Raw HTML is never rendered or executed — it is
retained as inert text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from markdown_it import MarkdownIt
from markdown_it.token import Token

_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"'`|]+", re.IGNORECASE)
_TRAILING_PUNCT = ".,;:!?*_~"
_FRONT_MATTER_KEYS_FOR_URL = ("source_url", "url", "source", "link", "canonical_url")

# maxNesting: ePolicies step tables become lists nested ~20 levels deep. markdown-it's default limit (20)
# silently stops parsing structure there, so every later heading became part of that list.
_md = MarkdownIt("commonmark", {"html": False, "maxNesting": 100}).enable(["table", "strikethrough"])


@dataclass
class TableData:
    header_lines: list[str]  # header row + delimiter row
    row_lines: list[str]


@dataclass
class Block:
    kind: str  # heading | paragraph | list | table | code | blockquote | html | hr
    text: str
    heading_path: tuple[str, ...]
    line_start: int
    line_end: int
    level: int | None = None
    links: list[str] = field(default_factory=list)
    table: TableData | None = None
    list_items: list[str] | None = None

    @property
    def urls(self) -> list[str]:
        return [u for u in self.links if is_external_url(u)]


@dataclass
class ParsedDocument:
    title: str
    front_matter: dict[str, str]
    source_url: str | None
    blocks: list[Block]
    headings: list[tuple[int, str]]

    @property
    def urls(self) -> list[str]:
        return _dedupe(u for b in self.blocks for u in b.urls)

    @property
    def internal_links(self) -> list[str]:
        return _dedupe(u for b in self.blocks for u in b.links if not is_external_url(u))


def is_external_url(url: str) -> bool:
    return url.lower().startswith(("http://", "https://"))


def _dedupe(items) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


def clean_url(url: str) -> str:
    url = url.strip()
    while url and url[-1] in _TRAILING_PUNCT:
        url = url[:-1]
    # Balance a trailing ')' that belongs to surrounding prose.
    if url.endswith(")") and url.count("(") < url.count(")"):
        url = url[:-1]
    return url


def extract_bare_urls(text: str) -> list[str]:
    return [clean_url(m.group(0)) for m in _URL_RE.finditer(text)]


def split_front_matter(text: str) -> tuple[dict[str, str], str, int]:
    """Parse a simple YAML-style front matter block (``key: value`` lines only).

    Returns (front_matter, body, number_of_lines_consumed).
    """
    if not text.startswith("---\n"):
        return {}, text, 0
    lines = text.split("\n")
    for idx in range(1, min(len(lines), 200)):
        if lines[idx].strip() in ("---", "..."):
            meta: dict[str, str] = {}
            for raw in lines[1:idx]:
                if ":" not in raw or raw.startswith((" ", "\t", "-")):
                    continue
                key, _, value = raw.partition(":")
                key = key.strip().lower()
                value = value.strip().strip("'\"")
                if key and value:
                    meta[key] = value
            return meta, "\n".join(lines[idx + 1 :]), idx + 1
    return {}, text, 0


def _inline_text(token: Token | None) -> str:
    if token is None or not token.children:
        return token.content.strip() if token else ""
    parts: list[str] = []
    for child in token.children:
        if child.type in ("text", "code_inline"):
            parts.append(child.content)
        elif child.type in ("softbreak", "hardbreak"):
            parts.append(" ")
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _collect_links(tokens: list[Token]) -> list[str]:
    links: list[str] = []
    for token in tokens:
        for child in token.children or []:
            if child.type == "link_open":
                href = str(child.attrs.get("href", "")).strip()
                if href and not href.startswith("#"):
                    links.append(clean_url(href) if is_external_url(href) else href)
    return links


def _title_from_filename(relative_path: str) -> str:
    stem = PurePosixPath(relative_path).stem
    return re.sub(r"[_\-]+", " ", stem).strip() or stem


def parse_markdown(text: str, relative_path: str) -> ParsedDocument:
    front_matter, body, offset = split_front_matter(text)
    lines = body.split("\n")
    tokens = _md.parse(body)

    blocks: list[Block] = []
    headings: list[tuple[int, str]] = []
    heading_stack: list[tuple[int, str]] = []

    def source(start: int, end: int) -> str:
        return "\n".join(lines[start:end]).strip("\n")

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.level != 0 or tok.map is None or tok.nesting == -1:
            i += 1
            continue
        start, end = tok.map
        # Find the matching close token so we can inspect the block's children.
        j = i
        if tok.nesting == 1:
            depth = 0
            while j < len(tokens):
                if tokens[j].level == 0 and tokens[j].nesting == 1:
                    depth += 1
                elif tokens[j].level == 0 and tokens[j].nesting == -1:
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
        inner = tokens[i : j + 1]
        inline_tokens = [t for t in inner if t.type == "inline"]
        raw = source(start, end)
        links = _dedupe(_collect_links(inline_tokens) + extract_bare_urls(raw))

        if tok.type == "heading_open":
            level = int(tok.tag[1])
            heading_text = _inline_text(inline_tokens[0] if inline_tokens else None) or raw.lstrip("# ").strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            headings.append((level, heading_text))
            blocks.append(
                Block("heading", raw, tuple(h for _, h in heading_stack), start + offset, end + offset,
                      level=level, links=links)
            )
        else:
            path = tuple(h for _, h in heading_stack)
            kind = {
                "paragraph_open": "paragraph",
                "bullet_list_open": "list",
                "ordered_list_open": "list",
                "table_open": "table",
                "blockquote_open": "blockquote",
                "fence": "code",
                "code_block": "code",
                "html_block": "html",
                "hr": "hr",
            }.get(tok.type, "paragraph")
            if kind == "hr" or not raw.strip():
                i = j + 1
                continue
            block = Block(kind, raw, path, start + offset, end + offset, links=links)
            if kind == "table":
                rows = [t.map for t in inner if t.type == "tr_open" and t.map]
                if rows:
                    header_end = rows[0][1]
                    # Header row plus the delimiter row that follows it.
                    block.table = TableData(
                        header_lines=lines[start : header_end + 1],
                        row_lines=[source(r[0], r[1]) for r in rows[1:]],
                    )
            elif kind == "list":
                items = [t.map for t in inner if t.type == "list_item_open" and t.level == 1 and t.map]
                block.list_items = [source(a, b) for a, b in items if source(a, b).strip()]
            blocks.append(block)
        i = j + 1

    title = front_matter.get("title") or next((h for lvl, h in headings if lvl == 1), None)
    title = title or (headings[0][1] if headings else None) or _title_from_filename(relative_path)
    source_url = next(
        (clean_url(front_matter[k]) for k in _FRONT_MATTER_KEYS_FOR_URL
         if k in front_matter and is_external_url(front_matter[k])),
        None,
    )
    return ParsedDocument(
        title=title.strip(),
        front_matter=front_matter,
        source_url=source_url,
        blocks=blocks,
        headings=headings,
    )
