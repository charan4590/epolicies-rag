"""Discovery, encoding, identity, ZIP import, parsing and chunking."""

from __future__ import annotations

import io
import stat
import zipfile
from pathlib import Path

import pytest

from app.ingestion.chunker import chunk_document, estimate_tokens, make_chunk_id
from app.ingestion.markdown_loader import (
    DocumentReadError,
    SourceDirectoryError,
    decode_markdown_bytes,
    discover_markdown_files,
    import_zip,
    load_document,
    make_document_id,
)
from app.ingestion.markdown_parser import extract_bare_urls, parse_markdown, split_front_matter
from tests.conftest import FIXTURES


# --------------------------------------------------------------------------- discovery
def test_discovery_is_recursive_sorted_and_skips_hidden(tmp_path: Path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "a" / "b" / "deep.md").write_text("# Deep")
    (tmp_path / "top.MD").write_text("# Top")
    (tmp_path / "notes.markdown").write_text("# Notes")
    (tmp_path / "ignore.txt").write_text("not markdown")
    (tmp_path / ".hidden" / "secret.md").write_text("# hidden")
    (tmp_path / ".dotfile.md").write_text("# dot")
    found = [f.relative_path for f in discover_markdown_files(tmp_path)]
    assert found == ["a/b/deep.md", "notes.markdown", "top.MD"]


def test_discovery_of_fixtures_finds_nested_files():
    paths = [f.relative_path for f in discover_markdown_files(FIXTURES)]
    assert "hr/remote-work-policy.md" in paths and "it/remote-work-policy.md" in paths
    assert "finance/archive/travel-expenses-copy.md" in paths
    assert len(paths) == 9


def test_missing_directory_has_actionable_error(tmp_path: Path):
    with pytest.raises(SourceDirectoryError, match="MARKDOWN_DIR does not exist"):
        discover_markdown_files(tmp_path / "nope")


def test_symlink_outside_root_is_ignored(tmp_path: Path):
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link.md").symlink_to(outside)
    (root / "real.md").write_text("# Real")
    assert [f.relative_path for f in discover_markdown_files(root)] == ["real.md"]


# --------------------------------------------------------------------------- identity & encoding
def test_document_ids_are_stable_and_path_based():
    assert make_document_id("hr/remote-work-policy.md") == make_document_id("hr/remote-work-policy.md")
    assert make_document_id("hr/remote-work-policy.md") != make_document_id("it/remote-work-policy.md")
    assert make_document_id("./hr/x.md") == make_document_id("hr/x.md")
    assert make_document_id("hr\\x.md") == make_document_id("hr/x.md")
    assert make_document_id("a.md").startswith("doc_") and len(make_document_id("a.md")) == 28


def test_decode_utf8_with_bom_and_crlf():
    text, encoding, warnings = decode_markdown_bytes("﻿# Title\r\nLine\r\n".encode("utf-8"))
    assert text == "# Title\nLine\n" and encoding == "utf-8" and warnings == []


def test_decode_falls_back_to_cp1252_with_warning():
    text, encoding, warnings = decode_markdown_bytes("Café “quoted”".encode("cp1252"))
    assert text == "Café “quoted”" and encoding == "cp1252" and warnings


def test_binary_file_rejected():
    with pytest.raises(DocumentReadError, match="binary"):
        decode_markdown_bytes(b"\x89PNG\x00\x00")


def test_content_hash_ignores_line_endings_and_file_size_limit(tmp_path: Path):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_bytes(b"# T\nbody\n")
    b.write_bytes(b"# T\r\nbody\r\n")
    sources = {s.relative_path: s for s in discover_markdown_files(tmp_path)}
    assert load_document(sources["a.md"], 1000).content_hash == load_document(sources["b.md"], 1000).content_hash
    with pytest.raises(DocumentReadError, match="above MAX_MARKDOWN_FILE_BYTES"):
        load_document(sources["a.md"], 3)


# --------------------------------------------------------------------------- ZIP import
def _zip(entries: dict[str, bytes], symlink: str | None = None) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "/etc/passwd")
    buf.seek(0)
    return buf


def test_zip_import_preserves_structure_and_rejects_traversal(tmp_path: Path):
    root = tmp_path / "kb"
    root.mkdir()
    archive = tmp_path / "policies.zip"
    archive.write_bytes(_zip({
        "hr/leave.md": b"# Leave",
        "it/sec/aup.md": b"# AUP",
        "../escape.md": b"# bad",
        "/abs.md": b"# bad",
        "__MACOSX/hr/._leave.md": b"junk",
        "readme.txt": b"not md",
    }, symlink="hr/link.md").getvalue())
    report = import_zip(archive, root)
    assert sorted(report.extracted) == ["policies/hr/leave.md", "policies/it/sec/aup.md"]
    assert (root / "policies" / "it" / "sec" / "aup.md").read_text() == "# AUP"
    reasons = {r["member"]: r["reason"] for r in report.rejected}
    assert reasons["../escape.md"] == "unsafe path" and reasons["/abs.md"] == "unsafe path"
    assert "symlinks" in reasons["hr/link.md"]
    assert report.skipped_non_markdown == 1
    assert not (tmp_path / "escape.md").exists()


def test_zip_import_never_overwrites_without_flag(tmp_path: Path):
    root = tmp_path / "kb"
    (root / "batch").mkdir(parents=True)
    (root / "batch" / "a.md").write_text("original")
    archive = tmp_path / "a.zip"
    archive.write_bytes(_zip({"a.md": b"changed", "b.md": b"new"}).getvalue())
    report = import_zip(archive, root, target_subdir="batch")
    assert report.conflicts == ["batch/a.md"] and report.extracted == ["batch/b.md"]
    assert (root / "batch" / "a.md").read_text() == "original"
    report = import_zip(archive, root, target_subdir="batch", overwrite=True)
    assert (root / "batch" / "a.md").read_text() == "changed"
    with pytest.raises(ValueError):
        import_zip(archive, root, target_subdir="../x")


# --------------------------------------------------------------------------- parsing
SAMPLE = """---
title: Sample Policy
source_url: https://policies.example.com/sample
---

# Heading One

Intro with a [link](https://example.com/a), an autolink <https://auto.example.com> and
a bare URL https://bare.example.org/path. plus a [relative](other.md) link.

## Section A

| Col 1 | Col 2 |
|---|---|
| a | b |

### Sub A.1

- item one
  - nested item
- item two

## Section B

Text B.
"""


def test_front_matter_and_title():
    meta, body, consumed = split_front_matter(SAMPLE)
    assert meta["title"] == "Sample Policy" and consumed == 4 and body.startswith("\n# Heading One")
    parsed = parse_markdown(SAMPLE, "dir/sample.md")
    assert parsed.title == "Sample Policy"
    assert parsed.source_url == "https://policies.example.com/sample"


def test_title_fallbacks():
    assert parse_markdown("## Only H2\n\ntext", "x.md").title == "Only H2"
    assert parse_markdown("# Main\n\n## Sub", "x.md").title == "Main"
    assert parse_markdown("plain text", "dir/annual_leave-guide.md").title == "annual leave guide"


def test_heading_hierarchy_and_blocks():
    parsed = parse_markdown(SAMPLE, "sample.md")
    assert parsed.headings == [(1, "Heading One"), (2, "Section A"), (3, "Sub A.1"), (2, "Section B")]
    table = next(b for b in parsed.blocks if b.kind == "table")
    assert table.heading_path == ("Heading One", "Section A")
    assert table.table.header_lines == ["| Col 1 | Col 2 |", "|---|---|"]
    lst = next(b for b in parsed.blocks if b.kind == "list")
    assert lst.heading_path == ("Heading One", "Section A", "Sub A.1")
    assert len(lst.list_items) == 2 and "nested item" in lst.list_items[0]
    para_b = [b for b in parsed.blocks if b.kind == "paragraph"][-1]
    assert para_b.heading_path == ("Heading One", "Section B")


def test_url_extraction():
    parsed = parse_markdown(SAMPLE, "sample.md")
    assert parsed.urls == ["https://example.com/a", "https://auto.example.com", "https://bare.example.org/path"]
    assert parsed.internal_links == ["other.md"]
    assert extract_bare_urls("see (https://x.example.com/a).") == ["https://x.example.com/a"]


def test_raw_html_is_kept_as_inert_text():
    parsed = parse_markdown("# T\n\n<script>alert(1)</script>\n\nAfter.", "x.md")
    assert any("<script>" in b.text for b in parsed.blocks)


# --------------------------------------------------------------------------- chunking
def _chunks(text: str, **kw):
    kw.setdefault("chunk_size", 512)
    kw.setdefault("chunk_overlap", 64)
    kw.setdefault("min_chunk_tokens", 40)
    return chunk_document(parse_markdown(text, "doc.md"), "doc_test", **kw)


def test_chunks_carry_heading_paths_urls_and_order():
    chunks = _chunks(SAMPLE, min_chunk_tokens=0)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.text.strip() for c in chunks)
    assert chunks[0].heading_path == ["Heading One"]
    assert "https://example.com/a" in chunks[0].source_urls
    assert chunks[0].metadata["internal_links"] == ["other.md"]
    table_chunk = next(c for c in chunks if c.metadata["has_table"])
    assert "| Col 1 | Col 2 |" in table_chunk.text and table_chunk.heading_path[:2] == ["Heading One", "Section A"]


def test_small_sections_are_merged_and_heading_only_sections_never_stand_alone():
    chunks = _chunks("# Title\n\n## Empty heading\n\n### Child\n\nSome text about the child section here.")
    assert len(chunks) == 1
    assert chunks[0].heading_path == ["Title", "Empty heading", "Child"]
    assert "## Empty heading" in chunks[0].text


def test_large_table_split_repeats_header():
    rows = "\n".join(f"| R{i} | value number {i} with some descriptive words |" for i in range(200))
    text = f"# T\n\n## Big table\n\n| ID | Description |\n|---|---|\n{rows}\n"
    chunks = _chunks(text, chunk_size=300, chunk_overlap=30)
    table_chunks = [c for c in chunks if c.metadata["has_table"]]
    assert len(table_chunks) > 3
    for c in table_chunks:
        assert "| ID | Description |\n|---|---|" in c.text
        assert c.token_count <= 300
    all_rows = [line for c in table_chunks for line in c.text.splitlines() if line.startswith("| R")]
    assert len(all_rows) == 200 and len(set(all_rows)) == 200  # every row exactly once, in order
    assert all_rows[0].startswith("| R0 ") and all_rows[-1].startswith("| R199 ")
    assert table_chunks[0].metadata["table_parts"][0][0] == 1


def test_long_prose_split_with_overlap_and_size_bound():
    sentences = " ".join(f"Sentence number {i} explains an important rule about expenses." for i in range(120))
    chunks = _chunks(f"# T\n\n## Long\n\n{sentences}", chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 3
    assert all(c.token_count <= 200 for c in chunks)
    assert all(c.heading_path == ["T", "Long"] for c in chunks)
    assert chunks[1].metadata["overlap_tokens"] > 0
    tail = chunks[0].text.split(". ")[-1][:30]
    assert tail.strip() and tail.strip() in chunks[1].text


def test_long_list_split_between_items():
    items = "\n".join(f"- Item {i}: employees must follow requirement number {i} at all times." for i in range(80))
    chunks = _chunks(f"# T\n\n## Rules\n\n{items}", chunk_size=200, chunk_overlap=0)
    assert len(chunks) > 2
    for c in chunks:
        for line in c.text.splitlines():
            assert not line.strip() or line.startswith(("- Item", "## Rules", "# T"))


def test_chunk_ids_deterministic_and_content_sensitive():
    a1 = _chunks(SAMPLE)
    a2 = _chunks(SAMPLE)
    assert [c.chunk_id for c in a1] == [c.chunk_id for c in a2]
    b = _chunks(SAMPLE.replace("Text B.", "Text B changed."))
    assert a1[-1].chunk_id != b[-1].chunk_id and a1[0].chunk_id == b[0].chunk_id
    assert make_chunk_id("d", 0, "h") != make_chunk_id("d", 1, "h")


def test_embedding_input_includes_context_and_hash_matches():
    import hashlib

    chunk = _chunks(SAMPLE)[0]
    assert chunk.embedding_input.startswith("Document: Sample Policy\nSection: Heading One")
    assert chunk.content_hash == hashlib.sha256(chunk.embedding_input.encode()).hexdigest()


def test_empty_document_has_no_chunks():
    assert _chunks("") == [] and _chunks("---\ntitle: x\n---\n\n") == []


def test_token_estimate():
    assert estimate_tokens("") == 0
    assert 20 <= estimate_tokens("word " * 20) <= 30
