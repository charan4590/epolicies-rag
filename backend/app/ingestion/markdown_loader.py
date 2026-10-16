"""Markdown discovery, reading, hashing, stable identifiers and safe ZIP import.

Original source files are only ever *read*; nothing here writes to the
configured MARKDOWN_DIR except :func:`import_zip`, which extracts into a new
sub-directory and never overwrites differing files unless asked to.
"""

from __future__ import annotations

import hashlib
import os
import stat
import unicodedata
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

MARKDOWN_SUFFIXES = {".md", ".markdown"}
_FALLBACK_ENCODINGS = ("cp1252", "latin-1")


class SourceDirectoryError(RuntimeError):
    """The configured Markdown directory is missing or unusable."""


class DocumentReadError(RuntimeError):
    """A single document could not be read; ingestion records and continues."""


@dataclass(frozen=True)
class SourceFile:
    relative_path: str  # normalised POSIX path relative to the source root (NFC)
    absolute_path: Path
    size_bytes: int
    modified_at: datetime

    @property
    def document_id(self) -> str:
        return make_document_id(self.relative_path)

    @property
    def filename(self) -> str:
        return PurePosixPath(self.relative_path).name


@dataclass
class LoadedDocument:
    source: SourceFile
    text: str
    encoding: str
    content_hash: str
    warnings: list[str] = field(default_factory=list)

    @property
    def document_id(self) -> str:
        return self.source.document_id

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_relative_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    posix = PurePosixPath(*rel.parts).as_posix()
    return unicodedata.normalize("NFC", posix)


def make_document_id(relative_path: str) -> str:
    """Stable ID derived from the normalised relative path (not the filename alone,
    so identically named files in different directories never collide)."""
    normalised = unicodedata.normalize("NFC", relative_path.replace("\\", "/").removeprefix("./"))
    return "doc_" + hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:24]


def validate_source_dir(root: Path) -> Path:
    root = root.expanduser()
    if not root.exists():
        raise SourceDirectoryError(
            f"MARKDOWN_DIR does not exist: {root}. Create it and copy your .md files there, "
            "or point MARKDOWN_DIR at the existing folder (absolute paths are supported)."
        )
    if not root.is_dir():
        raise SourceDirectoryError(f"MARKDOWN_DIR is not a directory: {root}")
    if not os.access(root, os.R_OK | os.X_OK):
        raise SourceDirectoryError(f"MARKDOWN_DIR is not readable by this user: {root}")
    return root.resolve()


def discover_markdown_files(root: Path) -> list[SourceFile]:
    """Recursively find Markdown files under ``root`` (sorted, deterministic).

    Hidden directories/files are skipped. Symlinks are followed only when they
    resolve inside ``root`` (prevents escaping the configured directory).
    """
    root = validate_source_dir(root)
    found: list[SourceFile] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__MACOSX")
        for name in sorted(filenames):
            if name.startswith(".") or Path(name).suffix.lower() not in MARKDOWN_SUFFIXES:
                continue
            path = Path(dirpath) / name
            try:
                if path.is_symlink():
                    target = path.resolve()
                    if root not in target.parents:
                        continue
                info = path.stat()
            except OSError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            found.append(
                SourceFile(
                    relative_path=normalize_relative_path(path, root),
                    absolute_path=path,
                    size_bytes=info.st_size,
                    modified_at=datetime.fromtimestamp(info.st_mtime, tz=timezone.utc),
                )
            )
    found.sort(key=lambda s: s.relative_path)
    return found


def decode_markdown_bytes(raw: bytes) -> tuple[str, str, list[str]]:
    """Decode bytes as UTF-8 (BOM tolerated); fall back to cp1252/latin-1 with a warning."""
    warnings: list[str] = []
    if b"\x00" in raw:
        raise DocumentReadError("file contains NUL bytes and appears to be binary, not Markdown")
    try:
        text = raw.decode("utf-8-sig")
        encoding = "utf-8"
    except UnicodeDecodeError:
        for candidate in _FALLBACK_ENCODINGS:
            try:
                text = raw.decode(candidate)
                encoding = candidate
                warnings.append(f"not valid UTF-8; decoded as {candidate}")
                break
            except UnicodeDecodeError:
                continue
        else:  # pragma: no cover - latin-1 decodes any byte sequence
            raise DocumentReadError("could not decode file as UTF-8, cp1252 or latin-1")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    return text, encoding, warnings


def load_document(source: SourceFile, max_bytes: int) -> LoadedDocument:
    if source.size_bytes > max_bytes:
        raise DocumentReadError(
            f"file is {source.size_bytes} bytes, above MAX_MARKDOWN_FILE_BYTES={max_bytes}"
        )
    try:
        raw = source.absolute_path.read_bytes()
    except PermissionError as exc:
        raise DocumentReadError("permission denied while reading file") from exc
    except OSError as exc:
        raise DocumentReadError(f"could not read file: {exc.strerror or exc}") from exc
    text, encoding, warnings = decode_markdown_bytes(raw)
    return LoadedDocument(
        source=source,
        text=text,
        encoding=encoding,
        content_hash=sha256_text(text),
        warnings=warnings,
    )


# --------------------------------------------------------------------------- ZIP
@dataclass
class ZipImportReport:
    destination: str
    extracted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped_non_markdown: int = 0
    conflicts: list[str] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)


MAX_ZIP_MEMBERS = 20_000
MAX_ZIP_TOTAL_BYTES = 500 * 1024 * 1024


def _safe_member_path(name: str) -> PurePosixPath | None:
    """Validate an archive member name; return a safe relative path or None."""
    if not name or "\x00" in name:
        return None
    normalised = name.replace("\\", "/")
    if normalised.startswith("/") or (len(normalised) > 1 and normalised[1] == ":"):
        return None  # absolute path or Windows drive letter
    parts = PurePosixPath(normalised).parts
    if any(part in ("..", "") for part in parts):
        return None
    return PurePosixPath(*parts)


def import_zip(
    zip_path: Path,
    markdown_root: Path,
    *,
    target_subdir: str | None = None,
    overwrite: bool = False,
    max_file_bytes: int = 5 * 1024 * 1024,
) -> ZipImportReport:
    """Extract Markdown members of ``zip_path`` into ``markdown_root/<target_subdir>``.

    - rejects absolute paths, ``..`` traversal, symlinks and oversized members
    - preserves directory structure
    - never overwrites an existing file with different content unless ``overwrite``
    """
    markdown_root = validate_source_dir(markdown_root)
    subdir = target_subdir or zip_path.stem
    safe_subdir = _safe_member_path(subdir)
    if safe_subdir is None or len(safe_subdir.parts) != 1:
        raise ValueError("target_subdir must be a single directory name")
    destination = (markdown_root / safe_subdir.as_posix()).resolve()
    if markdown_root not in destination.parents:
        raise ValueError("target directory escapes MARKDOWN_DIR")
    report = ZipImportReport(destination=str(destination))

    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        if len(members) > MAX_ZIP_MEMBERS:
            raise ValueError(f"archive has {len(members)} members; limit is {MAX_ZIP_MEMBERS}")
        if sum(m.file_size for m in members) > MAX_ZIP_TOTAL_BYTES:
            raise ValueError("archive uncompressed size exceeds the 500 MB safety limit")
        for member in members:
            if member.is_dir():
                continue
            rel = _safe_member_path(member.filename)
            if rel is None:
                report.rejected.append({"member": member.filename, "reason": "unsafe path"})
                continue
            if rel.parts[0] == "__MACOSX" or any(p.startswith(".") for p in rel.parts):
                continue
            if rel.suffix.lower() not in MARKDOWN_SUFFIXES:
                report.skipped_non_markdown += 1
                continue
            mode = (member.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                report.rejected.append({"member": member.filename, "reason": "symlinks are not allowed"})
                continue
            if member.file_size > max_file_bytes:
                report.rejected.append({"member": member.filename, "reason": "file too large"})
                continue
            target = (destination / rel.as_posix()).resolve()
            if destination not in target.parents:
                report.rejected.append({"member": member.filename, "reason": "path traversal"})
                continue
            data = archive.read(member)
            rel_display = f"{safe_subdir.as_posix()}/{rel.as_posix()}"
            if target.exists():
                if target.read_bytes() == data:
                    report.unchanged.append(rel_display)
                    continue
                if not overwrite:
                    report.conflicts.append(rel_display)
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)
            report.extracted.append(rel_display)
    return report
