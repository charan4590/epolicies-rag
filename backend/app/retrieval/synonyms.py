"""Synonym and acronym expansion for keyword search.

Vector search already matches meaning ("termination of pregnancy" finds the Abortion policy), but
keyword search only matches words, so two wordings of the same question could retrieve different
extra passages. Before a keyword search, any known equivalent terms found in the query are added:

* a small built-in list of common policy synonyms,
* optional groups from ``SYNONYMS_FILE`` (JSON list of lists of terms),
* acronyms defined in the indexed policies themselves – "Coordination Of Benefits (COB)" in the text or
  "| MSA | Medicare Set Aside |" in an acronym table – learned and cached until the index changes.

Only keyword search is expanded; the question itself and vector search are unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

import psycopg

logger = logging.getLogger(__name__)

BUILTIN_GROUPS: tuple[tuple[str, ...], ...] = (
    ("abortion", "termination of pregnancy"),
    ("precertification", "precert", "prior authorization", "preauthorization"),
    ("department", "dept"),
    ("non-participating", "out-of-network"),
    ("joint venture", "joint partnership"),
)
_SMALL_WORDS = {"of", "and", "the", "for", "to", "in", "on", "a", "an", "&"}
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "dc": "district of columbia", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland", "ma": "massachusetts",
    "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "pr": "puerto rico", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia",
    "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
}
_ACRONYM_SQL = r"""
    SELECT m[1] AS before, m[2] AS acronym
    FROM chunks c, regexp_matches(c.text, '((?:[A-Za-z][A-Za-z/-]*\s+){1,8})\(([A-Z][A-Z0-9&]{1,7})\)', 'g') AS m
"""
# Acronym tables: "| MSA | Medicare Set Aside |"
_ACRONYM_TABLE_SQL = r"""
    SELECT m[2] AS before, m[1] AS acronym
    FROM chunks c, regexp_matches(c.text, '\|\s*([A-Z][A-Z0-9&]{1,7})\s*\|\s*([A-Za-z][^|\n]{2,80}?)\s*\|', 'g') AS m
"""


def _norm(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower())


def acronym_from(words_before: str, acronym: str) -> str | None:
    """The phrase whose initials spell ``acronym`` ("Order of Benefit Determination", "OBD"), or None."""
    words = words_before.split()
    letters = acronym.replace("&", "").lower()
    for start in range(len(words)):
        phrase = words[start:]
        if not phrase or phrase[0].lower() in _SMALL_WORDS:
            continue
        initials_all = "".join(w[0].lower() for w in phrase)
        initials_main = "".join(w[0].lower() for w in phrase if w.lower() not in _SMALL_WORDS)
        if letters in (initials_all, initials_main):
            return " ".join(phrase)
    return None


def load_file_groups(path: Path | None) -> list[tuple[str, ...]]:
    if path is None or not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [tuple(str(t) for t in group if str(t).strip()) for group in data if len(group) > 1]
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("synonyms_file_invalid", extra={"path": str(path), "error": str(exc)})
        return []


def _meaning(phrase: str) -> str:
    """'Coordination of Benefits' and 'coordination of benefit' are the same meaning."""
    return " ".join(w[:-1] if w.endswith("s") and len(w) > 3 else w for w in phrase.split() if w not in _SMALL_WORDS)


class SynonymIndex:
    """Equivalent-term groups (lower-cased); acronyms are refreshed when the index changes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._signature: tuple | None = None
        self._acronym_groups: list[tuple[str, ...]] = []

    def acronym_groups(self, conn: psycopg.Connection) -> list[tuple[str, ...]]:
        signature = tuple(conn.execute("SELECT count(*) AS n, max(created_at) AS newest FROM chunks").fetchone().values())
        with self._lock:
            if signature == self._signature:
                return self._acronym_groups
        pairs: dict[str, set[str]] = {}
        rows = conn.execute(_ACRONYM_SQL).fetchall() + conn.execute(_ACRONYM_TABLE_SQL).fetchall()
        for row in rows:
            phrase = acronym_from(row["before"], row["acronym"])
            if phrase and len(row["acronym"]) >= 2:
                pairs.setdefault(row["acronym"].lower(), set()).add(_norm(phrase))
        # A state code used as a state anywhere ("Massachusetts (MA)") is ambiguous when it was also learned
        # with another meaning ("MA" = Medicare Advantage), so expanding it would search the wrong topic.
        state_codes = [acr for acr in pairs if acr in US_STATES and pairs[acr] != {US_STATES[acr]}]
        used_as_state = {
            acr for acr in state_codes
            if conn.execute("SELECT EXISTS (SELECT 1 FROM chunks WHERE text ~* %s) AS used",
                            (rf"{US_STATES[acr]}\s*\({acr}\)",)).fetchone()["used"]
        }
        # An acronym learned with different meanings is ambiguous too; only one meaning is expanded.
        groups = [(acr, *phrases) for acr, phrases in sorted(pairs.items())
                  if len({_meaning(p) for p in phrases}) == 1 and acr not in used_as_state]
        groups = [(acr, sorted(rest, key=len)[0]) for acr, *rest in groups]
        with self._lock:
            self._signature, self._acronym_groups = signature, groups
        return groups


_index = SynonymIndex()


def _contains(text: str, term: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text) is not None


def _mentions(query: str, text: str, member: str) -> bool:
    """Phrases match case-insensitively; a 2-letter acronym only in capitals ("IT", not "it")."""
    if len(member) <= 2 and " " not in member:
        return re.search(r"(?<![A-Za-z0-9])" + re.escape(member.upper()) + r"(?![A-Za-z0-9])", query) is not None
    return _contains(text, member)


def expansion_terms(query: str, groups: Iterable[Sequence[str]]) -> list[str]:
    """Equivalent terms for every group that has a member in ``query`` (members already present are skipped)."""
    text = _norm(query)
    extra: list[str] = []
    for group in groups:
        members = [_norm(t) for t in group if t.strip()]
        if any(_mentions(query, text, m) for m in members):
            extra.extend(m for m in members if not _contains(text, m) and m not in extra)
    return extra


def keyword_query_with_synonyms(conn: psycopg.Connection | None, query: str, *, file_path: Path | None = None) -> str:
    groups: list[Sequence[str]] = [*BUILTIN_GROUPS, *load_file_groups(file_path)]
    if conn is not None:
        try:
            with conn.transaction():  # savepoint: a failure here must not abort the caller's transaction
                groups.extend(_index.acronym_groups(conn))
        except psycopg.Error:
            logger.warning("acronym_index_unavailable", exc_info=True)
    extra = expansion_terms(query, groups)
    return f"{query} {' '.join(extra)}" if extra else query
