"""Keyword retrieval with PostgreSQL full-text search, plus an exact
identifier/phrase channel for terms that stemming would break apart
(e.g. ``HR-POL-012``, ``401(k)``, ``ISO 27001``)."""

from __future__ import annotations

import difflib
import re
import threading
from collections.abc import Sequence

import psycopg

from app.retrieval.types import CHUNK_COLUMNS, RetrievedChunk

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
MAX_TERMS = 32
HEADING_BOOST = 1.0  # added × share of the query terms found in the chunk's own section heading

# A small stop-list so OR-queries are not dominated by question words. PostgreSQL's
# english configuration removes standard stop words as well.
_QUESTION_WORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "does", "do", "did", "can",
    "could", "should", "would", "will", "shall", "may", "might", "must", "is", "are", "was", "were", "be",
    "the", "a", "an", "of", "to", "in", "for", "on", "at", "by", "with", "about", "tell", "me", "please",
    "explain", "describe", "our", "my", "we", "i", "you", "policy", "policies",
}


def keyword_terms(text: str) -> list[str]:
    """Alphanumeric terms suitable for an OR-combined tsquery. Only [A-Za-z0-9]
    characters survive, so tsquery operators cannot be injected."""
    terms: list[str] = []
    for word in _WORD_RE.findall(text):
        lower = word.lower()
        if lower in _QUESTION_WORDS or (len(lower) < 2 and not lower.isdigit()):
            continue
        if lower not in terms:
            terms.append(lower)
    return terms[:MAX_TERMS]


def keyword_search(
    conn: psycopg.Connection,
    query: str,
    *,
    top_k: int,
    ts_config: str = "english",
    phrases: Sequence[str] = (),
) -> list[RetrievedChunk]:
    """Rank chunks by ``ts_rank_cd`` over an OR of the query's terms (title,
    headings and body are weighted A/B/C). Exact phrase matches get a boost."""
    terms = keyword_terms(query)
    if not terms:
        return []
    tsquery = " | ".join(terms)
    phrase_list = [p for p in phrases if p.strip()][:5]
    rows = conn.execute(
        f"""
        WITH q AS (SELECT to_tsquery(%(cfg)s::regconfig, %(tsq)s) AS query)
        SELECT {CHUNK_COLUMNS},
               ts_rank_cd(c.search_tsv, q.query, 32)
               -- boost when the question matches the chunk's own section heading, so a short section
               -- ("Reimbursement for Travel for an Elective Abortion") is not buried under long ones
               + %(heading_boost)s * (
                     SELECT count(*) FROM unnest(%(terms)s::text[]) t
                     WHERE to_tsvector(%(cfg)s::regconfig, coalesce(c.heading_path[array_length(c.heading_path, 1)], ''))
                           @@ plainto_tsquery(%(cfg)s::regconfig, t)
                 )::float / %(n_terms)s
               + COALESCE((SELECT count(*) FROM unnest(%(phrases)s::text[]) p
                           WHERE c.search_tsv @@ phraseto_tsquery(%(cfg)s::regconfig, p)), 0) AS score
        FROM chunks c JOIN documents d ON d.document_id = c.document_id, q
        WHERE d.status <> 'deleted' AND numnode(q.query) > 0 AND c.search_tsv @@ q.query
        ORDER BY score DESC, c.chunk_id
        LIMIT %(k)s
        """,
        {"cfg": ts_config, "tsq": tsquery, "k": top_k, "phrases": phrase_list, "heading_boost": HEADING_BOOST,
         "terms": terms, "n_terms": len(terms)},
    ).fetchall()
    results: list[RetrievedChunk] = []
    for rank, row in enumerate(rows, start=1):
        chunk = RetrievedChunk.from_row(row)
        chunk.keyword_rank = rank
        chunk.keyword_score = float(row["score"])
        results.append(chunk)
    return results


def unknown_terms(conn: psycopg.Connection, text: str, *, ts_config: str = "english") -> list[str]:
    """Question words that occur in no indexed chunk – usually spelling mistakes ("jpoint partership").

    Each term is checked with the GIN full-text index, so this is fast. Stop words and terms that
    stem to nothing are ignored.
    """
    terms = keyword_terms(text)
    if not terms or conn.execute("SELECT NOT EXISTS (SELECT 1 FROM chunks) AS empty").fetchone()["empty"]:
        return []
    rows = conn.execute(
        """
        SELECT t.term FROM unnest(%(terms)s::text[]) WITH ORDINALITY AS t(term, pos)
        WHERE numnode(plainto_tsquery(%(cfg)s::regconfig, t.term)) > 0
          AND NOT EXISTS (SELECT 1 FROM chunks c
                          WHERE c.search_tsv @@ plainto_tsquery(%(cfg)s::regconfig, t.term))
        ORDER BY t.pos
        """,
        {"terms": terms, "cfg": ts_config},
    ).fetchall()
    return [row["term"] for row in rows]


MIN_TYPO_LENGTH = 5
TYPO_SIMILARITY = 0.8


class _Vocabulary:
    """Stemmed words of all indexed chunks (from the full-text index), cached until the index changes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._signature: tuple | None = None
        self._words: list[str] = []

    def words(self, conn: psycopg.Connection) -> list[str]:
        signature = tuple(conn.execute("SELECT count(*) AS n, max(created_at) AS newest FROM chunks").fetchone().values())
        with self._lock:
            if signature == self._signature:
                return self._words
        rows = conn.execute("SELECT word FROM ts_stat('SELECT search_tsv FROM chunks')").fetchall()
        words = sorted({r["word"] for r in rows if r["word"].isalpha() and len(r["word"]) >= 4})
        with self._lock:
            self._signature, self._words = signature, words
        return words


_vocabulary = _Vocabulary()


def _close_to_vocabulary(stem: str, words: list[str]) -> bool:
    if difflib.get_close_matches(stem, words, n=1, cutoff=TYPO_SIMILARITY):
        return True
    # Stems are short ("subrogation" -> "subrog"): compare the same length ("subrogaton" -> "subrog").
    return any(len(w) >= MIN_TYPO_LENGTH and stem[:3] == w[:3] and len(stem) > len(w)
               and difflib.SequenceMatcher(None, stem[:len(w)], w).ratio() >= 0.85 for w in words)


def likely_typos(conn: psycopg.Connection, text: str, *, ts_config: str = "english") -> list[str]:
    """Question words found in no policy that look like misspellings of a policy word.

    "jpoint", "partership", "subrogaton", "acident" count; normal words that the policies simply do not
    use ("everything", "parking", "kind"), short words and anything with digits ("486", "99203") do not,
    so they do not cost an extra Gemini call.
    """
    candidates = [t for t in unknown_terms(conn, text, ts_config=ts_config)
                  if t.isalpha() and len(t) >= MIN_TYPO_LENGTH]
    if not candidates:
        return []
    words = _vocabulary.words(conn)
    typos: list[str] = []
    for term in candidates:
        stem = conn.execute("SELECT coalesce((tsvector_to_array(to_tsvector(%s::regconfig, %s)))[1], %s) AS s",
                            (ts_config, term, term)).fetchone()["s"]
        if _close_to_vocabulary(stem, words):
            typos.append(term)
    return typos


def word_boundary_pattern(term: str) -> str:
    """POSIX regex matching ``term`` literally, bounded by non-word characters.
    Every non-alphanumeric character is backslash-escaped (literal in ARE)."""
    escaped = "".join(ch if ch.isalnum() else "\\" + ch for ch in term.strip())
    escaped = re.sub(r"(\\\s)+", r"\\s+", escaped)  # tolerate whitespace variations
    return f"(^|[^[:alnum:]_]){escaped}($|[^[:alnum:]_])"


def exact_search(
    conn: psycopg.Connection,
    terms: Sequence[str],
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    """Case-insensitive whole-term matches of identifiers / quoted phrases in the
    chunk text, document title or heading path."""
    cleaned = [t.strip() for t in terms if t and 2 <= len(t.strip()) <= 120][:10]
    if not cleaned or top_k <= 0:
        return []
    patterns = [word_boundary_pattern(t) for t in cleaned]
    rows = conn.execute(
        f"""
        SELECT {CHUNK_COLUMNS},
               (SELECT count(*) FROM unnest(%(p)s::text[]) pat
                WHERE c.text ~* pat OR d.title ~* pat OR array_to_string(c.heading_path, ' ') ~* pat) AS hits
        FROM chunks c JOIN documents d ON d.document_id = c.document_id
        WHERE d.status <> 'deleted'
          AND (c.text ~* ANY(%(p)s) OR d.title ~* ANY(%(p)s) OR array_to_string(c.heading_path, ' ') ~* ANY(%(p)s))
        ORDER BY hits DESC, d.relative_path, c.chunk_index
        LIMIT %(k)s
        """,
        {"p": patterns, "k": top_k},
    ).fetchall()
    results: list[RetrievedChunk] = []
    for rank, row in enumerate(rows, start=1):
        chunk = RetrievedChunk.from_row(row)
        chunk.exact_rank = rank
        chunk.exact_hits = int(row["hits"])
        results.append(chunk)
    return results
