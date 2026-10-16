"""Keyword retrieval with PostgreSQL full-text search, plus an exact
identifier/phrase channel for terms that stemming would break apart
(e.g. ``HR-POL-012``, ``401(k)``, ``ISO 27001``)."""

from __future__ import annotations

import re
from collections.abc import Sequence

import psycopg

from app.retrieval.types import CHUNK_COLUMNS, RetrievedChunk

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
MAX_TERMS = 32

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
               + COALESCE((SELECT count(*) FROM unnest(%(phrases)s::text[]) p
                           WHERE c.search_tsv @@ phraseto_tsquery(%(cfg)s::regconfig, p)), 0) AS score
        FROM chunks c JOIN documents d ON d.document_id = c.document_id, q
        WHERE d.status <> 'deleted' AND numnode(q.query) > 0 AND c.search_tsv @@ q.query
        ORDER BY score DESC, c.chunk_id
        LIMIT %(k)s
        """,
        {"cfg": ts_config, "tsq": tsquery, "k": top_k, "phrases": phrase_list},
    ).fetchall()
    results: list[RetrievedChunk] = []
    for rank, row in enumerate(rows, start=1):
        chunk = RetrievedChunk.from_row(row)
        chunk.keyword_rank = rank
        chunk.keyword_score = float(row["score"])
        results.append(chunk)
    return results


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
