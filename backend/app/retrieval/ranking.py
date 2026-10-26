"""Reciprocal Rank Fusion and result de-duplication."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class RankedList:
    """One retrieval result list: chunk IDs in rank order (best first)."""

    name: str
    chunk_ids: list[str]
    weight: float = 1.0


@dataclass
class FusedItem:
    chunk_id: str
    score: float
    contributions: dict[str, int] = field(default_factory=dict)  # list name -> rank (1-based)

    @property
    def best_rank(self) -> int:
        return min(self.contributions.values()) if self.contributions else 10**9


def reciprocal_rank_fusion(lists: Sequence[RankedList], k: int = 60, max_rank: int | None = None) -> list[FusedItem]:
    """RRF_score(d) = Σ_i w_i / (k + rank_i(d)), ranks starting at 1.

    A chunk that appears more than once within the same list only counts at its
    best rank. With ``max_rank``, ranks below that cut-off add nothing: otherwise a
    chunk ranked e.g. 16th and 17th in two weak lists outscores the 1st result of a
    single strong list. Such chunks stay in the result (score 0) after all scored ones.
    Ties are broken by the best individual rank, then chunk ID, so the ordering is
    deterministic.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    fused: dict[str, FusedItem] = {}
    for ranked in lists:
        if ranked.weight <= 0:
            continue
        seen: set[str] = set()
        rank = 0
        for chunk_id in ranked.chunk_ids:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank += 1
            item = fused.setdefault(chunk_id, FusedItem(chunk_id, 0.0))
            if max_rank is None or rank <= max_rank:
                item.score += ranked.weight / (k + rank)
            prior = item.contributions.get(ranked.name)
            item.contributions[ranked.name] = rank if prior is None else min(prior, rank)
    return sorted(fused.values(), key=lambda it: (-it.score, it.best_rank, it.chunk_id))


def dedupe_by_content(items: Sequence[str], content_hash_of: dict[str, str]) -> tuple[list[str], dict[str, list[str]]]:
    """Collapse chunks with identical content (e.g. duplicate files), keeping the
    best-ranked one. Returns (kept_ids, {kept_id: [dropped_ids]})."""
    kept: list[str] = []
    first_by_hash: dict[str, str] = {}
    dropped: dict[str, list[str]] = {}
    for chunk_id in items:
        content = content_hash_of.get(chunk_id)
        if content is not None and content in first_by_hash:
            dropped.setdefault(first_by_hash[content], []).append(chunk_id)
            continue
        if content is not None:
            first_by_hash[content] = chunk_id
        kept.append(chunk_id)
    return kept, dropped
