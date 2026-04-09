"""
Memory System Interface — Abstract base + two built-in implementations.

Architecture:
  BaseMemorySystem          — abstract interface (ingest / retrieve / clear / stats)
  ├── SimpleMemorySystem    — keyword + metadata match, no external deps
  └── InvertedIndexSystem  — character/location/keyword inverted index

Key design:
  retrieve() returns List[RetrievalResult] (not strings).
  Each result carries event_id so retrieval recall scoring is exact.
  Scoring no longer needs an LLM — compare event_ids directly.

Usage:
  from engine.memory_system import SimpleMemorySystem, InvertedIndexSystem

  mem = InvertedIndexSystem()
  mem.ingest(events)
  results = mem.retrieve("Alice meeting", time_point=None, top_k=10)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict
import re

from engine.time_machine import TimePoint, MemoryEvent


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    A single retrieval result.

    Every memory system implementation must return this type.
    The event_id field is the key for retrieval recall scoring.
    """
    event_id: str
    title: str
    time: str                 # ISO date string
    content: str              # full event content (for LLM consumption)
    characters: List[str]     # structured character list
    locations: List[str]       # structured location list
    relevance_score: float     # 0-1, system-internal
    retrieval_method: str      # how this result was found

    def to_context_string(self) -> str:
        """Format for LLM consumption"""
        chars = ", ".join(sorted(self.characters)) if self.characters else "None"
        locs = ", ".join(sorted(self.locations)) if self.locations else "None"
        return (
            f"[{self.time}] {self.title}\n"
            f"{self.content}\n"
            f"Characters: {chars}\n"
            f"Locations: {locs}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Base
# ─────────────────────────────────────────────────────────────────────────────

class BaseMemorySystem(ABC):
    """
    Abstract interface for memory systems under benchmark.

    Contract:
      ingest(events)      — store events; called once before any retrieve
      retrieve(query, time_point, top_k) → List[RetrievalResult]
      clear()            — reset state (optional)
      get_stats()        — return metadata dict (optional)

    Scoring: The benchmark checks whether evidence_event_ids appear in
    the retrieved list. This is pure retrieval recall — no LLM needed.
    """

    @abstractmethod
    def ingest(self, events: List[MemoryEvent]) -> None:
        """Store events. Called once; subsequent calls replace state."""
        pass

    @abstractmethod
    def retrieve(self, query: str,
                 time_point: Optional[TimePoint] = None,
                 top_k: int = 10) -> List[RetrievalResult]:
        """
        Search for events relevant to the query.

        Args:
            query: natural language query string
            time_point: if set, only return events at or before this time
            top_k: maximum number of results

        Returns:
            List[RetrievalResult], sorted by relevance_score descending.
            Empty list if nothing matches.
        """
        pass

    def clear(self) -> None:
        """Reset state (optional override)"""
        pass

    def get_stats(self) -> dict:
        """Return system statistics (optional override)"""
        return {"type": self.__class__.__name__}


# ─────────────────────────────────────────────────────────────────────────────
# Implementation 1: Simple Keyword Memory
# ─────────────────────────────────────────────────────────────────────────────

class SimpleMemorySystem(BaseMemorySystem):
    """
    Simple keyword + metadata match. No external dependencies.

    Retrieval strategy:
      1. Exact keyword match in title (weight 0.5)
      2. Keyword match in content (weight 0.3)
      3. Character name in query matches event characters (weight 0.4)
      4. Location name in query matches event locations (weight 0.3)

    All weights are summed; results sorted by total score.
    """

    def __init__(self, name: str = "SimpleKeyword"):
        self.name = name
        self._events: List[MemoryEvent] = []

    def ingest(self, events: List[MemoryEvent]) -> None:
        self._events = list(events)

    def retrieve(self, query: str,
                 time_point: Optional[TimePoint] = None,
                 top_k: int = 10) -> List[RetrievalResult]:
        if not self._events:
            return []

        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        scored: List[tuple] = []  # (event, score, method)

        for ev in self._events:
            # Time filter
            if time_point and ev.time > time_point:
                continue

            score = 0.0
            methods = []

            # Title match
            title_words = set(ev.title.lower().split())
            title_overlap = query_words & title_words
            if title_overlap:
                score += 0.5 * len(title_overlap) / max(1, len(query_words))
                methods.append(f"title:{','.join(title_overlap)}")

            # Content match
            content_words = set(re.findall(r'\w+', ev.content.lower()))
            content_overlap = query_words & content_words
            if content_overlap:
                score += 0.3 * len(content_overlap) / max(1, len(query_words))
                if not title_overlap:
                    methods.append(f"content:{','.join(list(content_overlap)[:3])}")

            # Character match
            ev_chars_lower = {c.lower() for c in ev.characters}
            query_chars = {w for w in query_words if len(w) > 2}
            char_overlap = ev_chars_lower & query_chars
            if char_overlap:
                score += 0.4
                methods.append(f"char:{','.join(char_overlap)}")

            # Location match — use the full query token set (not query_chars which
            # is shared with character matching and causes location names to be missed)
            ev_locs_lower = {l.lower() for l in ev.locations}
            loc_overlap = ev_locs_lower & query_words
            if loc_overlap:
                score += 0.3
                methods.append(f"loc:{','.join(loc_overlap)}")

            if score > 0:
                scored.append((ev, min(score, 1.0), "; ".join(methods)))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                event_id=ev.id,
                title=ev.title,
                time=str(ev.time),
                content=ev.content,
                characters=list(ev.characters),
                locations=list(ev.locations),
                relevance_score=score,
                retrieval_method=method,
            )
            for ev, score, method in scored[:top_k]
        ]

    def clear(self) -> None:
        self._events.clear()

    def get_stats(self) -> dict:
        return {"type": "simple_keyword", "num_events": len(self._events)}


# ─────────────────────────────────────────────────────────────────────────────
# Implementation 2: Inverted Index System
# ─────────────────────────────────────────────────────────────────────────────

class InvertedIndexSystem(BaseMemorySystem):
    """
    Inverted index on characters, locations, and keyword tokens.

    Retrieval strategy:
      - Query is tokenized into chars/locs/tokens
      - Each posting list is fetched; results merged by event_id
      - Score = sum of posting weights (char=0.4, loc=0.3, token=0.1 per occurrence)

    No external dependencies. Fully local.
    """

    def __init__(self, name: str = "InvertedIndex"):
        self.name = name
        self._events: Dict[str, MemoryEvent] = {}   # event_id → event
        self._sorted_events: List[MemoryEvent] = []

        # Inverted indexes: token → {event_id}
        self._char_index: Dict[str, Set[str]] = {}      # char_lower → {event_ids}
        self._loc_index: Dict[str, Set[str]] = {}        # loc_lower → {event_ids}
        self._token_index: Dict[str, Set[str]] = {}      # word → {event_ids}

        # Posting weights
        self._char_weight = 0.4
        self._loc_weight = 0.3
        self._token_weight = 0.1

    def ingest(self, events: List[MemoryEvent]) -> None:
        # Reset
        self._events.clear()
        self._sorted_events.clear()
        self._char_index.clear()
        self._loc_index.clear()
        self._token_index.clear()

        for ev in events:
            self._events[ev.id] = ev

        self._sorted_events = sorted(
            events,
            key=lambda e: (e.time.year, e.time.month, e.time.day)
        )

        # Build indexes
        for ev in events:
            # Character index
            for char in ev.characters:
                cl = char.lower()
                if cl not in self._char_index:
                    self._char_index[cl] = set()
                self._char_index[cl].add(ev.id)

            # Location index
            for loc in ev.locations:
                ll = loc.lower()
                if ll not in self._loc_index:
                    self._loc_index[ll] = set()
                self._loc_index[ll].add(ev.id)

            # Token index (from title + content)
            tokens = set(re.findall(r'\w{3,}', (ev.title + " " + ev.content).lower()))
            for tok in tokens:
                if tok not in self._token_index:
                    self._token_index[tok] = set()
                self._token_index[tok].add(ev.id)

    def retrieve(self, query: str,
                 time_point: Optional[TimePoint] = None,
                 top_k: int = 10) -> List[RetrievalResult]:
        if not self._events:
            return []

        # Parse query tokens
        query_lower = query.lower()
        query_tokens = set(re.findall(r'\w{3,}', query_lower))

        # Determine which chars/locs in query by checking against indexes
        query_chars = {w for w in query_tokens if w in self._char_index}
        query_locs = {w for w in query_tokens if w in self._loc_index}
        query_words = query_tokens - query_chars - query_locs

        # Aggregate scores per event
        event_scores: Dict[str, tuple] = {}  # event_id → (score, method_str)

        # Character matches
        for char in query_chars:
            for eid in self._char_index[char]:
                ev = self._events[eid]
                if time_point and ev.time > time_point:
                    continue
                score, method = event_scores.get(eid, (0.0, ""))
                new_score = score + self._char_weight
                new_method = method + f"char:{char};"
                event_scores[eid] = (new_score, new_method)

        # Location matches
        for loc in query_locs:
            for eid in self._loc_index[loc]:
                ev = self._events[eid]
                if time_point and ev.time > time_point:
                    continue
                score, method = event_scores.get(eid, (0.0, ""))
                new_score = score + self._loc_weight
                new_method = method + f"loc:{loc};"
                event_scores[eid] = (new_score, new_method)

        # Token matches
        for tok in query_words:
            if tok not in self._token_index:
                continue
            for eid in self._token_index[tok]:
                ev = self._events[eid]
                if time_point and ev.time > time_point:
                    continue
                score, method = event_scores.get(eid, (0.0, ""))
                new_score = score + self._token_weight
                new_method = method + f"tok:{tok};"
                event_scores[eid] = (new_score, new_method)

        # Sort by score
        sorted_ids = sorted(event_scores.keys(), key=lambda eid: event_scores[eid][0], reverse=True)

        results = []
        for eid in sorted_ids[:top_k]:
            ev = self._events[eid]
            score, method = event_scores[eid]
            results.append(RetrievalResult(
                event_id=ev.id,
                title=ev.title,
                time=str(ev.time),
                content=ev.content,
                characters=list(ev.characters),
                locations=list(ev.locations),
                relevance_score=min(score, 1.0),
                retrieval_method=method.strip(";"),
            ))

        # Fallback: if no index matches at all (all query tokens OOV), return
        # the most recent events so the caller always gets something back.
        if not results:
            fallback = self._sorted_events
            if time_point:
                fallback = [e for e in fallback if e.time <= time_point]
            for ev in reversed(fallback[-top_k:]):
                results.append(RetrievalResult(
                    event_id=ev.id,
                    title=ev.title,
                    time=str(ev.time),
                    content=ev.content,
                    characters=list(ev.characters),
                    locations=list(ev.locations),
                    relevance_score=0.0,
                    retrieval_method="fallback:recency",
                ))

        return results

    def clear(self) -> None:
        self._events.clear()
        self._sorted_events.clear()
        self._char_index.clear()
        self._loc_index.clear()
        self._token_index.clear()

    def get_stats(self) -> dict:
        return {
            "type": "inverted_index",
            "num_events": len(self._events),
            "num_char_index_entries": len(self._char_index),
            "num_loc_index_entries": len(self._loc_index),
            "num_token_index_entries": len(self._token_index),
        }


# ─── Backward compatibility alias ─────────────────────────────────────────────
# Existing code references 'MemorySystem' as the base class.
# Now BaseMemorySystem is the abstract class; MemorySystem is an alias.
MemorySystem = BaseMemorySystem
