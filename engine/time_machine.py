"""
Virtual Time Machine — Backed by the event graph engine.

Core capability: Jump to any point in the timeline and query what the
memory system knows at that moment.

Internal architecture:
  - CharacterGraph  → character tracking, relationship strength
  - TemporalGraph   → event ordering, causal links, temporal distance
  - SpatialContext  → location-based organisation

Public API
──────────
Ingestion:
  add_event(event)
  ingest_events(events)

Point-in-time queries:
  get_snapshot(at_time)                     → MemorySnapshot
  get_character_state_at(char, time)        → dict

Range / filter queries:
  get_events_before(time, n)                → List[MemoryEvent]
  get_events_after(time, n)                 → List[MemoryEvent]
  get_events_in_range(t1, t2)              → List[MemoryEvent]
  get_events_by_character(char, time_point) → List[MemoryEvent]
  get_events_by_location(loc, time_point)   → List[MemoryEvent]
  get_events_by_tag(tag, time_point)        → List[MemoryEvent]
  search_events(keyword, time_point, top_k) → List[MemoryEvent]
  get_coappearances(char1, char2)           → List[MemoryEvent]

Temporal reasoning:
  get_temporal_distance(t1, t2)             → Dict[str, float]
  get_most_active_period(window_days)       → Tuple[TimePoint, TimePoint, int]
  replay(start, end, step_days)             → Iterator[MemorySnapshot]

Character / location analysis:
  get_character_appearances(char)           → List[Tuple[TimePoint, str]]
  get_character_trajectory(char)            → List[Tuple[TimePoint, str, str]]
  get_timeline_summary()                    → dict

Accessors:
  get_event_by_id(id)                       → Optional[MemoryEvent]
  get_timeline_span()                       → Tuple[TimePoint, TimePoint]
  get_random_time_point(seed)               → TimePoint
  get_character_graph()                     → CharacterGraph
  get_temporal_graph()                      → TemporalGraph
  get_spatial_context()                     → SpatialContext
  build_temporal_graph()                    → Dict[str, List[str]]
"""

from dataclasses import dataclass, field
from typing import (
    Dict, Iterator, List, Optional, Set, Tuple
)
from enum import Enum
import re

from engine.event_graph import (
    CharacterGraph, TemporalGraph, SpatialContext,
    RelationType, EventRelation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Time Granularity
# ─────────────────────────────────────────────────────────────────────────────

class TimeGranularity(Enum):
    YEAR  = "year"
    MONTH = "month"
    DAY   = "day"
    HOUR  = "hour"


# ─────────────────────────────────────────────────────────────────────────────
# TimePoint — immutable temporal coordinate
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TimePoint:
    """A specific moment in the timeline (year + optional month/day/hour)."""
    year:  int
    month: int = 1
    day:   int = 1
    hour:  int = 0

    # ── Comparison operators (tuple-based, no float arithmetic) ──────────────
    def _tuple(self):
        return (self.year, self.month, self.day, self.hour)

    def __lt__(self, other: 'TimePoint') -> bool:  return self._tuple() <  other._tuple()
    def __le__(self, other: 'TimePoint') -> bool:  return self._tuple() <= other._tuple()
    def __gt__(self, other: 'TimePoint') -> bool:  return self._tuple() >  other._tuple()
    def __ge__(self, other: 'TimePoint') -> bool:  return self._tuple() >= other._tuple()
    def __eq__(self, other) -> bool:
        return isinstance(other, TimePoint) and self._tuple() == other._tuple()
    def __hash__(self):
        return hash(self._tuple())

    def __repr__(self):
        if self.hour > 0:
            return f"{self.year}-{self.month:02d}-{self.day:02d} {self.hour:02d}:00"
        elif self.day > 1 or self.month > 1:
            return f"{self.year}-{self.month:02d}-{self.day:02d}"
        return f"{self.year}"

    @classmethod
    def from_string(cls, s: str) -> 'TimePoint':
        """Parse 'YYYY', 'YYYY-MM', 'YYYY-MM-DD', or 'YYYY-MM-DD-HH'."""
        parts = s.strip().split('-')
        return cls(
            year  = int(parts[0]),
            month = int(parts[1]) if len(parts) > 1 else 1,
            day   = int(parts[2]) if len(parts) > 2 else 1,
            hour  = int(parts[3]) if len(parts) > 3 else 0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MemoryEvent — core event record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryEvent:
    """A single event in the memoir."""
    id:               str
    time:             TimePoint
    title:            str
    content:          str
    characters:       Set[str]  = field(default_factory=set)
    locations:        Set[str]  = field(default_factory=set)
    tags:             Set[str]  = field(default_factory=set)
    related_events:   List[str] = field(default_factory=list)
    emotional_valence: float    = 0.0   # −1 (negative) … +1 (positive)

    def __repr__(self):
        return f"Event[{self.time}]: {self.title[:50]}"


# ─────────────────────────────────────────────────────────────────────────────
# MemorySnapshot — state of knowledge at time T
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MemorySnapshot:
    """Everything the system knows at or before at_time."""
    at_time:           TimePoint
    events:            List[MemoryEvent]
    # character → title of their most recent event
    character_states:  Dict[str, str]
    # timeline_position: 0.0 = start, 1.0 = end
    timeline_position: float
    # convenience extras
    active_characters: Set[str]           = field(default_factory=set)
    active_locations:  Set[str]           = field(default_factory=set)
    recent_events:     List[MemoryEvent]  = field(default_factory=list)

    @property
    def event_count(self) -> int:
        return len(self.events)


# ─────────────────────────────────────────────────────────────────────────────
# TimeMachine
# ─────────────────────────────────────────────────────────────────────────────

class TimeMachine:
    """
    Core time-navigation engine backed by three graph indexes.

    Invariant: self.events is the single source of truth.
    All graphs and the sorted-events cache are derived indexes that stay
    in sync whenever events are added.
    """

    # Number of "recent" events to include in a snapshot
    SNAPSHOT_RECENT_N = 5

    def __init__(self):
        self.events: Dict[str, MemoryEvent] = {}

        self.char_graph = CharacterGraph()
        self.temp_graph = TemporalGraph()
        self.spatial    = SpatialContext()

        self.book_title: str             = ""
        self.book_start: Optional[TimePoint] = None
        self.book_end:   Optional[TimePoint] = None

        # Lazy-sorted cache — invalidated on every add_event
        self._sorted_events: Optional[List[MemoryEvent]] = None
        # Flag: PRECEDES links need rebuilding after bulk ingest
        self._precedes_dirty: bool = False

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_events(self, events: List[MemoryEvent]) -> None:
        """Load a list of events.  Rebuilds PRECEDES links once at the end."""
        for event in events:
            self._index_event(event)
        self._rebuild_precedes_links()

    def add_event(self, event: MemoryEvent) -> None:
        """Add a single event and keep all indexes consistent."""
        self._index_event(event)
        self._rebuild_precedes_links()

    def _index_event(self, event: MemoryEvent) -> None:
        """Index one event into all three graphs (no PRECEDES links yet)."""
        self.events[event.id] = event
        self._sorted_events = None   # invalidate cache

        # ── Book bounds ───────────────────────────────────────────────────────
        if self.book_start is None or event.time < self.book_start:
            self.book_start = event.time
        if self.book_end is None or event.time > self.book_end:
            self.book_end = event.time

        # ── CharacterGraph ────────────────────────────────────────────────────
        for char_name in event.characters:
            char_id = f"C_{char_name}"
            if char_id not in self.char_graph.characters:
                self.char_graph.add_character(
                    char_id=char_id, name=char_name,
                    role="unknown", traits=[], goals=[]
                )
            self.char_graph.link_event_to_character(char_id, event.id)

        char_list = sorted(event.characters)
        for i, c1 in enumerate(char_list):
            for c2 in char_list[i + 1:]:
                try:
                    self.char_graph.add_relation(
                        f"C_{c1}", f"C_{c2}", RelationType.COLLEAGUE, strength=0.3
                    )
                except ValueError:
                    pass

        # ── TemporalGraph (node only — links added later) ─────────────────────
        self.temp_graph.add_event(
            event_id  = event.id,
            time_str  = str(event.time),
            title     = event.title,
            summary   = event.content[:200],
            importance= 0.5,
            mood      = "neutral",
            characters= list(event.characters),
            locations = list(event.locations),
        )

        # ── SpatialContext ────────────────────────────────────────────────────
        for loc_name in event.locations:
            loc_id = f"L_{loc_name}"
            if loc_id not in self.spatial.locations:
                self.spatial.add_location(
                    loc_id=loc_id, name=loc_name,
                    loc_type="unknown", significance=0.5
                )
            self.spatial.record_event_at(event.id, [loc_id])

    def _rebuild_precedes_links(self) -> None:
        """
        Rebuild all PRECEDES links from scratch based on current sort order.

        Called after every add_event / ingest_events so that inserting events
        out of chronological order never leaves stale links.
        """
        sorted_evs = self._get_sorted_events()
        # Clear existing PRECEDES links
        for eid in self.temp_graph.outgoing:
            self.temp_graph.outgoing[eid] = [
                lnk for lnk in self.temp_graph.outgoing[eid]
                if lnk.relation != EventRelation.PRECEDES
            ]
        for eid in self.temp_graph.incoming:
            self.temp_graph.incoming[eid] = [
                lnk for lnk in self.temp_graph.incoming[eid]
                if lnk.relation != EventRelation.PRECEDES
            ]
        # Add fresh consecutive PRECEDES links
        for i in range(len(sorted_evs) - 1):
            self.temp_graph.link(
                sorted_evs[i].id, sorted_evs[i + 1].id,
                EventRelation.PRECEDES, weight=1.0
            )

    # ── Sorted events (lazy cache) ────────────────────────────────────────────

    def _get_sorted_events(self) -> List[MemoryEvent]:
        if self._sorted_events is None:
            self._sorted_events = sorted(
                self.events.values(),
                key=lambda e: (e.time.year, e.time.month, e.time.day, e.time.hour)
            )
        return self._sorted_events

    # ── Point-in-time queries ─────────────────────────────────────────────────

    def get_snapshot(self, at_time: TimePoint) -> MemorySnapshot:
        """Return all events at or before at_time, plus contextual metadata."""
        events_up_to_T = [
            e for e in self._get_sorted_events() if e.time <= at_time
        ]

        # Most recent event per character (state at time T)
        character_states: Dict[str, str] = {}
        for event in reversed(events_up_to_T):
            for char in event.characters:
                if char not in character_states:
                    character_states[char] = event.title

        # Timeline position 0.0 – 1.0
        if self.book_start and self.book_end:
            total = self._time_to_float(self.book_end) - self._time_to_float(self.book_start)
            pos   = (
                (self._time_to_float(at_time) - self._time_to_float(self.book_start)) / total
                if total > 0 else 1.0
            )
            pos = max(0.0, min(1.0, pos))   # clamp to [0, 1]
        else:
            pos = 0.0

        active_characters: Set[str] = set()
        active_locations:  Set[str] = set()
        for e in events_up_to_T:
            active_characters |= e.characters
            active_locations  |= e.locations

        recent = events_up_to_T[-self.SNAPSHOT_RECENT_N:]

        return MemorySnapshot(
            at_time           = at_time,
            events            = events_up_to_T,
            character_states  = character_states,
            timeline_position = pos,
            active_characters = active_characters,
            active_locations  = active_locations,
            recent_events     = recent,
        )

    def get_character_state_at(self, character: str, time: TimePoint) -> dict:
        """
        Summarise everything known about a character at or before `time`.

        Returns a dict with:
          - appearances: list of (TimePoint, event_title) up to time
          - last_seen_at: most recent location
          - event_count: total number of events
          - coappearances: {other_char: count} of co-event partners
          - emotional_arc: avg emotional_valence across their events
        """
        char_events = [
            e for e in self._get_sorted_events()
            if character in e.characters and e.time <= time
        ]
        if not char_events:
            return {"character": character, "found": False}

        # Co-appearances
        coapp: Dict[str, int] = {}
        for e in char_events:
            for other in e.characters:
                if other != character:
                    coapp[other] = coapp.get(other, 0) + 1
        coapp_sorted = sorted(coapp.items(), key=lambda x: x[1], reverse=True)

        # Emotional arc
        avg_valence = sum(e.emotional_valence for e in char_events) / len(char_events)

        # Last known location
        last_locs: List[str] = []
        for e in reversed(char_events):
            if e.locations:
                last_locs = sorted(e.locations)
                break

        return {
            "character":     character,
            "found":         True,
            "event_count":   len(char_events),
            "appearances":   [(e.time, e.title) for e in char_events],
            "last_seen_at":  last_locs,
            "coappearances": coapp_sorted[:10],
            "emotional_arc": avg_valence,
        }

    # ── Range / filter queries ────────────────────────────────────────────────

    def get_events_before(self, time: TimePoint, n: int = 5) -> List[MemoryEvent]:
        """Up to n events strictly before `time`, most-recent first."""
        return [e for e in reversed(self._get_sorted_events()) if e.time < time][:n]

    def get_events_after(self, time: TimePoint, n: int = 5) -> List[MemoryEvent]:
        """Up to n events strictly after `time`, earliest first."""
        return [e for e in self._get_sorted_events() if e.time > time][:n]

    def get_events_in_range(self, t1: TimePoint, t2: TimePoint) -> List[MemoryEvent]:
        """All events with t1 <= event.time <= t2, in chronological order."""
        lo, hi = (t1, t2) if t1 <= t2 else (t2, t1)
        return [e for e in self._get_sorted_events() if lo <= e.time <= hi]

    def get_events_by_character(
        self,
        character: str,
        time_point: Optional[TimePoint] = None,
    ) -> List[MemoryEvent]:
        """All events featuring `character`, optionally filtered by time_point."""
        return [
            e for e in self._get_sorted_events()
            if character in e.characters
            and (time_point is None or e.time <= time_point)
        ]

    def get_events_by_location(
        self,
        location: str,
        time_point: Optional[TimePoint] = None,
    ) -> List[MemoryEvent]:
        """All events at `location` (case-insensitive), optionally up to time_point."""
        loc_lower = location.lower()
        return [
            e for e in self._get_sorted_events()
            if any(l.lower() == loc_lower for l in e.locations)
            and (time_point is None or e.time <= time_point)
        ]

    def get_events_by_tag(
        self,
        tag: str,
        time_point: Optional[TimePoint] = None,
    ) -> List[MemoryEvent]:
        """All events carrying `tag` (case-insensitive)."""
        tag_lower = tag.lower()
        return [
            e for e in self._get_sorted_events()
            if any(t.lower() == tag_lower for t in e.tags)
            and (time_point is None or e.time <= time_point)
        ]

    def search_events(
        self,
        keyword: str,
        time_point: Optional[TimePoint] = None,
        top_k: int = 10,
    ) -> List[MemoryEvent]:
        """
        Keyword search over title + content + characters + locations + tags.

        Scoring (additive):
          title match      → 3 pts per matched word
          character match  → 2 pts per matched name
          location match   → 2 pts per matched name
          tag match        → 2 pts per matched tag
          content match    → 1 pt per matched word

        Returns up to top_k events sorted by score descending.
        """
        tokens = set(re.findall(r'\w{2,}', keyword.lower()))
        if not tokens:
            return []

        scored: List[Tuple[MemoryEvent, float]] = []
        for e in self._get_sorted_events():
            if time_point and e.time > time_point:
                continue

            score = 0.0
            title_words   = set(re.findall(r'\w+', e.title.lower()))
            content_words = set(re.findall(r'\w+', e.content.lower()))
            chars_lower   = {c.lower() for c in e.characters}
            locs_lower    = {l.lower() for l in e.locations}
            tags_lower    = {t.lower() for t in e.tags}

            for tok in tokens:
                if tok in title_words:   score += 3
                if tok in chars_lower:   score += 2
                if tok in locs_lower:    score += 2
                if tok in tags_lower:    score += 2
                if tok in content_words: score += 1

            if score > 0:
                scored.append((e, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]

    def get_coappearances(
        self,
        char1: str,
        char2: str,
        time_point: Optional[TimePoint] = None,
    ) -> List[MemoryEvent]:
        """Events where both char1 and char2 appear together."""
        return [
            e for e in self._get_sorted_events()
            if char1 in e.characters and char2 in e.characters
            and (time_point is None or e.time <= time_point)
        ]

    # ── Temporal reasoning ────────────────────────────────────────────────────

    def get_temporal_distance(self, time1: TimePoint, time2: TimePoint) -> Dict[str, float]:
        """
        Approximate calendar distance between two TimePoints.

        Returns hours / days / months / years (all floats, approximate).
        Uses 30-day months, 365-day years for simplicity.
        """
        diff_hours = abs(self._time_to_float(time1) - self._time_to_float(time2))
        days = diff_hours / 24.0
        return {
            "hours":  diff_hours,
            "days":   days,
            "months": days / 30.0,
            "years":  days / 365.0,
        }

    def get_most_active_period(
        self,
        window_days: int = 30,
    ) -> Tuple[Optional[TimePoint], Optional[TimePoint], int]:
        """
        Find the sliding window of `window_days` days that contains the most events.

        Returns (window_start, window_end, event_count).
        Returns (None, None, 0) if there are no events.
        """
        sorted_evs = self._get_sorted_events()
        if not sorted_evs:
            return None, None, 0

        best_count  = 0
        best_start  = sorted_evs[0].time
        best_end    = sorted_evs[0].time
        window_h    = window_days * 24.0

        # Sliding-window: two-pointer over sorted events
        left = 0
        for right, ev in enumerate(sorted_evs):
            # Advance left pointer until window fits
            while (
                self._time_to_float(ev.time) - self._time_to_float(sorted_evs[left].time)
                > window_h
            ):
                left += 1
            count = right - left + 1
            if count > best_count:
                best_count = count
                best_start = sorted_evs[left].time
                best_end   = ev.time

        return best_start, best_end, best_count

    def replay(
        self,
        start_time: Optional[TimePoint] = None,
        end_time:   Optional[TimePoint] = None,
        step_days:  int = 30,
    ) -> Iterator[MemorySnapshot]:
        """
        Step through the timeline in increments of `step_days`, yielding
        a MemorySnapshot at each step.

        Useful for visualising how knowledge accumulates over time.

        Example:
            for snap in tm.replay(step_days=7):
                print(snap.at_time, snap.event_count)
        """
        if not self.events:
            return

        t_start = start_time or self.book_start
        t_end   = end_time   or self.book_end

        if t_start is None or t_end is None:
            return

        step_hours = step_days * 24.0
        current_h  = self._time_to_float(t_start)
        end_h      = self._time_to_float(t_end)

        while current_h <= end_h:
            tp = self._float_to_timepoint(current_h)
            yield self.get_snapshot(tp)
            current_h += step_hours

        # Always yield final snapshot at exact end time
        if self._float_to_timepoint(current_h - step_hours) < t_end:
            yield self.get_snapshot(t_end)

    # ── Character / location analysis ─────────────────────────────────────────

    def get_character_appearances(self, character: str) -> List[Tuple[TimePoint, str]]:
        """Chronological list of (time, event_title) for every event featuring `character`."""
        char_id    = f"C_{character}"
        event_ids  = self.char_graph.character_events.get(char_id, [])
        result     = []
        for eid in event_ids:
            ev = self.events.get(eid)
            if ev:
                result.append((ev.time, ev.title))
        result.sort(key=lambda x: x[0])
        return result

    def get_character_trajectory(self, character: str) -> List[Tuple[TimePoint, str, str]]:
        """
        Return (time, event_title, location) for every event featuring `character`
        that has at least one location attached.  Chronological order.

        Useful for answering "where was X at time T?".
        """
        trajectory = []
        for e in self._get_sorted_events():
            if character not in e.characters:
                continue
            primary_loc = sorted(e.locations)[0] if e.locations else "(unknown)"
            trajectory.append((e.time, e.title, primary_loc))
        return trajectory

    def get_timeline_summary(self) -> dict:
        """
        High-level statistics about the loaded events.

        Includes event count, time span, most/least active periods,
        character and location counts.
        """
        sorted_evs = self._get_sorted_events()
        if not sorted_evs:
            return {"total_events": 0}

        # Character counts
        char_counts: Dict[str, int] = {}
        loc_counts:  Dict[str, int] = {}
        tag_counts:  Dict[str, int] = {}
        for e in sorted_evs:
            for c in e.characters:
                char_counts[c] = char_counts.get(c, 0) + 1
            for l in e.locations:
                loc_counts[l] = loc_counts.get(l, 0) + 1
            for t in e.tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1

        top_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_locs  = sorted(loc_counts.items(),  key=lambda x: x[1], reverse=True)[:5]
        top_tags  = sorted(tag_counts.items(),  key=lambda x: x[1], reverse=True)[:5]

        span = self.get_temporal_distance(self.book_start, self.book_end)

        win_start, win_end, win_count = self.get_most_active_period(30)

        avg_valence = (
            sum(e.emotional_valence for e in sorted_evs) / len(sorted_evs)
        )

        return {
            "total_events":      len(sorted_evs),
            "span_days":         round(span["days"], 1),
            "span_years":        round(span["years"], 2),
            "start":             str(self.book_start),
            "end":               str(self.book_end),
            "unique_characters": len(char_counts),
            "unique_locations":  len(loc_counts),
            "unique_tags":       len(tag_counts),
            "top_characters":    top_chars,
            "top_locations":     top_locs,
            "top_tags":          top_tags,
            "avg_emotional_valence": round(avg_valence, 3),
            "most_active_window": {
                "start":       str(win_start),
                "end":         str(win_end),
                "event_count": win_count,
            },
        }

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_event_by_id(self, event_id: str) -> Optional[MemoryEvent]:
        return self.events.get(event_id)

    def get_timeline_span(self) -> Tuple[Optional[TimePoint], Optional[TimePoint]]:
        return (self.book_start, self.book_end)

    def get_character_graph(self) -> CharacterGraph:
        return self.char_graph

    def get_temporal_graph(self) -> TemporalGraph:
        return self.temp_graph

    def get_spatial_context(self) -> SpatialContext:
        return self.spatial

    def build_temporal_graph(self) -> Dict[str, List[str]]:
        """
        Flat adjacency dict: events within 7 days or sharing characters/locations
        are considered "related".
        """
        sorted_evs = self._get_sorted_events()
        graph: Dict[str, List[str]] = {e.id: [] for e in sorted_evs}

        for i, event in enumerate(sorted_evs):
            for j in range(max(0, i - 5), min(len(sorted_evs), i + 6)):
                if i == j:
                    continue
                other = sorted_evs[j]
                if self.get_temporal_distance(event.time, other.time)["days"] <= 7:
                    if other.id not in graph[event.id]:
                        graph[event.id].append(other.id)

            for j in range(len(sorted_evs)):
                if i == j:
                    continue
                other = sorted_evs[j]
                if event.characters & other.characters or event.locations & other.locations:
                    if other.id not in graph[event.id]:
                        graph[event.id].append(other.id)

        return graph

    def get_random_time_point(self, seed: Optional[int] = None) -> TimePoint:
        """
        Return a random TimePoint uniformly distributed across the loaded timeline.

        Uses linear interpolation between the two closest events to avoid the
        float→TimePoint decode ambiguity of pure arithmetic.
        """
        import random
        if seed is not None:
            random.seed(seed)

        sorted_evs = self._get_sorted_events()
        if not sorted_evs:
            return TimePoint(year=1900)

        t0 = self._time_to_float(self.book_start)
        t1 = self._time_to_float(self.book_end)
        if t0 == t1:
            return self.book_start

        target_h = t0 + random.random() * (t1 - t0)
        return self._float_to_timepoint(target_h)

    # ── Internal helpers ──────────────────────────────────────────────────────

    # Slot sizes (hours) — must be consistent between encode and decode.
    _HOURS_PER_YEAR  = 365 * 24          # 8760
    _HOURS_PER_MONTH = 30  * 24          # 720
    _HOURS_PER_DAY   = 24

    def _time_to_float(self, t: TimePoint) -> float:
        """
        Encode a TimePoint to hours.

        Layout (all components 0-based before shifting):
          H = year * 8760 + (month-1) * 720 + (day-1) * 24 + hour

        Ordering is guaranteed: t1 < t2  ⟺  _time_to_float(t1) < _time_to_float(t2).
        Max sub-year value: 11*720 + 29*24 + 23 = 8639 < 8760, so no year overflow.
        """
        return (
              t.year               * self._HOURS_PER_YEAR
            + (t.month - 1)        * self._HOURS_PER_MONTH
            + (t.day   - 1)        * self._HOURS_PER_DAY
            + t.hour
        )

    def _float_to_timepoint(self, h: float) -> TimePoint:
        """
        Exact inverse of _time_to_float.

        Decode sequence (mirrors the encode layout):
          year  = h // 8760
          r     = h % 8760     → 0..8639
          month = r // 720 + 1
          r2    = r % 720      → 0..719
          day   = r2 // 24 + 1
          hour  = r2 % 24
        """
        h = max(0, int(h))
        year    = h // self._HOURS_PER_YEAR
        r       = h  % self._HOURS_PER_YEAR
        month   = r  // self._HOURS_PER_MONTH + 1
        r2      = r  %  self._HOURS_PER_MONTH
        day     = r2 // self._HOURS_PER_DAY   + 1
        hour    = r2 %  self._HOURS_PER_DAY

        # Clamp to valid calendar ranges (safety net for edge floats)
        month = max(1, min(12, month))
        day   = max(1, min(30, day))

        return TimePoint(year=year, month=month, day=day, hour=hour)
