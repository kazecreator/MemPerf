"""
Event Graph - Three interconnected graph structures for memory systems.

- CharacterGraph: tracks who interacted with whom, relationship types
- TemporalGraph: event causal chains, dependencies, ordering
- SpatialContext: location networks, event co-occurrence

These three graphs form the backbone of the event generation engine.
"""

from dataclasses import dataclass, field
from typing import Set, Dict, List, Optional, Tuple
from enum import Enum
import random


# ─────────────────────────────────────────────────────────────────────────────
# Relationship Types
# ─────────────────────────────────────────────────────────────────────────────

class RelationType(Enum):
    FRIEND = "friend"
    COLLEAGUE = "colleague"
    FAMILY = "family"
    ROMANTIC = "romantic"
    CONFLICT = "conflict"
    MENTOR = "mentor"
    ACQUAINTANCE = "acquaintance"
    STRANGER = "stranger"


class EventRelation(Enum):
    CAUSED_BY = "caused_by"        # causal chain
    PRECEDES = "precedes"          # temporal ordering
    SAME_LOCATION = "same_location" # spatial co-occurrence
    SHARES_CHARACTERS = "shares_characters"
    CONTRADICTS = "contradicts"     # contradictory event
    REINFORCES = "reinforces"      # supporting event


# ─────────────────────────────────────────────────────────────────────────────
# CharacterGraph
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Character:
    """A person in the event system"""
    id: str
    name: str
    role: str = "unknown"           # protagonist, supporting, minor
    personality_traits: Set[str] = field(default_factory=set)
    goals: Set[str] = field(default_factory=set)


@dataclass
class CharacterRelation:
    """Relationship between two characters"""
    char1: str        # character id
    char2: str        # character id
    relation: RelationType
    strength: float   # 0.0-1.0, how strong the relationship is
    events: List[str] = field(default_factory=list)  # event ids where this relation manifested
    created_at: Optional[str] = None  # time string when this relation was established


class CharacterGraph:
    """
    Tracks character relationships over time.

    Usage:
      cg = CharacterGraph()
      cg.add_character("C1", "Alice", role="protagonist")
      cg.add_character("C2", "Bob", role="supporting")
      cg.add_relation("C1", "C2", RelationType.COLLEAGUE, strength=0.8)
      cg.get_relations("C1")  # → all relations for Alice
    """

    def __init__(self):
        self.characters: Dict[str, Character] = {}
        # {(char1, char2): CharacterRelation}, canonical ordering (lower id first)
        self.relations: Dict[Tuple[str, str], CharacterRelation] = {}
        # {char_id: [event_id]} - which events does this character appear in
        self.character_events: Dict[str, List[str]] = {}

    def add_character(self, char_id: str, name: str, role: str = "unknown",
                       traits: List[str] = None, goals: List[str] = None) -> Character:
        """Add a character to the graph"""
        char = Character(
            id=char_id,
            name=name,
            role=role,
            personality_traits=set(traits or []),
            goals=set(goals or [])
        )
        self.characters[char_id] = char
        self.character_events.setdefault(char_id, [])
        return char

    def add_relation(self, char1: str, char2: str, rel_type: RelationType,
                     strength: float = 0.5, event_id: Optional[str] = None) -> CharacterRelation:
        """Add or update a relationship between two characters"""
        key = (min(char1, char2), max(char1, char2))
        existing = self.relations.get(key)

        if existing:
            # Update existing relationship
            # If new relation type, add it; if same, strengthen
            if existing.relation == rel_type:
                existing.strength = min(1.0, existing.strength + strength * 0.2)
            else:
                # Mixed relationship — keep existing, optionally add event
                pass
            if event_id:
                existing.events.append(event_id)
            return existing
        else:
            rel = CharacterRelation(
                char1=char1, char2=char2,
                relation=rel_type, strength=strength,
                events=[event_id] if event_id else []
            )
            self.relations[key] = rel
            return rel

    def link_event_to_character(self, char_id: str, event_id: str):
        """Record that a character appeared in an event"""
        self.character_events.setdefault(char_id, []).append(event_id)

    def get_character(self, char_id: str) -> Optional[Character]:
        return self.characters.get(char_id)

    def get_relations(self, char_id: str) -> List[CharacterRelation]:
        """Get all relationships for a character"""
        result = []
        for (c1, c2), rel in self.relations.items():
            if c1 == char_id or c2 == char_id:
                result.append(rel)
        return result

    def get_related_characters(self, char_id: str,
                               rel_types: Set[RelationType] = None,
                               min_strength: float = 0.0) -> List[Tuple[str, CharacterRelation]]:
        """Find characters related to this one, optionally filtered"""
        result = []
        for rel in self.get_relations(char_id):
            if rel_types and rel.relation not in rel_types:
                continue
            if rel.strength < min_strength:
                continue
            other = rel.char2 if rel.char1 == char_id else rel.char1
            result.append((other, rel))
        # Sort by strength descending
        result.sort(key=lambda x: x[1].strength, reverse=True)
        return result

    def get_characters_in_event(self, event_id: str) -> List[str]:
        """Find all characters that appeared in an event"""
        return [cid for cid, events in self.character_events.items() if event_id in events]

    def co_occurrence_count(self, char1: str, char2: str) -> int:
        """How many events did two characters appear in together"""
        events1 = set(self.character_events.get(char1, []))
        events2 = set(self.character_events.get(char2, []))
        return len(events1 & events2)

    def summary(self) -> dict:
        return {
            "num_characters": len(self.characters),
            "num_relations": len(self.relations),
            "avg_relations_per_char": (
                sum(len(self.get_relations(c)) for c in self.characters) / max(1, len(self.characters))
            )
        }


# ─────────────────────────────────────────────────────────────────────────────
# TemporalGraph
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EventNode:
    """A node in the temporal event graph"""
    id: str
    time_str: str         # "YYYY-MM-DD"
    title: str
    summary: str
    importance: float      # 0-1, story importance
    mood: str              # positive, negative, neutral
    characters: Set[str] = field(default_factory=set)
    locations: Set[str] = field(default_factory=set)


@dataclass
class TemporalLink:
    """A directed link between events in time"""
    from_event: str
    to_event: str
    relation: EventRelation
    weight: float = 1.0    # strength of connection


class TemporalGraph:
    """
    Tracks temporal ordering and causal relationships between events.

    Usage:
      tg = TemporalGraph()
      tg.add_event("E1", "2024-01-01", "Alice starts new job", ...)
      tg.add_event("E2", "2024-01-15", "Alice gets promoted", ...)
      tg.link("E1", "E2", EventRelation.CAUSED_BY)  # promotion caused by starting job
      tg.get_preceding_events("E2")  # → [E1]
    """

    def __init__(self):
        # {event_id: EventNode}
        self.events: Dict[str, EventNode] = {}
        # adjacency list: {from_id: [(to_id, link)]}
        self.outgoing: Dict[str, List[TemporalLink]] = {}
        self.incoming: Dict[str, List[TemporalLink]] = {}
        # {event_id: list of event_ids sorted by time}
        self._by_time_cache: Optional[List[str]] = None

    def add_event(self, event_id: str, time_str: str, title: str,
                  summary: str = "", importance: float = 0.5,
                  mood: str = "neutral",
                  characters: List[str] = None,
                  locations: List[str] = None) -> EventNode:
        """Add an event to the temporal graph"""
        node = EventNode(
            id=event_id,
            time_str=time_str,
            title=title,
            summary=summary,
            importance=importance,
            mood=mood,
            characters=set(characters or []),
            locations=set(locations or [])
        )
        self.events[event_id] = node
        self.outgoing.setdefault(event_id, [])
        self.incoming.setdefault(event_id, [])
        self._by_time_cache = None  # invalidate cache
        return node

    def link(self, from_event: str, to_event: str,
             relation: EventRelation, weight: float = 1.0) -> TemporalLink:
        """Create a directed link between two events"""
        link = TemporalLink(from_event=from_event, to_event=to_event,
                            relation=relation, weight=weight)
        self.outgoing.setdefault(from_event, []).append(link)
        self.incoming.setdefault(to_event, []).append(link)
        return link

    def get_event(self, event_id: str) -> Optional[EventNode]:
        return self.events.get(event_id)

    def get_sorted_events(self) -> List[EventNode]:
        """Return all events sorted by time"""
        if self._by_time_cache is None:
            self._by_time_cache = sorted(
                self.events.values(),
                key=lambda e: e.time_str
            )
        return self._by_time_cache

    def get_preceding_events(self, event_id: str,
                             relation: EventRelation = None) -> List[Tuple[EventNode, TemporalLink]]:
        """Get events that come before this one (incoming links)"""
        result = []
        for link in self.incoming.get(event_id, []):
            if relation is None or link.relation == relation:
                node = self.events.get(link.from_event)
                if node:
                    result.append((node, link))
        result.sort(key=lambda x: x[0].time_str)
        return result

    def get_following_events(self, event_id: str,
                              relation: EventRelation = None) -> List[Tuple[EventNode, TemporalLink]]:
        """Get events that come after this one (outgoing links)"""
        result = []
        for link in self.outgoing.get(event_id, []):
            if relation is None or link.relation == relation:
                node = self.events.get(link.to_event)
                if node:
                    result.append((node, link))
        result.sort(key=lambda x: x[0].time_str)
        return result

    def get_causal_chain(self, event_id: str, max_depth: int = 5) -> List[List[str]]:
        """
        Find all causal chains leading to this event.
        Returns list of chains, each chain is a list of event_ids.
        """
        chains = []

        def dfs(current: str, path: List[str], depth: int):
            if depth >= max_depth:
                chains.append(list(path))
                return
            preceding = self.get_preceding_events(current, EventRelation.CAUSED_BY)
            if not preceding:
                chains.append(list(path))
                return
            for (node, _) in preceding:
                if node.id not in path:
                    path.append(node.id)
                    dfs(node.id, path, depth + 1)
                    path.pop()

        dfs(event_id, [event_id], 0)
        return chains

    def inject_contradiction(self, event_id1: str, event_id2: str):
        """Mark two events as contradictory (e.g., same person has two conflicting facts)"""
        self.link(event_id1, event_id2, EventRelation.CONTRADICTS, weight=1.0)
        self.link(event_id2, event_id1, EventRelation.CONTRADICTS, weight=1.0)

    def get_contradictions(self, event_id: str) -> List[EventNode]:
        """Find events that contradict this one"""
        result = []
        for (_, link) in self.get_following_events(event_id, EventRelation.CONTRADICTS):
            node = self.events.get(link.to_event)
            if node:
                result.append(node)
        return result

    def density_curve(self) -> Dict[str, float]:
        """
        Return event density per month.
        Useful for visualizing whether story is front-loaded, back-loaded, or uniform.
        """
        from collections import defaultdict
        month_counts = defaultdict(int)
        for event in self.events.values():
            month = event.time_str[:7]  # "YYYY-MM"
            month_counts[month] += 1

        if not month_counts:
            return {}
        max_count = max(month_counts.values())
        return {m: c / max_count for m, c in sorted(month_counts.items())}

    def summary(self) -> dict:
        return {
            "num_events": len(self.events),
            "num_links": sum(len(v) for v in self.outgoing.values()),
            "density_curve": self.density_curve(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SpatialContext
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Location:
    """A physical or abstract location"""
    id: str
    name: str
    type: str = "generic"   # office, home, outdoor, city, etc.
    significance: float = 0.5  # 0-1, how important this location is to the story


class SpatialContext:
    """
    Tracks where events happen and how locations are connected.

    Usage:
      sc = SpatialContext()
      sc.add_location("L1", "Alice's apartment", type="home", significance=0.8)
      sc.add_location("L2", "Office Building", type="office", significance=0.6)
      sc.record_event_at("E1", "L1")
      sc.record_event_at("E1", "L2")  # event spans multiple locations
      sc.get_events_at("L1")  # → all events at Alice's apartment
      sc.get_co_occurring_locations("L1")  # → locations that share events with L1
    """

    def __init__(self):
        self.locations: Dict[str, Location] = {}
        # {location_id: [event_ids]}
        self.location_events: Dict[str, List[str]] = {}
        # {(loc1, loc2): [event_ids]} - events that happened at both locations
        self.location_cooccurrence: Dict[Tuple[str, str], List[str]] = {}

    def add_location(self, loc_id: str, name: str,
                      loc_type: str = "generic",
                      significance: float = 0.5) -> Location:
        loc = Location(id=loc_id, name=name, type=loc_type, significance=significance)
        self.locations[loc_id] = loc
        self.location_events.setdefault(loc_id, [])
        return loc

    def record_event_at(self, event_id: str, location_ids: List[str]):
        """Record that an event occurred at one or more locations"""
        # Update location → events mapping
        for loc_id in location_ids:
            self.location_events.setdefault(loc_id, []).append(event_id)

        # Update co-occurrence (pairwise)
        for i, loc1 in enumerate(location_ids):
            for loc2 in location_ids[i+1:]:
                key = (min(loc1, loc2), max(loc1, loc2))
                self.location_cooccurrence.setdefault(key, []).append(event_id)

    def get_location(self, loc_id: str) -> Optional[Location]:
        return self.locations.get(loc_id)

    def get_events_at(self, location_id: str) -> List[str]:
        """All event IDs that occurred at a location"""
        return self.location_events.get(location_id, [])

    def get_co_occurring_locations(self, location_id: str) -> List[Tuple[str, List[str]]]:
        """
        Find locations that share events with this location.
        Returns [(loc_id, shared_event_ids), ...] sorted by number of shared events.
        """
        result = []
        for (loc1, loc2), event_ids in self.location_cooccurrence.items():
            if loc1 == location_id or loc2 == location_id:
                other = loc2 if loc1 == location_id else loc1
                result.append((other, event_ids))
        result.sort(key=lambda x: len(x[1]), reverse=True)
        return result

    def get_location_sequence(self, character_id: str,
                               character_events: List[str]) -> List[Tuple[str, str]]:
        """
        Get the sequence of locations a character visited.
        Returns [(event_id, location_name), ...] in chronological order.
        """
        # Filter events for this character
        # Returns sorted list of (event_id, primary_location_name)
        result = []
        for event_id in character_events:
            # Find location for this event
            for loc_id, event_list in self.location_events.items():
                if event_id in event_list:
                    loc = self.locations.get(loc_id)
                    if loc:
                        result.append((event_id, loc.name))
                        break
        return result

    def summary(self) -> dict:
        return {
            "num_locations": len(self.locations),
            "avg_events_per_location": (
                sum(len(v) for v in self.location_events.values()) / max(1, len(self.locations))
            ),
            "location_types": list(set(loc.type for loc in self.locations.values())),
        }
