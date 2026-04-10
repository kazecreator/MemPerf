"""
Event Generator - Build rich event datasets using event graphs.

The TMGenerator creates synthetic but coherent events using three graphs:
  1. CharacterGraph - who is involved with whom
  2. TemporalGraph - what happened when and why
  3. SpatialContext - where things happened

Supported input modes:
  - Natural language description → LLM parses and generates graph structure
  - Structured seed → characters, locations, timeline outline provided
  - Existing events → expand N events into M events (M > N)
  - Contradiction injection → add conflicting facts to test memory systems
  - Density curve → configurable front/center/back loaded
  - Dialog import → convert chat logs into events
"""

import random
import json
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

from engine.event_graph import (
    CharacterGraph, TemporalGraph, SpatialContext,
    Character, RelationType, EventRelation, Location
)
from engine.time_machine import TimePoint, MemoryEvent


# ─────────────────────────────────────────────────────────────────────────────
# Generator Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeneratorConfig:
    """Parameters for event generation"""
    # Quantity
    num_events: int = 30
    num_characters: int = 8
    num_locations: int = 5

    # Time span
    time_span_days: int = 365

    # Theme / domain
    theme: str = "general"  # general, work, travel, medical, conflict, startup, academic

    # Density curve: how events are distributed over time
    # "uniform", "front_loaded", "back_loaded", "center_loaded"
    density_curve: str = "center_loaded"

    # Relationship complexity
    # "simple" (linear chain), "network" (everyone connected), "cluster" ( subgroups)
    relationship_complexity: str = "network"

    # Inject contradictions (for testing memory conflict detection)
    inject_contradictions: bool = False
    num_contradictions: int = 2

    # Random seed for reproducibility
    seed: int = 42

    # Include characters in events (for question generation)
    use_characters: bool = True
    use_locations: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Theme Templates
# ─────────────────────────────────────────────────────────────────────────────

THEME_TEMPLATES = {
    "general": {
        "character_types": ["engineer", "designer", "manager", "researcher", "consultant"],
        "location_types": ["office", "home", "restaurant", "city park", "airport", "hotel"],
        "event_kinds": ["meeting", "conversation", "decision", "milestone", "conflict", "collaboration"],
        "moods": ["productive", "tense", "celebratory", "routine", "chaotic"],
    },
    "startup": {
        "character_types": ["founder", "investor", "engineer", "designer", "customer", "advisor"],
        "location_types": ["office", "co-working space", "investor meeting room", "cafe", "airport"],
        "event_kinds": ["funding", "product launch", "team hiring", "pivot", "conflict", "partnership"],
        "moods": ["exciting", "stressful", "triumphant", "desperate", "hopeful"],
    },
    "medical": {
        "character_types": ["doctor", "nurse", "patient", "family member", "specialist"],
        "location_types": ["hospital room", "clinic", "waiting room", "home", "pharmacy"],
        "event_kinds": ["diagnosis", "treatment", "recovery", "family meeting", "complication", "discharge"],
        "moods": ["anxious", "relieved", "gloomy", "hopeful", "routine"],
    },
    "travel": {
        "character_types": ["traveler", "local guide", "hotel staff", "fellow traveler", "family"],
        "location_types": ["hotel", "restaurant", "tourist site", "airport", "train station", "market"],
        "event_kinds": ["arrival", "sightseeing", "meal", "incident", "departure", "conversation"],
        "moods": ["adventurous", "tired", "delighted", "frustrated", "relaxed"],
    },
    "academic": {
        "character_types": ["professor", "PhD student", "postdoc", "department head", "colleague"],
        "location_types": ["lab", "conference room", "classroom", "library", "coffee shop", "home office"],
        "event_kinds": ["paper submission", "experiment", "seminar", "grant application", "collaboration", "conference"],
        "moods": ["focused", "anxious", "excited", "frustrated", "triumphant"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Main Generator
# ─────────────────────────────────────────────────────────────────────────────

class TMGenerator:
    """
    Generates synthetic but coherent event sequences using event graphs.

    Usage:

      # Basic generation
      gen = TMGenerator()
      events = gen.generate(config=GeneratorConfig(num_events=50, theme="startup"))

      # From structured seed
      seed = {
          "characters": [{"id": "C1", "name": "Alice"}, {"id": "C2", "name": "Bob"}],
          "locations": [{"id": "L1", "name": "SF Office"}],
          "outline": ["Alice starts company", "Bob joins 2 months later"]
      }
      events = gen.generate_from_seed(seed, config)

      # Expand existing events
      events = gen.expand_events(existing_events, target=100)

      # Inject contradictions
      events = gen.inject_contradictions(events, count=3)
    """

    def __init__(self):
        self.char_graph = CharacterGraph()
        self.temp_graph = TemporalGraph()
        self.spatial = SpatialContext()
        self._config: Optional[GeneratorConfig] = None
        self._id_counter = 0

    def _next_id(self, prefix: str = "E") -> str:
        self._id_counter += 1
        return f"{prefix}{self._id_counter:04d}"

    # ── Public API ─────────────────────────────────────────────────────────

    def generate(self, config: Optional[GeneratorConfig] = None) -> List[MemoryEvent]:
        """Generate events from scratch using a theme-based config"""
        self._config = config or GeneratorConfig()
        random.seed(self._config.seed)

        # Step 1: Build character graph
        characters = self._build_character_graph()

        # Step 2: Build spatial context
        locations = self._build_spatial_context()

        # Step 3: Generate timeline and events
        events = self._generate_events(characters, locations)

        # Step 4: Link events temporally
        self._build_temporal_links(events)

        # Step 5: Optionally inject contradictions
        if self._config.inject_contradictions:
            events = self._inject_contradictions(events)

        return events

    def generate_from_seed(self, seed: Dict[str, Any],
                          config: Optional[GeneratorConfig] = None) -> List[MemoryEvent]:
        """
        Generate events from a structured seed.

        seed = {
            "characters": [{"id": "C1", "name": "Alice", "role": "founder", ...}],
            "locations": [{"id": "L1", "name": "SF Office", "type": "office"}],
            "outline": ["Event title 1", "Event title 2", ...],  # ordered list
            "relations": [{"from": "C1", "to": "C2", "type": "colleague", "strength": 0.8}],
        }
        """
        self._config = config or GeneratorConfig()
        random.seed(self._config.seed)
        self._reset()

        # Load characters from seed
        for c in seed.get("characters", []):
            self.char_graph.add_character(
                char_id=c["id"],
                name=c.get("name", c["id"]),
                role=c.get("role", "unknown"),
                traits=c.get("traits", []),
                goals=c.get("goals", [])
            )

        # Load locations from seed
        for loc in seed.get("locations", []):
            self.spatial.add_location(
                loc_id=loc["id"],
                name=loc.get("name", loc["id"]),
                loc_type=loc.get("type", "generic"),
                significance=loc.get("significance", 0.5)
            )

        # Build relations
        for rel in seed.get("relations", []):
            self.char_graph.add_relation(
                rel["from"], rel["to"],
                RelationType(rel.get("type", "acquaintance")),
                strength=rel.get("strength", 0.5)
            )

        # Generate events from outline or from nothing
        outline = seed.get("outline", [])
        characters = list(self.char_graph.characters.values())
        locations = list(self.spatial.locations.values())

        if outline:
            events = self._generate_from_outline(outline, characters, locations)
        else:
            events = self._generate_events(characters, locations)

        self._build_temporal_links(events)

        return events

    def expand_events(self, existing: List[MemoryEvent],
                      target: int) -> List[MemoryEvent]:
        """Expand N events into a larger, richer event set"""
        self._reset()
        self._config = GeneratorConfig(num_events=target)

        # Index existing events into graphs
        for ev in existing:
            self._index_existing_event(ev)

        # Determine what's missing
        chars = list(self.char_graph.characters.values())
        locs = list(self.spatial.locations.values())

        # Generate new events to fill the gap
        need = target - len(existing)
        new_events = self._generate_events(chars, locs, start_idx=len(existing), count=need)

        # Rebuild temporal links across all events
        all_events = existing + new_events
        self._rebuild_temporal_links(all_events)

        return new_events

    def inject_contradictions(self, events: List[MemoryEvent],
                             count: int = 2) -> List[MemoryEvent]:
        """
        Inject contradictory facts into events.

        Example contradiction:
          Event A: "Alice worked at Google from 2018-2020"
          Event B: "Alice worked at Microsoft from 2019-2021"
          (Temporal overlap, contradictory employment)

        This tests whether memory systems can detect and flag conflicts.
        """
        if len(events) < 4:
            return events

        # Pick random pairs to make contradictory
        contradiction_pairs = []
        chars = list(set(c for e in events for c in e.characters))
        locs = list(set(l for e in events for l in e.locations))

        for _ in range(count):
            if len(chars) < 2:
                break
            # Pick two events with different times and shared characters
            e1 = random.choice(events)
            e2 = random.choice([e for e in events if abs(
                self._date_to_ord(e1.time) - self._date_to_ord(e.time)
            ) > 30 and e.id != e1.id])
            if not e2:
                continue

            # Add contradictory fact
            # For now, modify e2's content to contradict e1
            char = random.choice(list(e1.characters)) if e1.characters else None
            if char:
                e2.content += f" [CONTRADICTION NOTE: {char} had conflicting accounts about events]"
                self.temp_graph.inject_contradiction(e1.id, e2.id)
                contradiction_pairs.append((e1.id, e2.id))

        return events

    # ── Internal ───────────────────────────────────────────────────────────

    def _reset(self):
        self.char_graph = CharacterGraph()
        self.temp_graph = TemporalGraph()
        self.spatial = SpatialContext()
        self._id_counter = 0

    def _build_character_graph(self) -> List[Character]:
        cfg = self._config
        theme = THEME_TEMPLATES.get(cfg.theme, THEME_TEMPLATES["general"])
        char_types = theme["character_types"]

        characters = []
        for i in range(cfg.num_characters):
            cid = f"C{i+1:02d}"
            name = f"{char_types[i % len(char_types)]} {chr(65+i)}"  # Engineer A, Engineer B...
            role = "protagonist" if i == 0 else ("supporting" if i < 3 else "minor")

            self.char_graph.add_character(
                char_id=cid, name=name, role=role
            )
            characters.append(self.char_graph.characters[cid])

        # Build relationships based on complexity
        self._build_relationships(characters)

        return characters

    def _build_relationships(self, characters: List[Character]):
        cfg = self._config
        complexity = cfg.relationship_complexity

        if complexity == "simple":
            # Linear chain: each person only connected to next
            for i in range(len(characters) - 1):
                self.char_graph.add_relation(
                    characters[i].id, characters[i+1].id,
                    RelationType.ACQUAINTANCE, strength=0.3
                )

        elif complexity == "network":
            # Everyone connected to protagonist (C1)
            for c in characters[1:]:
                rel_type = random.choice([
                    RelationType.COLLEAGUE, RelationType.FRIEND,
                    RelationType.FAMILY, RelationType.ACQUAINTANCE
                ])
                self.char_graph.add_relation(
                    characters[0].id, c.id, rel_type,
                    strength=random.uniform(0.4, 0.9)
                )
            # Some peer connections
            for i in range(len(characters)):
                for j in range(i+1, min(i+3, len(characters))):
                    if random.random() < 0.4:
                        self.char_graph.add_relation(
                            characters[i].id, characters[j].id,
                            RelationType.ACQUAINTANCE, strength=random.uniform(0.2, 0.5)
                        )

        elif complexity == "cluster":
            # Two subgroups
            half = len(characters) // 2
            for i in range(half):
                for j in range(i+1, half):
                    if random.random() < 0.6:
                        self.char_graph.add_relation(
                            characters[i].id, characters[j].id,
                            RelationType.FRIEND, strength=random.uniform(0.5, 0.9)
                        )
            for i in range(half, len(characters)):
                for j in range(i+1, len(characters)):
                    if random.random() < 0.6:
                        self.char_graph.add_relation(
                            characters[i].id, characters[j].id,
                            RelationType.COLLEAGUE, strength=random.uniform(0.3, 0.7)
                        )

    def _build_spatial_context(self) -> List[Location]:
        cfg = self._config
        theme = THEME_TEMPLATES.get(cfg.theme, THEME_TEMPLATES["general"])
        loc_types = theme["location_types"]

        locations = []
        for i in range(cfg.num_locations):
            lid = f"L{i+1:02d}"
            name = f"{loc_types[i % len(loc_types)]} {chr(65+i)}"
            self.spatial.add_location(
                loc_id=lid, name=name,
                loc_type=loc_types[i % len(loc_types)],
                significance=random.uniform(0.3, 0.9)
            )
            locations.append(self.spatial.locations[lid])

        return locations

    def _generate_events(self, characters: List[Character],
                        locations: List[Location],
                        start_idx: int = 0,
                        count: Optional[int] = None) -> List[MemoryEvent]:
        cfg = self._config
        theme = THEME_TEMPLATES.get(cfg.theme, THEME_TEMPLATES["general"])
        event_kinds = theme["event_kinds"]
        moods = theme["moods"]

        n = count or cfg.num_events
        density = cfg.density_curve

        # Compute time distribution
        time_points = self._distribute_times(n, density)

        events = []
        protagonist = characters[0] if characters else None

        for i in range(n):
            eid = f"E{start_idx + i + 1:04d}"
            time_str = time_points[i]
            event_kind = random.choice(event_kinds)
            mood = random.choice(moods)

            # Pick characters involved
            if characters and cfg.use_characters:
                # Protagonist always involved
                involved = [protagonist] if protagonist else []
                # Add 1-3 others based on relationship strength
                for c in characters[1:]:
                    if c not in involved and random.random() < 0.6:
                        involved.append(c)
                    if len(involved) >= 4:
                        break
            else:
                involved = []

            # Pick locations
            if locations and cfg.use_locations:
                involved_locs = random.sample(
                    locations, min(random.randint(1, 2), len(locations))
                )
            else:
                involved_locs = []

            # Generate event title and content
            title = self._generate_event_title(event_kind, involved, protagonist)
            content = self._generate_event_content(event_kind, title, involved, involved_locs, mood)

            event = MemoryEvent(
                id=eid,
                time=TimePoint.from_string(time_str),
                title=title,
                content=content,
                characters={c.name for c in involved},
                locations={loc.name for loc in involved_locs},
                tags={event_kind},
                emotional_valence=self._mood_to_valence(mood),
            )

            events.append(event)

            # Index into graphs
            self.temp_graph.add_event(
                event_id=eid, time_str=time_str,
                title=title, summary=content[:100],
                importance=random.uniform(0.3, 0.9),
                mood=mood,
                characters=[c.name for c in involved],
                locations=[loc.name for loc in involved_locs]
            )
            self.spatial.record_event_at(eid, [loc.id for loc in involved_locs])
            for c in involved:
                self.char_graph.link_event_to_character(c.id, eid)

        return events

    def _generate_from_outline(self, outline: List[str],
                               characters: List[Character],
                               locations: List[Location]) -> List[MemoryEvent]:
        """Generate events from a textual outline"""
        n = len(outline)
        time_points = self._distribute_times(n, "uniform")
        protagonist = characters[0] if characters else None

        events = []
        for i, title in enumerate(outline):
            eid = f"E{i+1:04d}"
            time_str = time_points[i]

            # Pick random subset of characters
            involved = []
            if characters and protagonist:
                involved = [protagonist]
                for c in random.sample(characters[1:], min(random.randint(0, 3), len(characters)-1)):
                    involved.append(c)

            locs = random.sample(locations, min(1, len(locations))) if locations else []

            content = f"{title}. This was a significant event."

            event = MemoryEvent(
                id=eid,
                time=TimePoint.from_string(time_str),
                title=title,
                content=content,
                characters={c.name for c in involved},
                locations={loc.name for loc in locs},
                tags=set(),
                emotional_valence=0.0,
            )
            events.append(event)

            self.temp_graph.add_event(
                eid, time_str, title, content[:100],
                characters=[c.name for c in involved],
                locations=[loc.name for loc in locs]
            )

        return events

    def _distribute_times(self, n: int, density: str) -> List[str]:
        """Distribute n events over time_span_days according to density curve"""
        cfg = self._config
        span = cfg.time_span_days

        if density == "uniform":
            offsets = sorted(random.sample(range(span), min(n, span)))

        elif density == "front_loaded":
            # More events early
            early = max(1, int(n * 0.6))
            late = n - early
            early_offsets = sorted(random.sample(range(int(span * 0.5)), early))
            late_offsets = sorted([
                int(span * 0.5) + random.randint(0, int(span * 0.5))
                for _ in range(late)
            ])
            offsets = sorted(early_offsets + late_offsets)

        elif density == "back_loaded":
            # More events late
            late = max(1, int(n * 0.6))
            early = n - late
            early_offsets = sorted([
                random.randint(0, int(span * 0.5)) for _ in range(early)
            ])
            late_offsets = sorted(random.sample(range(int(span * 0.5), span), late))
            offsets = sorted(early_offsets + late_offsets)

        elif density == "center_loaded":
            # Events concentrated in middle period
            center_start = span // 4
            center_end = span * 3 // 4
            center = n - max(2, n // 5)
            wings = n - center
            center_offsets = sorted(random.sample(
                range(center_start, center_end), center
            ))
            wing_early = sorted(random.sample(range(center_start), wings // 2))
            wing_late = sorted([
                center_end + random.randint(0, span - center_end)
                for _ in range(wings - wings // 2)
            ])
            offsets = sorted(wing_early + center_offsets + wing_late)

        else:
            offsets = sorted(random.sample(range(span), min(n, span)))

        # Convert to YYYY-MM-DD strings from base date 2024-01-01
        from datetime import date, timedelta
        base = date(2024, 1, 1)
        return [(base + timedelta(days=o)).isoformat() for o in offsets]

    def _generate_event_title(self, kind: str, characters: List[Character],
                              protagonist: Optional[Character]) -> str:
        templates = {
            "meeting": ["{char} meets with team about {topic}",
                         "Project kickoff meeting hosted by {char}"],
            "conversation": ["{char} has a one-on-one with {other}",
                           "Coffee chat between {char} and {other}"],
            "decision": ["{char} makes final call on {topic}",
                        "Key decision: {topic} approved by {char}"],
            "milestone": ["{char} reaches major milestone: {topic}",
                         "Celebration: {topic} successfully completed"],
            "conflict": ["Tension rises between {char} and {other} over {topic}",
                        "{char} and {other} disagree on approach to {topic}"],
            "collaboration": ["{char} and {other} work together on {topic}",
                             "Cross-team collaboration on {topic} begins"],
            "funding": ["{char} closes seed round for {topic}",
                       "Investment secured for {topic}"],
            "product launch": ["{char} leads {topic} launch",
                             "{topic} goes live after months of work"],
            "team hiring": ["{char} brings new hire for {topic}",
                           "Key hire: {other} joins the {topic} team"],
            "pivot": ["{char} announces strategic pivot to {topic}",
                     "Company shifts focus to {topic}"],
            "partnership": ["{char} signs partnership agreement for {topic}",
                           "Strategic partnership formed around {topic}"],
            "arrival": ["{char} arrives at {location}",
                       "Welcome: {char} lands in new city"],
            "sightseeing": ["{char} visits {location}",
                          "{char} explores local attractions"],
            "meal": ["{char} dines at {location}",
                    "Dinner at {location} with team"],
            "incident": ["Unexpected incident during {topic}",
                        "{char} deals with urgent situation"],
            "departure": ["{char} leaves for {location}",
                         "{char} departs after successful trip"],
            "diagnosis": ["{char} receives medical update",
                        "Doctor reviews {char}'s test results"],
            "treatment": ["{char} begins new treatment plan",
                         "Medical procedure for {char}"],
            "recovery": ["{char} shows positive recovery signs",
                        "Progress in {char}'s healing journey"],
            "family meeting": ["Family meeting about {char}'s care",
                              "{char} discusses treatment options"],
            "complication": ["Unexpected complication for {char}",
                           "Medical team addresses {char}'s setback"],
            "discharge": ["{char} discharged from hospital",
                         "{char} returns home after treatment"],
            "paper submission": ["{char} submits paper on {topic}",
                                "{char} sends {topic} manuscript to journal"],
            "experiment": ["{char} runs key experiment on {topic}",
                          "Lab work: {char} tests hypothesis on {topic}"],
            "seminar": ["{char} attends seminar on {topic}",
                       "{char} presents at {topic} seminar"],
            "grant application": ["{char} submits grant proposal for {topic}",
                                "{char} applies for funding for {topic}"],
            "conference": ["{char} attends {topic} conference",
                          "{char} presents poster at {topic} conference"],
        }

        topics = ["Q1 roadmap", "user research", "product strategy", "budget planning",
                  "team dynamics", "client feedback", "technical debt", "hiring plan"]
        topic = random.choice(topics)

        char = protagonist.name if protagonist else "Team"
        other = characters[1].name if len(characters) > 1 else "colleague"
        location = "new office" if not characters else f"{characters[0].name}'s workspace"

        kind_templates = templates.get(kind, templates["meeting"])
        tpl = random.choice(kind_templates)
        return tpl.format(char=char, other=other, topic=topic, location=location)

    # Content templates per event kind.
    # Written WITHOUT the event title so that keyword systems cannot trivially
    # win by matching the title tokens embedded in the question.
    _CONTENT_TEMPLATES = {
        "meeting": [
            "{chars} gathered at {loc} to align on priorities. {detail}",
            "A session at {loc} brought together {chars}. {detail}",
            "{chars} assembled at {loc} to work through outstanding issues. {detail}",
        ],
        "conversation": [
            "{chars} exchanged views at {loc}. {detail}",
            "At {loc}, {chars} spoke at length about current concerns. {detail}",
            "An informal exchange unfolded between {chars} at {loc}. {detail}",
        ],
        "decision": [
            "{chars} reached a conclusion at {loc}. {detail}",
            "After deliberation at {loc}, {chars} settled on a course of action. {detail}",
            "A final call was made by {chars} at {loc}. {detail}",
        ],
        "milestone": [
            "{chars} achieved a significant milestone at {loc}. {detail}",
            "Progress culminated for {chars} at {loc}. {detail}",
            "A key goal was reached at {loc} by {chars}. {detail}",
        ],
        "conflict": [
            "{chars} found themselves at odds at {loc}. {detail}",
            "Disagreement surfaced between {chars} at {loc}. {detail}",
            "Tensions emerged among {chars} at {loc}. {detail}",
        ],
        "collaboration": [
            "{chars} combined their efforts at {loc}. {detail}",
            "Joint work between {chars} began at {loc}. {detail}",
            "{chars} coordinated closely at {loc}. {detail}",
        ],
        "funding": [
            "{chars} secured new resources at {loc}. {detail}",
            "A financial commitment was reached at {loc} involving {chars}. {detail}",
            "{chars} finalised an investment deal at {loc}. {detail}",
        ],
        "product launch": [
            "{chars} unveiled something new at {loc}. {detail}",
            "A new release went live, spearheaded by {chars} from {loc}. {detail}",
            "{chars} completed a long-awaited launch at {loc}. {detail}",
        ],
        "team hiring": [
            "{chars} welcomed a new addition at {loc}. {detail}",
            "The group at {loc} grew as {chars} finalised a key hire. {detail}",
            "{chars} brought on a new member at {loc}. {detail}",
        ],
        "pivot": [
            "{chars} shifted direction at {loc}. {detail}",
            "A strategic change was announced by {chars} at {loc}. {detail}",
            "{chars} decided to change course at {loc}. {detail}",
        ],
        "partnership": [
            "{chars} formalised a new alliance at {loc}. {detail}",
            "An agreement between {chars} was reached at {loc}. {detail}",
            "{chars} signed a collaboration agreement at {loc}. {detail}",
        ],
        "arrival": [
            "{chars} reached {loc} after a journey. {detail}",
            "The group including {chars} arrived at {loc}. {detail}",
            "{chars} checked in at {loc}. {detail}",
        ],
        "sightseeing": [
            "{chars} explored {loc}. {detail}",
            "Time was spent by {chars} taking in the sights at {loc}. {detail}",
            "{chars} spent the day visiting {loc}. {detail}",
        ],
        "meal": [
            "{chars} sat down to eat at {loc}. {detail}",
            "Over a meal at {loc}, {chars} gathered. {detail}",
            "{chars} shared a table at {loc}. {detail}",
        ],
        "incident": [
            "{chars} dealt with an unexpected situation at {loc}. {detail}",
            "Something unplanned occurred for {chars} at {loc}. {detail}",
            "{chars} had to respond to an urgent matter at {loc}. {detail}",
        ],
        "departure": [
            "{chars} left {loc} to continue their journey. {detail}",
            "The group including {chars} departed from {loc}. {detail}",
            "{chars} wrapped up and headed out from {loc}. {detail}",
        ],
        "diagnosis": [
            "Medical staff reviewed the situation of {chars} at {loc}. {detail}",
            "{chars} received a clinical update at {loc}. {detail}",
            "Test results were discussed with {chars} at {loc}. {detail}",
        ],
        "treatment": [
            "{chars} underwent a new procedure at {loc}. {detail}",
            "Care was provided to {chars} at {loc}. {detail}",
            "A treatment plan was started for {chars} at {loc}. {detail}",
        ],
        "recovery": [
            "{chars} showed signs of improvement at {loc}. {detail}",
            "Progress in recovery was noted for {chars} at {loc}. {detail}",
            "The situation improved for {chars} at {loc}. {detail}",
        ],
        "family meeting": [
            "{chars} gathered at {loc} to discuss personal matters. {detail}",
            "A private gathering of {chars} took place at {loc}. {detail}",
            "{chars} sat down together at {loc} for a family discussion. {detail}",
        ],
        "complication": [
            "An unexpected setback arose for {chars} at {loc}. {detail}",
            "{chars} faced difficulties at {loc}. {detail}",
            "Things became more complicated for {chars} at {loc}. {detail}",
        ],
        "discharge": [
            "{chars} were cleared to leave {loc}. {detail}",
            "Permission to depart was granted to {chars} at {loc}. {detail}",
            "{chars} completed their stay at {loc}. {detail}",
        ],
        "paper submission": [
            "{chars} completed a major written work at {loc}. {detail}",
            "Months of effort culminated in a submission by {chars} at {loc}. {detail}",
            "{chars} sent off their manuscript from {loc}. {detail}",
        ],
        "experiment": [
            "{chars} ran tests at {loc}. {detail}",
            "Laboratory work progressed for {chars} at {loc}. {detail}",
            "{chars} conducted a series of trials at {loc}. {detail}",
        ],
        "seminar": [
            "{chars} attended a presentation at {loc}. {detail}",
            "Knowledge was shared among {chars} at {loc}. {detail}",
            "{chars} participated in a knowledge-sharing session at {loc}. {detail}",
        ],
        "grant application": [
            "{chars} submitted a proposal from {loc}. {detail}",
            "Funding was sought by {chars} at {loc}. {detail}",
            "{chars} put in an application for support at {loc}. {detail}",
        ],
        "conference": [
            "{chars} represented their work at {loc}. {detail}",
            "Peers gathered with {chars} at {loc} for an academic exchange. {detail}",
            "{chars} presented findings to colleagues at {loc}. {detail}",
        ],
    }

    _MOOD_DETAILS = {
        "productive":   "Great progress was made.",
        "tense":        "There was friction, but it was eventually resolved.",
        "celebratory":  "Everyone was excited and relieved.",
        "routine":      "The day proceeded without incident.",
        "chaotic":      "Things were disorganised but ultimately resolved.",
        "stressful":    "Long hours and high pressure shaped the atmosphere.",
        "triumphant":   "A major success was achieved.",
        "desperate":    "Difficult decisions had to be made under pressure.",
        "hopeful":      "There was renewed optimism about what lay ahead.",
        "anxious":      "Uncertainty about the outcome caused visible concern.",
        "relieved":     "A burden was lifted after the event concluded.",
        "gloomy":       "The mood remained somber throughout.",
        "excited":      "Enthusiasm ran high among everyone present.",
        "frustrated":   "Several obstacles made progress difficult.",
        "relaxed":      "The atmosphere was calm and unhurried.",
        "focused":      "Everyone present was deeply concentrated on the task.",
        "adventurous":  "An unexpected element made things interesting.",
        "tired":        "Fatigue was visible, though spirits stayed up.",
        "delighted":    "People left with a sense of real satisfaction.",
    }

    def _generate_event_content(self, kind: str, title: str,
                                 characters: List[Character],
                                 locations: List[Location],
                                 mood: str) -> str:
        chars  = ", ".join(c.name for c in characters[:3]) if characters else "the team"
        loc    = locations[0].name if locations else "the venue"
        detail = self._MOOD_DETAILS.get(mood, "The event proceeded as planned.")

        templates = self._CONTENT_TEMPLATES.get(kind, [
            "{chars} came together at {loc}. {detail}",
            "An event involving {chars} took place at {loc}. {detail}",
        ])
        tpl = random.choice(templates)
        return tpl.format(chars=chars, loc=loc, detail=detail)

    def _mood_to_valence(self, mood: str) -> float:
        valence_map = {
            "productive": 0.4, "celebratory": 0.7, "hopeful": 0.6, "excited": 0.7,
            "triumphant": 0.8, "relieved": 0.5, "delighted": 0.7, "relaxed": 0.5,
            "focused": 0.2, "adventurous": 0.4,
            "tense": -0.3, "desperate": -0.6, "anxious": -0.4, "frustrated": -0.4,
            "gloomy": -0.5, "chaotic": -0.2, "stressful": -0.4,
            "tired": -0.1, "routine": 0.0,
        }
        return valence_map.get(mood, 0.0)

    def _date_to_ord(self, tp: TimePoint) -> int:
        """Convert TimePoint to ordinal for comparison"""
        return tp.year * 365 + tp.month * 30 + tp.day

    def _build_temporal_links(self, events: List[MemoryEvent]):
        """Create PRECEDES and CAUSED_BY links between events"""
        if not events:
            return

        sorted_events = sorted(events, key=lambda e: (e.time.year, e.time.month, e.time.day))

        for i in range(len(sorted_events) - 1):
            e1 = sorted_events[i]
            e2 = sorted_events[i + 1]

            # Always add temporal precedence
            self.temp_graph.link(e1.id, e2.id, EventRelation.PRECEDES, weight=1.0)

            # Add location co-occurrence link
            if e1.locations & e2.locations:
                self.temp_graph.link(e1.id, e2.id, EventRelation.SAME_LOCATION, weight=0.5)

            # Add shared character link
            if e1.characters & e2.characters:
                self.temp_graph.link(e1.id, e2.id, EventRelation.SHARES_CHARACTERS, weight=0.7)

            # Occasionally add causal link (for important events)
            if random.random() < 0.3:
                self.temp_graph.link(e1.id, e2.id, EventRelation.CAUSED_BY, weight=0.5)

    def _rebuild_temporal_links(self, events: List[MemoryEvent]):
        """Rebuild all temporal links for an event set"""
        self.temp_graph = TemporalGraph()
        for ev in events:
            self.temp_graph.add_event(
                ev.id, str(ev.time), ev.title, ev.content[:100],
                characters=list(ev.characters),
                locations=list(ev.locations)
            )
        self._build_temporal_links(events)

    def _index_existing_event(self, ev: MemoryEvent):
        """Add an existing event to the graphs"""
        # Add characters
        for char_name in ev.characters:
            cid = f"C_{char_name[:10]}"
            if cid not in self.char_graph.characters:
                self.char_graph.add_character(cid, char_name)

        # Add locations
        for loc_name in ev.locations:
            lid = f"L_{loc_name[:10]}"
            if lid not in self.spatial.locations:
                self.spatial.add_location(lid, loc_name)

        # Add to temporal graph
        self.temp_graph.add_event(
            ev.id, str(ev.time), ev.title, ev.content[:100],
            characters=list(ev.characters),
            locations=list(ev.locations)
        )

    # ── Diagnostics ───────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "characters": self.char_graph.summary(),
            "temporal": self.temp_graph.summary(),
            "spatial": self.spatial.summary(),
        }
