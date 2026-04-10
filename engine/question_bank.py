"""
Question Bank — deterministic, title-free question generation.

Questions are generated from event *attributes* (characters, locations,
date, event kind) — never from the event title.  This prevents keyword
systems from trivially winning by matching title tokens and ensures the
benchmark exercises semantic retrieval.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import random

from engine.time_machine import TimeMachine, TimePoint, MemoryEvent


class QuestionType(Enum):
    FACTUAL_WHO          = "factual_who"
    FACTUAL_WHEN         = "factual_when"
    FACTUAL_WHERE        = "factual_where"
    FACTUAL_WHAT         = "factual_what"
    FACTUAL_COUNT        = "factual_count"
    SEQUENTIAL_BEFORE    = "sequential_before"
    SEQUENTIAL_AFTER     = "sequential_after"
    SEQUENTIAL_ORDER     = "sequential_order"
    TEMPORAL_DURATION    = "temporal_duration"
    CHARACTER_APPEARANCE = "character_appearance"
    CHARACTER_STATE      = "character_state"
    LOCATION_TRACKING    = "location_tracking"
    CAUSAL_REASONING     = "causal_reasoning"


@dataclass
class Question:
    id:                 str
    type:               QuestionType
    difficulty:         int
    question:           str
    answer:             str
    evidence_event_ids: List[str]
    time_point:         Optional[TimePoint]
    metadata:           dict

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "type":       self.type.value,
            "difficulty": self.difficulty,
            "question":   self.question,
            "answer":     self.answer,
            "evidence":   self.evidence_event_ids,
            "time_point": str(self.time_point) if self.time_point else None,
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pchar(ev: MemoryEvent) -> Optional[str]:
    """Primary (alphabetically first) character, or None."""
    return sorted(ev.characters)[0] if ev.characters else None

def _ploc(ev: MemoryEvent) -> Optional[str]:
    """Primary location, or None."""
    return sorted(ev.locations)[0] if ev.locations else None

def _pkind(ev: MemoryEvent) -> str:
    """Primary event kind from tags, or generic fallback."""
    return sorted(ev.tags)[0] if ev.tags else "event"

def _chars_str(ev: MemoryEvent) -> str:
    names = sorted(ev.characters)
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


# ── Question Bank ──────────────────────────────────────────────────────────────

class TemplateQuestionBank:
    """
    Generate deterministic, title-free questions from event attributes.

    Usage:
        bank  = TemplateQuestionBank(time_machine)
        questions = bank.generate(num_questions=30, seed=42)
    """

    def __init__(self, time_machine: TimeMachine):
        self.tm = time_machine
        self._questions: List[Question] = []
        self._qid_counter = 0

    def _next_id(self) -> str:
        self._qid_counter += 1
        return f"TQ{self._qid_counter:04d}"

    def generate(self, num_questions: int = 30,
                 difficulty_range: Tuple[int, int] = (1, 5),
                 seed: int = 42) -> List[Question]:
        random.seed(seed)
        self._questions = []

        all_events = sorted(
            self.tm.events.values(),
            key=lambda e: (e.time.year, e.time.month, e.time.day)
        )
        if len(all_events) < 2:
            return []

        gen_methods = {
            QuestionType.FACTUAL_WHO:          self._gen_factual_who,
            QuestionType.FACTUAL_WHEN:         self._gen_factual_when,
            QuestionType.FACTUAL_WHERE:        self._gen_factual_where,
            QuestionType.FACTUAL_WHAT:         self._gen_factual_what,
            QuestionType.FACTUAL_COUNT:        self._gen_factual_count,
            QuestionType.SEQUENTIAL_BEFORE:    self._gen_sequential_before,
            QuestionType.SEQUENTIAL_AFTER:     self._gen_sequential_after,
            QuestionType.TEMPORAL_DURATION:    self._gen_temporal_duration,
            QuestionType.SEQUENTIAL_ORDER:     self._gen_sequential_order,
            QuestionType.CHARACTER_APPEARANCE: self._gen_character_appearance,
            QuestionType.LOCATION_TRACKING:    self._gen_location_tracking,
        }

        for qtype, method in gen_methods.items():
            self._questions.extend(method(all_events))

        random.shuffle(self._questions)
        self._questions = self._questions[:num_questions]

        for i, q in enumerate(self._questions):
            q.id = self._next_id()

        return self._questions

    # ── Internal factory ──────────────────────────────────────────────────────

    def _q(self, qtype: QuestionType, difficulty: int,
           question: str, answer: str, evidence: List[str],
           time_point: Optional[TimePoint] = None) -> Question:
        return Question(
            id="", type=qtype, difficulty=difficulty,
            question=question, answer=answer,
            evidence_event_ids=evidence,
            time_point=time_point, metadata={},
        )

    def _pick(self, options: list):
        return random.choice(options)

    # ── Generators ────────────────────────────────────────────────────────────
    # All questions describe events via attributes (character, location, date,
    # event kind) — never by quoting the event title.

    def _gen_factual_who(self, events: List[MemoryEvent]) -> List[Question]:
        qs = []
        for ev in events:
            if not ev.characters:
                continue
            kind = _pkind(ev)
            date = str(ev.time)
            loc  = _ploc(ev)
            ans  = _chars_str(ev)

            if loc:
                q = self._pick([
                    f"Who was present at the {kind} held at {loc} on {date}?",
                    f"Who took part in the {kind} at {loc} on {date}?",
                    f"Which people attended the {kind} at {loc} on {date}?",
                ])
            else:
                q = self._pick([
                    f"Who participated in the {kind} on {date}?",
                    f"Who was involved in the {kind} that occurred on {date}?",
                ])
            qs.append(self._q(QuestionType.FACTUAL_WHO, 1, q, ans, [ev.id]))
        return qs

    def _gen_factual_when(self, events: List[MemoryEvent]) -> List[Question]:
        qs = []
        for ev in events:
            char = _pchar(ev)
            kind = _pkind(ev)
            loc  = _ploc(ev)
            if not char:
                continue

            if loc:
                q = self._pick([
                    f"When did {char} attend a {kind} at {loc}?",
                    f"On what date did {char} take part in a {kind} at {loc}?",
                    f"What was the date of {char}'s {kind} at {loc}?",
                ])
            else:
                q = self._pick([
                    f"When did {char} participate in a {kind}?",
                    f"On what date did {char} take part in a {kind}?",
                ])
            qs.append(self._q(QuestionType.FACTUAL_WHEN, 1, q, str(ev.time), [ev.id]))
        return qs

    def _gen_factual_where(self, events: List[MemoryEvent]) -> List[Question]:
        qs = []
        for ev in events:
            if not ev.locations:
                continue
            char = _pchar(ev)
            kind = _pkind(ev)
            date = str(ev.time)
            ans  = ", ".join(sorted(ev.locations))

            if char:
                q = self._pick([
                    f"Where did {char} attend a {kind} on {date}?",
                    f"At what location did {char} take part in a {kind} on {date}?",
                    f"In what place did {char} have a {kind} on {date}?",
                ])
            else:
                q = self._pick([
                    f"Where did the {kind} take place on {date}?",
                    f"At what location did the {kind} occur on {date}?",
                ])
            qs.append(self._q(QuestionType.FACTUAL_WHERE, 1, q, ans, [ev.id]))
        return qs

    def _gen_factual_what(self, events: List[MemoryEvent]) -> List[Question]:
        qs = []
        for ev in events:
            char = _pchar(ev)
            loc  = _ploc(ev)
            date = str(ev.time)
            ans  = ev.content[:100]

            if char and loc:
                q = self._pick([
                    f"What took place at {loc} on {date} involving {char}?",
                    f"What happened between the participants at {loc} on {date}?",
                    f"Describe what occurred at {loc} on {date} with {char} present.",
                ])
            elif char:
                q = self._pick([
                    f"What did {char} do on {date}?",
                    f"What occurred involving {char} on {date}?",
                ])
            elif loc:
                q = self._pick([
                    f"What happened at {loc} on {date}?",
                    f"What took place at {loc} on {date}?",
                ])
            else:
                continue
            qs.append(self._q(QuestionType.FACTUAL_WHAT, 2, q, ans, [ev.id]))
        return qs

    def _gen_factual_count(self, events: List[MemoryEvent]) -> List[Question]:
        qs = []
        char_counts: dict = {}
        for ev in events:
            for c in ev.characters:
                char_counts[c] = char_counts.get(c, 0) + 1

        for char, count in char_counts.items():
            if count < 2:
                continue
            evidence = [ev.id for ev in events if char in ev.characters]
            q = self._pick([
                f"How many events involved {char}?",
                f"In how many separate events did {char} appear?",
                f"How many times does {char} appear across all events?",
            ])
            qs.append(self._q(
                QuestionType.FACTUAL_COUNT, 2, q, f"{count} times", evidence
            ))
        return qs

    def _gen_sequential_before(self, events: List[MemoryEvent]) -> List[Question]:
        qs = []
        for i in range(1, len(events)):
            ev   = events[i]
            prev = events[i - 1]
            char = _pchar(ev)
            kind = _pkind(ev)
            loc  = _ploc(ev)
            date = str(ev.time)
            ans  = f"{prev.title} ({prev.time})"

            if char and loc:
                q = self._pick([
                    f"What happened just before {char}'s {kind} at {loc} on {date}?",
                    f"What occurred prior to the {kind} involving {char} at {loc} on {date}?",
                ])
            elif char:
                q = self._pick([
                    f"What happened just before {char}'s {kind} on {date}?",
                    f"What came immediately before {char}'s {kind} on {date}?",
                ])
            else:
                q = f"What happened just before the {kind} on {date}?"

            qs.append(self._q(
                QuestionType.SEQUENTIAL_BEFORE, 2, q, ans,
                [prev.id, ev.id], time_point=ev.time
            ))
        return qs

    def _gen_sequential_after(self, events: List[MemoryEvent]) -> List[Question]:
        qs = []
        for i in range(len(events) - 1):
            ev      = events[i]
            next_ev = events[i + 1]
            char    = _pchar(ev)
            kind    = _pkind(ev)
            loc     = _ploc(ev)
            date    = str(ev.time)
            ans     = f"{next_ev.title} ({next_ev.time})"

            if char and loc:
                q = self._pick([
                    f"What followed {char}'s {kind} at {loc} on {date}?",
                    f"What came after the {kind} involving {char} at {loc} on {date}?",
                ])
            elif char:
                q = self._pick([
                    f"What happened after {char}'s {kind} on {date}?",
                    f"What followed {char}'s {kind} on {date}?",
                ])
            else:
                q = f"What happened after the {kind} on {date}?"

            qs.append(self._q(
                QuestionType.SEQUENTIAL_AFTER, 2, q, ans,
                [ev.id, next_ev.id], time_point=next_ev.time
            ))
        return qs

    def _gen_temporal_duration(self, events: List[MemoryEvent]) -> List[Question]:
        qs = []
        for i in range(0, len(events) - 1, 2):
            e1, e2 = events[i], events[i + 1]
            dist   = self.tm.get_temporal_distance(e1.time, e2.time)
            k1, k2 = _pkind(e1), _pkind(e2)

            q = self._pick([
                f"How many days passed between the {k1} on {e1.time} and the {k2} on {e2.time}?",
                f"What was the time gap between the {k1} on {e1.time} and the {k2} on {e2.time}?",
                f"How much time elapsed from the {k1} on {e1.time} to the {k2} on {e2.time}?",
            ])
            qs.append(self._q(
                QuestionType.TEMPORAL_DURATION, 3, q,
                f"{dist['days']:.0f} days", [e1.id, e2.id]
            ))
        return qs

    def _gen_sequential_order(self, events: List[MemoryEvent]) -> List[Question]:
        qs = []
        for i in range(len(events) - 2):
            e1, e2  = events[i], events[i + 1]
            k1, k2  = _pkind(e1), _pkind(e2)
            c1, c2  = _pchar(e1), _pchar(e2)
            l1, l2  = _ploc(e1), _ploc(e2)
            ans = f"1. {e1.title} ({e1.time}), then 2. {e2.title} ({e2.time})"

            # Describe each event without its title
            def _desc(e, c, l, k):
                if c and l:
                    return f"the {k} involving {c} at {l}"
                elif c:
                    return f"the {k} involving {c}"
                elif l:
                    return f"the {k} at {l}"
                else:
                    return f"the {k} on {e.time}"

            d1, d2 = _desc(e1, c1, l1, k1), _desc(e2, c2, l2, k2)
            q = self._pick([
                f"Which came first: {d1} or {d2}?",
                f"Put in chronological order: {d1} and {d2}.",
            ])
            qs.append(self._q(
                QuestionType.SEQUENTIAL_ORDER, 3, q, ans, [e1.id, e2.id]
            ))
        return qs

    def _gen_character_appearance(self, events: List[MemoryEvent]) -> List[Question]:
        qs = []
        char_events: dict = {}
        for ev in events:
            for c in ev.characters:
                char_events.setdefault(c, []).append(ev)

        for char, evs in char_events.items():
            if len(evs) < 2:
                continue
            sorted_evs = sorted(evs, key=lambda e: (e.time.year, e.time.month, e.time.day))
            first, last = sorted_evs[0], sorted_evs[-1]

            qs.append(self._q(
                QuestionType.CHARACTER_APPEARANCE, 2,
                f"When was {char} first mentioned?",
                f"{first.title} ({first.time})",
                [first.id],
            ))
            qs.append(self._q(
                QuestionType.CHARACTER_APPEARANCE, 2,
                f"When did {char} last appear?",
                f"{last.title} ({last.time})",
                [last.id],
            ))
            all_titles = "; ".join(f"{e.title} ({e.time})" for e in sorted_evs)
            qs.append(self._q(
                QuestionType.CHARACTER_APPEARANCE, 3,
                f"In which events was {char} involved?",
                all_titles,
                [e.id for e in sorted_evs],
            ))
        return qs

    def _gen_location_tracking(self, events: List[MemoryEvent]) -> List[Question]:
        qs = []
        loc_events: dict = {}
        for ev in events:
            for loc in ev.locations:
                loc_events.setdefault(loc, []).append(ev)

        for loc, evs in loc_events.items():
            if len(evs) < 2:
                continue
            sample_char = sorted({c for e in evs for c in e.characters})[0] \
                          if any(e.characters for e in evs) else None
            titles = ", ".join(e.title for e in evs[:3])

            q = self._pick([
                q for q in [
                    f"What events happened at {loc}?",
                    f"Which events took place at {loc}?",
                    f"Where did events involving {sample_char} occur?" if sample_char else None,
                ] if q is not None
            ])
            qs.append(self._q(
                QuestionType.LOCATION_TRACKING, 2, q, titles,
                [e.id for e in evs[:3]],
            ))
        return qs

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_by_type(self, qtype: QuestionType) -> List[Question]:
        return [q for q in self._questions if q.type == qtype]

    def get_by_difficulty(self, difficulty: int) -> List[Question]:
        return [q for q in self._questions if q.difficulty == difficulty]

    def export(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump([q.to_dict() for q in self._questions], f, indent=2, default=str)


# Backward-compat alias
QuestionGenerator = TemplateQuestionBank
