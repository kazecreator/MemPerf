"""
Question Bank - Pre-defined question templates (no LLM dependency).

These templates generate factual questions deterministically — no LLM needed.
The same event set always produces the same question set.

LLM-powered question generation is in question_generator.py
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Set, Optional, Tuple
import random

from engine.time_machine import TimeMachine, TimePoint, MemoryEvent


class QuestionType(Enum):
    """Categories of questions for capability benchmarking"""
    # Factual
    FACTUAL_WHO = "factual_who"
    FACTUAL_WHEN = "factual_when"
    FACTUAL_WHERE = "factual_where"
    FACTUAL_WHAT = "factual_what"
    FACTUAL_COUNT = "factual_count"

    # Temporal / Sequential
    SEQUENTIAL_BEFORE = "sequential_before"
    SEQUENTIAL_AFTER = "sequential_after"
    SEQUENTIAL_ORDER = "sequential_order"
    TEMPORAL_DURATION = "temporal_duration"

    # Character & Location tracking
    CHARACTER_APPEARANCE = "character_appearance"
    CHARACTER_STATE = "character_state"
    LOCATION_TRACKING = "location_tracking"

    # Reasoning
    CAUSAL_REASONING = "causal_reasoning"


# ─────────────────────────────────────────────────────────────────────────────
# Question dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Question:
    """
    A benchmark question. Generated deterministically from events — no LLM needed.

    Fields:
      type: which capability dimension this tests
      difficulty: 1-5
      question: the question text
      answer: the correct answer
      evidence_event_ids: which events contain the answer (for recall scoring)
      time_point: optional — if set, query only at/before this time
    """
    id: str
    type: QuestionType
    difficulty: int
    question: str
    answer: str
    evidence_event_ids: List[str]     # events needed to answer correctly
    time_point: Optional[TimePoint]   # for temporal queries
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "difficulty": self.difficulty,
            "question": self.question,
            "answer": self.answer,
            "evidence": self.evidence_event_ids,
            "time_point": str(self.time_point) if self.time_point else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Question Templates
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_TEMPLATES = {
    QuestionType.FACTUAL_WHO: [
        "Who was involved in '{title}'?",
        "Who participated in '{title}'?",
        "Who took part in '{title}'?",
    ],
    QuestionType.FACTUAL_WHEN: [
        "When did '{title}' happen?",
        "What date was '{title}'?",
        "At what time did '{title}' occur?",
    ],
    QuestionType.FACTUAL_WHERE: [
        "Where did '{title}' take place?",
        "In what location did '{title}' occur?",
    ],
    QuestionType.FACTUAL_WHAT: [
        "What happened during '{title}'?",
        "What was '{title}' about?",
        "Can you describe '{title}'?",
    ],
    QuestionType.FACTUAL_COUNT: [
        "How many times did {character} appear?",
        "How many events involved {character}?",
        "How many events was {character} mentioned in?",
    ],
    QuestionType.SEQUENTIAL_BEFORE: [
        "What happened just before '{title}'?",
        "What led up to '{title}'?",
        "What occurred prior to '{title}'?",
    ],
    QuestionType.SEQUENTIAL_AFTER: [
        "What happened after '{title}'?",
        "What followed '{title}'?",
        "What occurred following '{title}'?",
    ],
    QuestionType.TEMPORAL_DURATION: [
        "How much time passed between '{title1}' and '{title2}'?",
        "What was the time gap between '{title1}' and '{title2}'?",
        "How many days were there between '{title1}' and '{title2}'?",
    ],
    QuestionType.SEQUENTIAL_ORDER: [
        "Which event came first: '{title1}' or '{title2}'?",
        "Put these in order: '{title1}' then '{title2}' — what happened in between?",
    ],
    QuestionType.CHARACTER_APPEARANCE: [
        "When was {character} first mentioned?",
        "When did {character} last appear?",
        "In which events was {character} involved?",
    ],
    QuestionType.CHARACTER_STATE: [
        "What role did {character} play overall?",
        "How many different events was {character} involved in?",
    ],
    QuestionType.LOCATION_TRACKING: [
        "What events happened at {location}?",
        "Which events took place in {location}?",
        "Where did events involving {character} occur?",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Template-based Question Generator (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

class TemplateQuestionBank:
    """
    Generate deterministic questions from event templates.
    No LLM — same events always produce the same questions.

    Usage:
      bank = TemplateQuestionBank(time_machine)
      questions = bank.generate(num_questions=30)
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
        """
        Generate a set of template questions.

        Questions are drawn from the event set and cover all QuestionTypes
        as evenly as possible.
        """
        random.seed(seed)
        self._questions = []

        all_events = sorted(self.tm.events.values(),
                           key=lambda e: (e.time.year, e.time.month, e.time.day))

        if len(all_events) < 2:
            return []

        # Generate questions for each type
        for qtype in QuestionType:
            qs = self._generate_type(qtype, all_events)
            self._questions.extend(qs)

        # Trim to requested number
        random.shuffle(self._questions)
        self._questions = self._questions[:num_questions]

        # Re-assign IDs
        for i, q in enumerate(self._questions):
            q.id = self._next_id()

        return self._questions

    def _generate_type(self, qtype: QuestionType,
                      events: List[MemoryEvent]) -> List[Question]:
        gen_methods = {
            QuestionType.FACTUAL_WHO: self._gen_factual_who,
            QuestionType.FACTUAL_WHEN: self._gen_factual_when,
            QuestionType.FACTUAL_WHERE: self._gen_factual_where,
            QuestionType.FACTUAL_WHAT: self._gen_factual_what,
            QuestionType.FACTUAL_COUNT: self._gen_factual_count,
            QuestionType.SEQUENTIAL_BEFORE: self._gen_sequential_before,
            QuestionType.SEQUENTIAL_AFTER: self._gen_sequential_after,
            QuestionType.TEMPORAL_DURATION: self._gen_temporal_duration,
            QuestionType.SEQUENTIAL_ORDER: self._gen_sequential_order,
            QuestionType.CHARACTER_APPEARANCE: self._gen_character_appearance,
            QuestionType.LOCATION_TRACKING: self._gen_location_tracking,
        }

        method = gen_methods.get(qtype)
        if method:
            return method(events)
        return []

    def _pick_template(self, qtype: QuestionType) -> str:
        templates = QUESTION_TEMPLATES.get(qtype, [])
        return random.choice(templates) if templates else "{question}"

    def _make_q(self, qtype: QuestionType, difficulty: int, question: str,
                answer: str, evidence: List[str],
                time_point: Optional[TimePoint] = None) -> Question:
        return Question(
            id="",
            type=qtype,
            difficulty=difficulty,
            question=question,
            answer=answer,
            evidence_event_ids=evidence,
            time_point=time_point,
            metadata={},
        )

    # ── Question generators per type ─────────────────────────────────────

    def _gen_factual_who(self, events: List[MemoryEvent]) -> List[Question]:
        questions = []
        for ev in events:
            if not ev.characters:
                continue
            chars = sorted(ev.characters)
            tpl = self._pick_template(QuestionType.FACTUAL_WHO)
            questions.append(self._make_q(
                QuestionType.FACTUAL_WHO, 1,
                tpl.format(title=ev.title),
                ", ".join(chars),
                [ev.id]
            ))
        return questions

    def _gen_factual_when(self, events: List[MemoryEvent]) -> List[Question]:
        questions = []
        for ev in events:
            tpl = self._pick_template(QuestionType.FACTUAL_WHEN)
            questions.append(self._make_q(
                QuestionType.FACTUAL_WHEN, 1,
                tpl.format(title=ev.title),
                str(ev.time),
                [ev.id]
            ))
        return questions

    def _gen_factual_where(self, events: List[MemoryEvent]) -> List[Question]:
        questions = []
        for ev in events:
            if not ev.locations:
                continue
            locs = sorted(ev.locations)
            tpl = self._pick_template(QuestionType.FACTUAL_WHERE)
            questions.append(self._make_q(
                QuestionType.FACTUAL_WHERE, 1,
                tpl.format(title=ev.title),
                ", ".join(locs),
                [ev.id]
            ))
        return questions

    def _gen_factual_what(self, events: List[MemoryEvent]) -> List[Question]:
        questions = []
        for ev in events:
            tpl = self._pick_template(QuestionType.FACTUAL_WHAT)
            questions.append(self._make_q(
                QuestionType.FACTUAL_WHAT, 2,
                tpl.format(title=ev.title),
                ev.content[:100],
                [ev.id]
            ))
        return questions

    def _gen_factual_count(self, events: List[MemoryEvent]) -> List[Question]:
        questions = []
        # Count appearances per character
        char_counts: dict = {}
        for ev in events:
            for c in ev.characters:
                char_counts[c] = char_counts.get(c, 0) + 1

        for char, count in char_counts.items():
            if count >= 2:
                tpl = self._pick_template(QuestionType.FACTUAL_COUNT)
                # Populate evidence with all events that feature this character
                # so retrieval recall scoring can measure actual performance.
                evidence_ids = [ev.id for ev in events if char in ev.characters]
                questions.append(self._make_q(
                    QuestionType.FACTUAL_COUNT, 2,
                    tpl.format(character=char),
                    f"{count} times",
                    evidence_ids
                ))
        return questions

    def _gen_sequential_before(self, events: List[MemoryEvent]) -> List[Question]:
        questions = []
        for i in range(1, len(events)):
            ev = events[i]
            prev = events[i-1]
            tpl = self._pick_template(QuestionType.SEQUENTIAL_BEFORE)
            questions.append(self._make_q(
                QuestionType.SEQUENTIAL_BEFORE, 2,
                tpl.format(title=ev.title),
                f"{prev.title} ({prev.time})",
                [prev.id, ev.id],
                time_point=ev.time
            ))
        return questions

    def _gen_sequential_after(self, events: List[MemoryEvent]) -> List[Question]:
        questions = []
        for i in range(len(events) - 1):
            ev = events[i]
            next_ev = events[i+1]
            tpl = self._pick_template(QuestionType.SEQUENTIAL_AFTER)
            questions.append(self._make_q(
                QuestionType.SEQUENTIAL_AFTER, 2,
                tpl.format(title=ev.title),
                f"{next_ev.title} ({next_ev.time})",
                [ev.id, next_ev.id],
                time_point=next_ev.time
            ))
        return questions

    def _gen_temporal_duration(self, events: List[MemoryEvent]) -> List[Question]:
        questions = []
        if len(events) < 2:
            return []
        for i in range(0, len(events) - 1, 2):  # sample pairs
            e1, e2 = events[i], events[i+1]
            dist = self.tm.get_temporal_distance(e1.time, e2.time)
            tpl = self._pick_template(QuestionType.TEMPORAL_DURATION)
            questions.append(self._make_q(
                QuestionType.TEMPORAL_DURATION, 3,
                tpl.format(title1=e1.title, title2=e2.title),
                f"{dist['days']} days",
                [e1.id, e2.id]
            ))
        return questions

    def _gen_sequential_order(self, events: List[MemoryEvent]) -> List[Question]:
        questions = []
        if len(events) < 3:
            return []
        for i in range(len(events) - 2):
            e1, e2 = events[i], events[i+1]
            tpl = self._pick_template(QuestionType.SEQUENTIAL_ORDER)
            answer = f"1. {e1.title} ({e1.time}), then 2. {e2.title} ({e2.time})"
            questions.append(self._make_q(
                QuestionType.SEQUENTIAL_ORDER, 3,
                tpl.format(title1=e1.title, title2=e2.title),
                answer,
                [e1.id, e2.id]
            ))
        return questions

    def _gen_character_appearance(self, events: List[MemoryEvent]) -> List[Question]:
        questions = []
        char_events: dict = {}
        for ev in events:
            for c in ev.characters:
                char_events.setdefault(c, []).append(ev)

        for char, evs in char_events.items():
            if len(evs) < 2:
                continue
            sorted_evs = sorted(evs, key=lambda e: (e.time.year, e.time.month, e.time.day))
            first, last = sorted_evs[0], sorted_evs[-1]

            # Use explicit templates so question and answer always agree.
            # "When was X first mentioned?" — answer = first event
            questions.append(self._make_q(
                QuestionType.CHARACTER_APPEARANCE, 2,
                f"When was {char} first mentioned?",
                f"{first.title} ({first.time})",
                [first.id]
            ))
            # "When did X last appear?" — answer = last event
            questions.append(self._make_q(
                QuestionType.CHARACTER_APPEARANCE, 2,
                f"When did {char} last appear?",
                f"{last.title} ({last.time})",
                [last.id]
            ))
            # "In which events was X involved?" — answer = all event titles
            all_titles = "; ".join(f"{e.title} ({e.time})" for e in sorted_evs)
            questions.append(self._make_q(
                QuestionType.CHARACTER_APPEARANCE, 3,
                f"In which events was {char} involved?",
                all_titles,
                [e.id for e in sorted_evs]
            ))
        return questions

    def _gen_location_tracking(self, events: List[MemoryEvent]) -> List[Question]:
        questions = []
        loc_events: dict = {}
        for ev in events:
            for loc in ev.locations:
                loc_events.setdefault(loc, []).append(ev)

        for loc, evs in loc_events.items():
            if len(evs) >= 2:
                tpl = self._pick_template(QuestionType.LOCATION_TRACKING)
                titles = ", ".join([e.title for e in evs[:3]])
                # Collect a sample character from these events for templates
                sample_chars = list(set(c for e in evs for c in e.characters))
                sample_char = sample_chars[0] if sample_chars else "someone"
                try:
                    question_text = tpl.format(location=loc, character=sample_char)
                except KeyError:
                    question_text = tpl.format(location=loc)
                questions.append(self._make_q(
                    QuestionType.LOCATION_TRACKING, 2,
                    question_text,
                    titles,
                    [e.id for e in evs[:3]]
                ))
        return questions

    # ── Accessors ─────────────────────────────────────────────────────────

    def get_by_type(self, qtype: QuestionType) -> List[Question]:
        return [q for q in self._questions if q.type == qtype]

    def get_by_difficulty(self, difficulty: int) -> List[Question]:
        return [q for q in self._questions if q.difficulty == difficulty]

    def export(self, path: str):
        """Export questions to JSON"""
        import json
        with open(path, "w") as f:
            json.dump([q.to_dict() for q in self._questions], f, indent=2, default=str)


# ─── Backward compatibility alias ───────────────────────────────────────────
# Existing benchmark scripts use QuestionGenerator as the class name.
# TemplateQuestionBank is the new canonical name.
QuestionGenerator = TemplateQuestionBank
