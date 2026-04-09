"""
MemPerf Engine
==============
Core components used by benchmark.py:

  TMGenerator, GeneratorConfig     — synthetic event generation
  TimeMachine, TimePoint,
  MemoryEvent                      — event store + temporal graph
  TemplateQuestionBank, Question,
  QuestionType                     — deterministic question generation
  BaseMemorySystem, RetrievalResult,
  SimpleMemorySystem,
  InvertedIndexSystem              — built-in memory systems
  RetrievalScorer, RecallResult    — recall scoring (no LLM)
"""

from engine.time_machine    import TimeMachine, TimePoint, MemoryEvent
from engine.event_generator import TMGenerator, GeneratorConfig
from engine.question_bank   import TemplateQuestionBank, Question, QuestionType
from engine.memory_system   import (
    BaseMemorySystem,
    RetrievalResult,
    SimpleMemorySystem,
    InvertedIndexSystem,
)
from engine.scorer import RetrievalScorer, RecallResult

__all__ = [
    "TimeMachine", "TimePoint", "MemoryEvent",
    "TMGenerator", "GeneratorConfig",
    "TemplateQuestionBank", "Question", "QuestionType",
    "BaseMemorySystem", "RetrievalResult", "SimpleMemorySystem", "InvertedIndexSystem",
    "RetrievalScorer", "RecallResult",
]
