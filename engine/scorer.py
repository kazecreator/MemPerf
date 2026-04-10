"""
Retrieval Recall Scorer
=======================
Scores memory systems based purely on whether they return the right events.
No LLM required — ground truth is the evidence_event_ids on each Question.

Metrics:
  R@1   — is the top result the right event?
  R@3   — is the right event in the top 3?
  R@5   — is the right event in the top 5?
  R@K   — is the right event in the returned list? (primary metric)
  MRR   — mean reciprocal rank
"""

from dataclasses import dataclass
from typing import List


@dataclass
class RecallResult:
    question_id: str
    question_type: str
    evidence_ids: List[str]      # ground-truth event IDs
    retrieved_ids: List[str]     # event IDs returned by the memory system
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_k: float           # primary metric
    mrr: float
    score: float                 # alias for recall_at_k


class RetrievalScorer:
    """Score a memory system's retrieval quality without an LLM.

    Usage:
        scorer = RetrievalScorer()
        for q in questions:
            results = memory.retrieve(q.question, q.time_point, top_k=10)
            scorer.score_recall(q, results)
    """

    def score_recall(self, question, retrieved_results) -> RecallResult:
        evidence_ids = set(question.evidence_event_ids)
        retrieved_ids = [r.event_id for r in retrieved_results]

        def recall_at(n):
            if not evidence_ids:
                return 1.0
            return len(evidence_ids & set(retrieved_ids[:n])) / len(evidence_ids)

        # R@K: fraction of evidence events found anywhere in the returned list.
        # Using the full returned list (not capped at |evidence|) means R@K is
        # always >= R@5 and gives a consistent "did the system surface everything?"
        # signal regardless of how many evidence events a question has.
        rk = recall_at(len(retrieved_ids))

        # MRR: average reciprocal rank across all evidence events.
        # For single-evidence questions this equals the classic MRR.
        # For multi-evidence questions (temporal_duration, sequential, etc.)
        # it rewards finding all relevant events, not just the first one.
        if not evidence_ids:
            mrr = 1.0
        else:
            rr_sum = 0.0
            for eid in evidence_ids:
                for rank, rid in enumerate(retrieved_ids, 1):
                    if rid == eid:
                        rr_sum += 1.0 / rank
                        break
            mrr = rr_sum / len(evidence_ids)

        return RecallResult(
            question_id=question.id,
            question_type=question.type.value if hasattr(question.type, "value") else str(question.type),
            evidence_ids=list(evidence_ids),
            retrieved_ids=retrieved_ids,
            recall_at_1=recall_at(1),
            recall_at_3=recall_at(3),
            recall_at_5=recall_at(5),
            recall_at_k=rk,
            mrr=mrr,
            score=rk,
        )
