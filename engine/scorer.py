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

        k = min(len(evidence_ids), 10) if evidence_ids else 5

        def recall_at(n):
            if not evidence_ids:
                return 1.0
            return len(evidence_ids & set(retrieved_ids[:n])) / len(evidence_ids)

        mrr = 0.0
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in evidence_ids:
                mrr = 1.0 / rank
                break

        rk = recall_at(k)
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
