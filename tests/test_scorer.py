"""
Unit tests for engine/scorer.py

Covers:
  - Single-evidence questions (classic R@K / MRR behaviour)
  - Multi-evidence questions (temporal_duration, sequential, etc.)
  - Edge cases: empty evidence, empty results, no overlap
  - Ordering invariant: R@1 <= R@3 <= R@5 <= R@K
  - MRR: correct reciprocal rank arithmetic
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from types import SimpleNamespace
from engine.scorer import RetrievalScorer, RecallResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_question(evidence_ids, qtype="factual_who", qid="Q0001"):
    """Minimal question stub — only fields scorer.score_recall() reads."""
    return SimpleNamespace(
        id=qid,
        type=SimpleNamespace(value=qtype),
        evidence_event_ids=evidence_ids,
    )


def make_results(event_ids):
    """List of retrieval result stubs."""
    return [SimpleNamespace(event_id=eid) for eid in event_ids]


scorer = RetrievalScorer()


# ── Single-evidence ────────────────────────────────────────────────────────────

class TestSingleEvidence:

    def test_perfect_top1(self):
        r = scorer.score_recall(
            make_question(["E1"]),
            make_results(["E1", "E2", "E3"]),
        )
        assert r.recall_at_1 == 1.0
        assert r.recall_at_3 == 1.0
        assert r.recall_at_k == 1.0
        assert r.mrr == pytest.approx(1.0)

    def test_found_at_rank2(self):
        r = scorer.score_recall(
            make_question(["E1"]),
            make_results(["E2", "E1", "E3"]),
        )
        assert r.recall_at_1 == 0.0
        assert r.recall_at_3 == 1.0
        assert r.recall_at_k == 1.0
        assert r.mrr == pytest.approx(0.5)

    def test_found_at_rank5(self):
        r = scorer.score_recall(
            make_question(["E1"]),
            make_results(["E2", "E3", "E4", "E5", "E1"]),
        )
        assert r.recall_at_1 == 0.0
        assert r.recall_at_3 == 0.0
        assert r.recall_at_5 == 1.0
        assert r.recall_at_k == 1.0
        assert r.mrr == pytest.approx(1 / 5)

    def test_not_found(self):
        r = scorer.score_recall(
            make_question(["E1"]),
            make_results(["E2", "E3", "E4"]),
        )
        assert r.recall_at_1 == 0.0
        assert r.recall_at_k == 0.0
        assert r.mrr == 0.0

    def test_ordering_invariant(self):
        r = scorer.score_recall(
            make_question(["E5"]),
            make_results(["E1", "E2", "E3", "E4", "E5", "E6"]),
        )
        assert r.recall_at_1 <= r.recall_at_3 <= r.recall_at_5 <= r.recall_at_k


# ── Multi-evidence ─────────────────────────────────────────────────────────────

class TestMultiEvidence:

    def test_both_found(self):
        r = scorer.score_recall(
            make_question(["E1", "E2"]),
            make_results(["E1", "E2", "E3"]),
        )
        assert r.recall_at_k == 1.0
        assert r.mrr == pytest.approx((1.0 + 0.5) / 2)

    def test_one_of_two_found(self):
        r = scorer.score_recall(
            make_question(["E1", "E2"]),
            make_results(["E1", "E3", "E4"]),
        )
        assert r.recall_at_k == pytest.approx(0.5)
        assert r.mrr == pytest.approx(0.5)   # (RR_E1=1.0 + RR_E2=0.0) / 2

    def test_neither_found(self):
        r = scorer.score_recall(
            make_question(["E1", "E2"]),
            make_results(["E3", "E4"]),
        )
        assert r.recall_at_k == 0.0
        assert r.mrr == 0.0

    def test_rk_uses_full_list_not_capped_at_evidence_size(self):
        # R@K should use all returned results, so evidence found at position 8
        # still counts — unlike old behaviour where k = min(|evidence|, 10) = 1
        r = scorer.score_recall(
            make_question(["E1"]),
            make_results(["E2", "E3", "E4", "E5", "E6", "E7", "E8", "E1"]),
        )
        assert r.recall_at_k == 1.0

    def test_rk_always_gte_r5(self):
        # Evidence at position 7 — inside R@K but outside R@5
        r = scorer.score_recall(
            make_question(["E7"]),
            make_results(["E1", "E2", "E3", "E4", "E5", "E6", "E7"]),
        )
        assert r.recall_at_5 == 0.0
        assert r.recall_at_k == 1.0
        assert r.recall_at_k >= r.recall_at_5

    def test_mrr_multi_evidence_average(self):
        # E1 at rank 1 (RR=1.0), E2 at rank 4 (RR=0.25) → MRR = (1.0+0.25)/2
        r = scorer.score_recall(
            make_question(["E1", "E2"]),
            make_results(["E1", "X", "X", "E2"]),
        )
        assert r.mrr == pytest.approx((1.0 + 0.25) / 2)

    def test_three_evidence_partial(self):
        # E1 at rank 1, E2 not found, E3 at rank 3 → MRR = (1.0 + 0 + 1/3) / 3
        r = scorer.score_recall(
            make_question(["E1", "E2", "E3"]),
            make_results(["E1", "X", "E3"]),
        )
        assert r.recall_at_k == pytest.approx(2 / 3)
        assert r.mrr == pytest.approx((1.0 + 0.0 + 1 / 3) / 3)


# ── Edge cases ─────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_evidence_returns_perfect_scores(self):
        r = scorer.score_recall(
            make_question([]),
            make_results(["E1", "E2"]),
        )
        assert r.recall_at_1 == 1.0
        assert r.recall_at_k == 1.0
        assert r.mrr == 1.0

    def test_empty_results(self):
        r = scorer.score_recall(
            make_question(["E1"]),
            make_results([]),
        )
        assert r.recall_at_k == 0.0
        assert r.mrr == 0.0

    def test_duplicate_retrieved_ids_do_not_inflate_score(self):
        # Returning the same event_id twice should not count twice
        r = scorer.score_recall(
            make_question(["E1", "E2"]),
            make_results(["E1", "E1", "E1"]),
        )
        assert r.recall_at_k == pytest.approx(0.5)

    def test_result_fields_populated(self):
        r = scorer.score_recall(
            make_question(["E1"], qid="Q42", qtype="temporal_duration"),
            make_results(["E1"]),
        )
        assert r.question_id == "Q42"
        assert r.question_type == "temporal_duration"
        assert "E1" in r.evidence_ids
        assert r.score == r.recall_at_k
