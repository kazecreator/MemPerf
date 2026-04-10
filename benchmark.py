#!/usr/bin/env python3
"""
MemPerf — memory system benchmark CLI
======================================

Evaluates any memory system on episodic recall tasks.
No LLM required — scoring is purely retrieval-based (R@1/3/5/K, MRR).

Usage
-----
python benchmark.py --system simple
python benchmark.py --system inverted
python benchmark.py --system oracle              # perfect-retrieval upper bound
python benchmark.py --system "python bridge.py"  # any external system

Options
-------
--events      synthetic events per seed  (default: 50)
--questions   questions per seed         (default: 100)
--seeds       comma-separated seeds      (default: 42,123,777)
--out         save JSON results to file
"""

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import textwrap
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.time_machine    import TimeMachine, MemoryEvent
from engine.event_generator import TMGenerator, GeneratorConfig
from engine.question_bank   import TemplateQuestionBank
from engine.memory_system   import RetrievalResult, SimpleMemorySystem, InvertedIndexSystem
from engine.scorer          import RetrievalScorer, RecallResult


# ─────────────────────────────────────────────────────────────────────────────
# Built-in systems
# ─────────────────────────────────────────────────────────────────────────────

class _SimpleWrapper:
    name = "SimpleKeyword"
    def __init__(self):       self._mem = SimpleMemorySystem()
    def ingest(self, events): self._mem.ingest(events)
    def retrieve(self, query, time_point=None, top_k=10): return self._mem.retrieve(query, time_point=time_point, top_k=top_k)
    def clear(self):          self._mem = SimpleMemorySystem()


class _InvertedWrapper:
    name = "InvertedIndex"
    def __init__(self):       self._mem = InvertedIndexSystem()
    def ingest(self, events): self._mem.ingest(events)
    def retrieve(self, query, time_point=None, top_k=10): return self._mem.retrieve(query, time_point=time_point, top_k=top_k)
    def clear(self):          self._mem = InvertedIndexSystem()


class _OracleWrapper:
    """Perfect retrieval — always returns the exact evidence events.
    Use this to establish the R@K upper bound for your event set."""
    name = "Oracle"

    def __init__(self):
        self._events = {}
        self._questions_map = {}

    def ingest(self, events):
        self._events = {e.id: e for e in events}

    def set_questions(self, questions):
        self._questions_map = {q.id: q.evidence_event_ids for q in questions}

    def retrieve_for_question(self, question, top_k=10) -> List[RetrievalResult]:
        evidence_ids = self._questions_map.get(question.id, [])
        results, seen = [], set()
        for eid in evidence_ids:
            ev = self._events.get(eid)
            if ev:
                results.append(self._to_result(ev, 1.0))
                seen.add(eid)
        for ev in sorted(self._events.values(), key=lambda e: (e.time.year, e.time.month, e.time.day)):
            if len(results) >= top_k:
                break
            if ev.id not in seen:
                results.append(self._to_result(ev, 0.5))
        return results[:top_k]

    def _to_result(self, ev, score) -> RetrievalResult:
        return RetrievalResult(
            event_id=ev.id, title=ev.title, time=str(ev.time), content=ev.content,
            characters=sorted(ev.characters), locations=sorted(ev.locations),
            relevance_score=score, retrieval_method="oracle",
        )

    def clear(self):
        self._events = {}
        self._questions_map = {}


BUILTIN_SYSTEMS = {
    "simple":   _SimpleWrapper,
    "inverted": _InvertedWrapper,
    "oracle":   _OracleWrapper,
}


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess bridge adapter
# ─────────────────────────────────────────────────────────────────────────────

class SubprocessAdapter:
    """Communicates with any external memory system via JSON-lines over stdio.

    The bridge protocol (two operations):

      ingest  → {"op": "ingest", "events": [{id, title, content, time,
                                              characters, locations,
                                              conversation}]}
              ← {"ok": true}

      `conversation` is a pre-formatted [user, assistant] message pair
      representing the event.  Use it when your system ingests chat
      history; ignore it for direct-storage systems.

      retrieve → {"op": "retrieve", "query": "...",
                  "time_point": "YYYY-MM-DD" | null, "top_k": 10}
               ← {"results": [{"event_id": "E001", "score": 0.9}, ...]}

      clear   → {"op": "clear"}
              ← {"ok": true}

    See examples/ for ready-made bridge scripts.
    """

    def __init__(self, command: str):
        self._command = command
        self._proc: subprocess.Popen = None

    def _start(self):
        self._proc = subprocess.Popen(
            shlex.split(self._command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,       # bridge stderr → visible in terminal
            text=True,
            bufsize=1,
        )

    def _call(self, payload: dict) -> dict:
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()
        response = self._proc.stdout.readline()
        if not response:
            raise RuntimeError(f"Bridge process exited unexpectedly (command: {self._command})")
        return json.loads(response)

    @staticmethod
    def _event_to_conversation(ev) -> list:
        """Represent a MemoryEvent as a user/assistant exchange.

        Uses explicit field labels so LLM-based extractors reliably
        preserve structured attributes (id, date, participants, location).
        The event_id appears in both turns as a plain label [ID:Exxxx].
        """
        who   = f"\nParticipants: {', '.join(sorted(ev.characters))}" if ev.characters else ""
        where = f"\nLocation: {', '.join(sorted(ev.locations))}"       if ev.locations  else ""
        return [
            {
                "role":    "user",
                "content": (
                    f"Please remember this event.\n"
                    f"ID: {ev.id}\n"
                    f"Date: {ev.time}\n"
                    f"Description: {ev.content}"
                    f"{who}{where}"
                ),
            },
            {
                "role":    "assistant",
                "content": (
                    f"Stored. [ID:{ev.id}] [DATE:{ev.time}]"
                    + (f" [WHO:{','.join(sorted(ev.characters))}]" if ev.characters else "")
                    + (f" [LOC:{','.join(sorted(ev.locations))}]"  if ev.locations  else "")
                ),
            },
        ]

    def ingest(self, events):
        self._start()
        self._call({
            "op": "ingest",
            "events": [
                {
                    "id":           ev.id,
                    "title":        ev.title,
                    "content":      ev.content,
                    "time":         str(ev.time),
                    "characters":   sorted(ev.characters),
                    "locations":    sorted(ev.locations),
                    "conversation": self._event_to_conversation(ev),
                }
                for ev in events
            ],
        })

    def retrieve(self, query: str, time_point=None, top_k: int = 10) -> List[RetrievalResult]:
        resp = self._call({
            "op":         "retrieve",
            "query":      query,
            "time_point": str(time_point) if time_point else None,
            "top_k":      top_k,
        })
        results = []
        for r in resp.get("results", []):
            eid = r.get("event_id", "")
            if eid:
                results.append(RetrievalResult(
                    event_id=eid,
                    title=r.get("title", ""),
                    time=r.get("time", ""),
                    content=r.get("content", ""),
                    characters=r.get("characters", []),
                    locations=r.get("locations", []),
                    relevance_score=float(r.get("score", 0.0)),
                    retrieval_method="subprocess",
                ))
        return results

    def clear(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._call({"op": "clear"})
            except Exception:
                pass
            self._proc.stdin.close()
            self._proc.wait(timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
# Single-seed run
# ─────────────────────────────────────────────────────────────────────────────

def run_seed(system_factory, seed: int, num_events: int, num_questions: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  seed={seed}  events={num_events}  questions={num_questions}")
    print(f"{'='*60}")

    events = TMGenerator().generate(GeneratorConfig(num_events=num_events, seed=seed))
    print(f"  Generated {len(events)} events")

    tm = TimeMachine()
    tm.ingest_events(events)

    questions = TemplateQuestionBank(tm).generate(num_questions=num_questions, seed=seed)
    print(f"  Generated {len(questions)} questions  ({len({q.type for q in questions})} types)")

    system = system_factory()
    is_oracle = isinstance(system, _OracleWrapper)
    if is_oracle:
        system.set_questions(questions)

    t0 = time.time()
    try:
        system.ingest(events)
    except Exception as e:
        raise RuntimeError(f"ingest() failed: {e}") from e
    print(f"  Ingest: {time.time()-t0:.1f}s")

    scorer = RetrievalScorer()
    results: List[RecallResult] = []
    t0 = time.time()
    for q in questions:
        ranked = system.retrieve_for_question(q) if is_oracle else \
                 system.retrieve(q.question, time_point=q.time_point, top_k=10)
        results.append(scorer.score_recall(q, ranked))
    elapsed = time.time() - t0

    def _mean(attr):
        return statistics.mean(getattr(r, attr) for r in results)

    r1, r3, r5, rk, mrr = (
        _mean("recall_at_1"), _mean("recall_at_3"), _mean("recall_at_5"),
        _mean("recall_at_k"), _mean("mrr"),
    )

    print(f"\n  Retrieval ({elapsed:.1f}s)")
    print(f"  R@1={r1:.3f}  R@3={r3:.3f}  R@5={r5:.3f}  R@K={rk:.3f}  MRR={mrr:.3f}")

    by_type = {}
    for q, r in zip(questions, results):
        by_type.setdefault(q.type.value, []).append(r.recall_at_k)
    print("  R@K by type:")
    for t, vals in sorted(by_type.items()):
        print(f"    {t:<30} {statistics.mean(vals):.3f}  (n={len(vals)})")

    system.clear()
    return {
        "seed": seed, "num_events": len(events), "num_questions": len(questions),
        "recall_at_1": r1, "recall_at_3": r3, "recall_at_5": r5,
        "recall_at_k": rk, "mrr": mrr,
        "by_type": {k: statistics.mean(v) for k, v in by_type.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate + report
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: list) -> dict:
    metrics = ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_k", "mrr"]
    agg = {}
    for m in metrics:
        vals = [r[m] for r in results]
        agg[m] = {"mean": statistics.mean(vals),
                  "std":  statistics.stdev(vals) if len(vals) > 1 else 0.0,
                  "vals": vals}
    all_types = set(t for r in results for t in r.get("by_type", {}))
    agg["by_type"] = {
        t: {"mean": statistics.mean(v := [r["by_type"][t] for r in results if t in r.get("by_type", {})]),
            "std":  statistics.stdev(v) if len(v) > 1 else 0.0}
        for t in sorted(all_types)
    }
    return agg


def print_report(system_name: str, agg: dict):
    print(f"\n{'='*60}")
    print(f"  {system_name.upper()} — AGGREGATE (mean ± std, {len(next(iter(agg.values()))['vals'])} seeds)")
    print(f"{'='*60}")
    for key, label in [("recall_at_1","R@1"), ("recall_at_3","R@3"),
                        ("recall_at_5","R@5"), ("recall_at_k","R@K"), ("mrr","MRR")]:
        d = agg[key]
        print(f"  {label:<5}: {d['mean']:.3f} ± {d['std']:.3f}  {[round(v,3) for v in d['vals']]}")
    print("\n  R@K by question type:")
    for t, d in sorted(agg["by_type"].items()):
        bar = "█" * int(d["mean"] * 20) + "░" * (20 - int(d["mean"] * 20))
        print(f"  {t:<30} {bar}  {d['mean']:.3f} ± {d['std']:.3f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MemPerf — LLM-free benchmark for AI memory systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Built-in systems:
              simple     — keyword + metadata matching (no external deps)
              inverted   — inverted index over characters/locations/keywords
              oracle     — perfect retrieval upper bound

            Custom system:
              ./my_mem.py:MyMemory  — implement ingest(events) and retrieve(query, ...)
        """),
    )
    p.add_argument("--system",    default="simple",
                   help="System to benchmark (default: simple)")
    p.add_argument("--events",    type=int, default=50,
                   help="Synthetic events per seed (default: 50)")
    p.add_argument("--questions", type=int, default=100,
                   help="Questions per seed (default: 100)")
    p.add_argument("--seeds",     default="42,123,777",
                   help="Comma-separated seeds (default: 42,123,777)")
    p.add_argument("--out",       default=None,
                   help="Save JSON results to this file")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    system_name = args.system

    if system_name in BUILTIN_SYSTEMS:
        system_factory = BUILTIN_SYSTEMS[system_name]
    else:
        # Subprocess bridge: "python bridge.py" / "node bridge.js" / any shell command
        def system_factory(_cmd=system_name): return SubprocessAdapter(_cmd)
        system_name = shlex.split(system_name)[-1].replace(".py", "").replace(".js", "")


    all_results = []
    for seed in seeds:
        try:
            all_results.append(run_seed(system_factory, seed, args.events, args.questions))
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] seed={seed} failed: {exc}")
            traceback.print_exc()

    if not all_results:
        print("All seeds failed.")
        sys.exit(1)

    agg = aggregate(all_results)
    print_report(system_name, agg)

    safe_name = system_name.replace("/", "_").replace(" ", "_")
    out_path = args.out or f"results_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump({"system": system_name, "seeds": all_results, "aggregate": agg}, f, indent=2, default=str)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
