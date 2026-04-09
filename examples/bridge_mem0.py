"""
MemPerf Bridge — mem0
=====================
Connects MemPerf to mem0 (https://github.com/mem0ai/mem0).

Install:
    pip install mem0ai

Usage:
    python benchmark.py --system "python examples/bridge_mem0.py"
"""

import sys
import json
from mem0 import Memory

mem = Memory()
USER_ID = "memperf_bench"


def handle_ingest(events):
    for ev in events:
        text = f"{ev['title']}\n{ev['content']}"
        mem.add(text, user_id=USER_ID, metadata={"memperf_id": ev["id"]})


def handle_retrieve(query, time_point, top_k):
    raw = mem.search(query, user_id=USER_ID, limit=top_k)
    results = []
    for r in raw.get("results", raw):  # mem0 v1 vs v2 shape
        mid = r.get("metadata", {}).get("memperf_id") or r.get("id", "")
        score = r.get("score", 1.0)
        if mid:
            results.append({"event_id": mid, "score": score})
    return results


def handle_clear():
    mem.delete_all(user_id=USER_ID)


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        op  = req["op"]
        if op == "ingest":
            handle_ingest(req["events"])
            print(json.dumps({"ok": True}), flush=True)
        elif op == "retrieve":
            results = handle_retrieve(req["query"], req.get("time_point"), req.get("top_k", 10))
            print(json.dumps({"results": results}), flush=True)
        elif op == "clear":
            handle_clear()
            print(json.dumps({"ok": True}), flush=True)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr, flush=True)
        print(json.dumps({"ok": False}), flush=True)
