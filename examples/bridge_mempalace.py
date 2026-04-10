"""
MemPerf bridge for MemPalace (https://github.com/milla-jovovich/mempalace)

Uses MemPalace's ChromaDB-backed vector store:
  - Ingest: add_drawer() stores event content directly; event_id is kept
    in the source_file metadata field so retrieve can recover it exactly.
  - Retrieve: search_memories() semantic search, filtered by wing.
  - Clear: new wing per run; palace directory wiped between seeds.

Install:
    uv pip install mempalace        # or: pip install mempalace

Usage:
    python benchmark.py --system ".venv/bin/python examples/bridge_mempalace.py"
"""

import sys
import json
import shutil
import uuid

PALACE_DIR = "/tmp/mempalace_memperf"
WING       = None   # set on first ingest / reset on clear

from mempalace.palace   import get_collection
from mempalace.miner    import add_drawer
from mempalace.searcher import search_memories

run_id = str(uuid.uuid4())[:8]
WING   = f"memperf_{run_id}"


def handle_ingest(events: list) -> None:
    collection = get_collection(PALACE_DIR)
    for ev in events:
        text = (
            f"[ID:{ev['id']}] Date: {ev['time']}\n"
            f"{ev['content']}"
            + (f"\nParticipants: {', '.join(ev['characters'])}" if ev["characters"] else "")
            + (f"\nLocation: {', '.join(ev['locations'])}"      if ev["locations"]  else "")
        )
        add_drawer(
            collection=collection,
            wing=WING,
            room="events",
            content=text,
            source_file=ev["id"],   # recovered as event_id in retrieve
            chunk_index=0,
            agent="memperf",
        )


def handle_retrieve(query: str, time_point, top_k: int) -> list:
    result = search_memories(
        query=query,
        palace_path=PALACE_DIR,
        wing=WING,
        n_results=top_k,
    )
    if "error" in result:
        return []
    seen, results = set(), []
    for r in result.get("results", []):
        eid = r.get("source_file", "")
        if eid and eid not in seen:
            seen.add(eid)
            results.append({"event_id": eid, "score": float(r.get("similarity", 0.0))})
    return results


def handle_clear() -> None:
    global run_id, WING
    run_id = str(uuid.uuid4())[:8]
    WING   = f"memperf_{run_id}"
    shutil.rmtree(PALACE_DIR, ignore_errors=True)


# ── Main loop ──────────────────────────────────────────────────────────────────
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        op  = req.get("op")

        if op == "ingest":
            handle_ingest(req["events"])
            print(json.dumps({"ok": True}), flush=True)

        elif op == "retrieve":
            results = handle_retrieve(
                req["query"], req.get("time_point"), req.get("top_k", 10)
            )
            print(json.dumps({"results": results}), flush=True)

        elif op == "clear":
            handle_clear()
            print(json.dumps({"ok": True}), flush=True)

        else:
            print(json.dumps({"error": f"unknown op: {op}"}), flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": str(e)}), flush=True)
