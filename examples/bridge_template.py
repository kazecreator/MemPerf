"""
MemPerf Bridge Template
=======================
Copy this file into your project and fill in the two sections marked TODO.

Usage:
    python benchmark.py --system "python bridge_template.py"

Protocol (handled automatically — do not change the I/O logic):
  MemPerf sends one JSON line per operation to stdin.
  Your bridge writes one JSON line response to stdout.

  ingest  → {"op":"ingest",  "events":[{"id","title","content","time","characters","locations"}]}
          ← {"ok": true}

  retrieve → {"op":"retrieve", "query":"...", "time_point":"YYYY-MM-DD"|null, "top_k":10}
           ← {"results": [{"event_id":"E001", "score":0.9}, ...]}

  clear   → {"op":"clear"}
          ← {"ok": true}
"""

import sys
import json

# ── TODO 1: import and initialise your memory system ─────────────────────────
# from my_memory import MyMemory
# mem = MyMemory()
# ─────────────────────────────────────────────────────────────────────────────


def handle_ingest(events: list) -> None:
    """Store events in your memory system.

    Each event dict has:
        id          — unique string, MUST be stored and returned in retrieve
        title       — short title of the event
        content     — full text description
        time        — "YYYY-MM-DD" string
        characters  — list of character names
        locations   — list of location names
    """
    for ev in events:
        # ── TODO 2a: store the event ──────────────────────────────────────────
        # The event_id MUST be preserved so retrieve() can return it.
        # Embed it in text, metadata, or however your system supports.
        #
        # Example (text embedding):
        #   text = f"[MEMPERF:{ev['id']}] {ev['title']}\n{ev['content']}"
        #   mem.add(text)
        #
        # Example (metadata):
        #   mem.add(ev["content"], metadata={"memperf_id": ev["id"]})
        # ─────────────────────────────────────────────────────────────────────
        raise NotImplementedError("fill in handle_ingest()")


def handle_retrieve(query: str, time_point, top_k: int) -> list:
    """Search your memory system and return a ranked list.

    Returns:
        List of dicts, each with at minimum:
            event_id  — the id you stored during ingest (required for scoring)
            score     — relevance score 0-1 (optional, use 1.0 if unavailable)
    """
    # ── TODO 2b: search and return results ───────────────────────────────────
    # raw = mem.search(query, limit=top_k)
    # return [{"event_id": r["metadata"]["memperf_id"], "score": r["score"]}
    #         for r in raw]
    # ─────────────────────────────────────────────────────────────────────────
    raise NotImplementedError("fill in handle_retrieve()")


def handle_clear() -> None:
    """Optional: clear all stored memories between seeds."""
    pass  # implement if your system supports it


# ── Main loop — do not modify ─────────────────────────────────────────────────
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
            results = handle_retrieve(req["query"], req.get("time_point"), req.get("top_k", 10))
            print(json.dumps({"results": results}), flush=True)

        elif op == "clear":
            handle_clear()
            print(json.dumps({"ok": True}), flush=True)

        else:
            print(json.dumps({"error": f"unknown op: {op}"}), flush=True)

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr, flush=True)
        print(json.dumps({"ok": False, "error": str(e)}), flush=True)
