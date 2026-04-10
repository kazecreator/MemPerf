"""
MemPerf Bridge Template
=======================
Copy this file and fill in the two sections marked TODO.

Usage:
    python benchmark.py --system "python bridge_template.py"

Protocol (do not change the I/O logic):
  MemPerf sends one JSON line per operation to stdin.
  Your bridge writes one JSON line response to stdout.

  ingest  → {"op":"ingest", "events":[{
                "id":           "E0001",
                "title":        "...",
                "content":      "...",
                "time":         "2024-03-15",
                "characters":   ["Alice", "Bob"],
                "locations":    ["Office A"],
                "conversation": [
                    {"role": "user",      "content": "Please remember this event: ..."},
                    {"role": "assistant", "content": "Got it. I've recorded event E0001: ..."}
                ]
            }]}
          ← {"ok": true}

  retrieve → {"op":"retrieve", "query":"...", "time_point":"2024-03-15"|null, "top_k":10}
           ← {"results": [{"event_id":"E0001", "score":0.9}, ...]}

  clear   → {"op":"clear"}
          ← {"ok": true}

Which fields to use?
  "content" / "title" / "characters" / "locations"
      → for direct-storage systems (vector stores, keyword search, graph DBs)

  "conversation"
      → for systems that ingest chat history; the event_id is already
        embedded in the assistant turn so extractors are likely to keep it.
        Maintain a local mapping to recover event_id from search results.

Embedding model matters:
  Retrieval quality depends heavily on the embedding model.
  Use a model trained for retrieval (not a generation model repurposed
  for embeddings).  Better embeddings → higher R@K scores.
"""

import sys
import json
import re

# ── TODO 1: import and initialise your memory system ─────────────────────────
# from my_memory import MyMemory
# mem = MyMemory()
# ─────────────────────────────────────────────────────────────────────────────

# Approach C only: local id map (see below)
# SESSION_ID = "memperf"
# _id_store: list = []   # insertion-order list of event_ids


def handle_ingest(events: list) -> None:
    """Store events in your memory system.

    Each event dict has:
        id           — unique string, MUST be returned in retrieve results
        title        — short title
        content      — full text description
        time         — "YYYY-MM-DD"
        characters   — list of names
        locations    — list of place names
        conversation — [{"role":"user","content":"..."},
                        {"role":"assistant","content":"..."}]
                       user turn has explicit ID/Date/Who/Loc labels;
                       assistant turn has [ID:Exxxx] [DATE:...] tags.

    ── Approach A — direct storage (vector / keyword stores) ─────────────────
        text = f"{ev['title']}\\n{ev['content']}"
        mem.add(text, metadata={"memperf_id": ev["id"]})

    ── Approach B — chat-history ingestion ───────────────────────────────────
        mem.add(ev["conversation"], metadata={"session_id": SESSION_ID})
        # The assistant turn embeds [ID:Exxxx] so extractors are likely to
        # keep it.  Maintain a local list to recover ids if needed.

    ── Approach C — LLM-extraction systems (Mem0 / TeleMem style) ────────────
        The LLM rewrites content during ingestion and may not preserve the
        event_id verbatim.  Keep a local id list; use the system only for
        ranking, not for id recovery.

        _id_store.append(ev["id"])
        mem.add(ev["conversation"], metadata={"session_id": SESSION_ID})
    """
    for ev in events:
        # ── TODO 2a ──────────────────────────────────────────────────────────
        raise NotImplementedError("fill in handle_ingest()")


def handle_retrieve(query: str, time_point, top_k: int) -> list:
    """Search and return a ranked list.

    Returns a list of dicts, each with at minimum:
        event_id  — the id stored during ingest (required for scoring)
        score     — relevance score 0–1 (use 1.0 if unavailable)

    ── Approach A — metadata carries the id ─────────────────────────────────
        raw = mem.search(query, limit=top_k)
        return [{"event_id": r["metadata"]["memperf_id"], "score": r["score"]}
                for r in raw]

    ── Approach B — parse id from returned text ──────────────────────────────
        text = mem.search(query)
        ids  = re.findall(r'\\[ID:(E\\d+)\\]', text)   # matches [ID:E0042]
        return [{"event_id": eid, "score": 1/(i+1)} for i, eid in enumerate(ids)]

    ── Approach C — LLM-extraction: rank by system, recover id locally ───────
        The memory system returns scored text chunks.  Map chunk text back to
        the event_id using string overlap with stored content, or simply
        return the insertion-order list ranked by the system's scores.

        results = mem.search(query, session_id=SESSION_ID, limit=top_k)
        seen, ranked = set(), []
        for r in results:
            # Try to extract [ID:Exxxx] from the chunk text first
            ids_in_text = re.findall(r'\\[ID:(E\\d+)\\]', r.get("text", ""))
            eid = ids_in_text[0] if ids_in_text else None
            # Fallback: match against local store by position/score
            if eid and eid not in seen:
                seen.add(eid)
                ranked.append({"event_id": eid, "score": r.get("score", 1.0)})
        return ranked
    """
    # ── TODO 2b ──────────────────────────────────────────────────────────────
    raise NotImplementedError("fill in handle_retrieve()")


def handle_clear() -> None:
    """Reset state between seeds.  Leave empty if not needed — MemPerf
    spawns a fresh process per seed so state is naturally isolated."""
    pass


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
