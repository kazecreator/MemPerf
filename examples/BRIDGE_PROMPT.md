# MemPerf Bridge Generator Prompt

Copy the prompt below, fill in the blanks, and feed it to any LLM.
You'll have a working bridge script in under a minute.

---

```
I want to benchmark a memory system using MemPerf.
MemPerf communicates with external memory systems via a subprocess bridge:
a script that reads JSON lines from stdin and writes JSON responses to stdout.

Protocol:

  ingest  → {"op":"ingest", "events":[{
                "id":           "E0001",
                "title":        "Engineer A meets with team",
                "content":      "Engineer A and Designer B gathered at Office A ...",
                "time":         "2024-03-15",
                "characters":   ["Engineer A", "Designer B"],
                "locations":    ["Office A"],
                "conversation": [
                    {"role": "user",      "content": "Please remember this event.\nID: E0001\nDate: 2024-03-15\nDescription: ...\nParticipants: Engineer A, Designer B\nLocation: Office A"},
                    {"role": "assistant", "content": "Stored. [ID:E0001] [DATE:2024-03-15] [WHO:Designer B,Engineer A] [LOC:Office A]"}
                ]
            }]}
          ← {"ok": true}

  retrieve → {"op":"retrieve", "query":"Who was at the meeting on 2024-03-15?",
              "time_point":"2024-03-15"|null, "top_k":10}
           ← {"results": [{"event_id":"E0001","score":0.9}, ...]}

  clear   → {"op":"clear"}
          ← {"ok": true}

IMPORTANT: event_id in retrieve results must exactly match the id from ingest.

Each event includes two representations:
  - Structured fields ("content", "title", "characters", "locations"):
    for vector stores, keyword search, graph databases, etc.
  - "conversation": a [user, assistant] pair for systems that ingest
    chat history. The user turn has explicit ID/Date/Participants/Location
    labels; the assistant turn has [ID:Exxxx] tags so LLM extractors
    are likely to preserve them.

My memory system is:
[DESCRIBE YOUR SYSTEM — e.g.
 "a vector store with add(text, metadata) and search(query) methods",
 "a graph database that ingests structured facts",
 "a chat-history memory system with add(messages) and search(query)"]

Installation / import:
[HOW TO IMPORT OR CONNECT]

The add/store method signature is:
[PASTE YOUR SYSTEM'S ADD METHOD + EXAMPLE]

The search/retrieve method signature is:
[PASTE YOUR SYSTEM'S SEARCH METHOD + EXAMPLE]

Please write a complete Python bridge script (bridge.py) that:
1. Implements the MemPerf stdin/stdout JSON protocol above
2. Uses my memory system for storage and retrieval
3. Handles the clear op (delete all stored memories between runs)
4. Handles errors gracefully (print to stderr, respond with {"ok": false})
5. Uses only the simple main-loop pattern — no asyncio, no threads

Run as:
    python benchmark.py --system "python bridge.py"
```

---

## Tips

**Embedding model quality matters.**
Retrieval scores reflect the quality of your embedding model as much as your system's architecture. Use a model trained specifically for retrieval tasks (e.g. `nomic-embed-text`, `bge-*`, `e5-*`). Repurposing a generation model for embeddings will produce lower scores regardless of system design.

---

**Direct-storage systems** (vector stores, keyword indexes) — store `ev["content"]` as text, keep `ev["id"]` in metadata:

```python
def handle_ingest(events):
    for ev in events:
        mem.add(ev["content"], metadata={"memperf_id": ev["id"]})

def handle_retrieve(query, time_point, top_k):
    raw = mem.search(query, limit=top_k)
    return [{"event_id": r["metadata"]["memperf_id"], "score": r["score"]}
            for r in raw]
```

---

**Chat-history systems** — use `ev["conversation"]`; structured labels in both turns help LLM extractors keep the event_id:

```python
def handle_ingest(events):
    for ev in events:
        mem.add(ev["conversation"], metadata={"session_id": SESSION_ID})

def handle_retrieve(query, time_point, top_k):
    text = mem.search(query, session_id=SESSION_ID)
    ids  = re.findall(r'\[ID:(E\d+)\]', text)
    return [{"event_id": eid, "score": 1/(i+1)} for i, eid in enumerate(ids)]
```

---

**LLM-extraction systems** (entity-memory or knowledge-graph pipelines that rewrite content on ingestion) — the LLM may lose the event_id; keep a local list and use the system only for ranking:

```python
import re
_id_store = []

def handle_ingest(events):
    for ev in events:
        _id_store.append(ev["id"])
        mem.add(ev["conversation"])

def handle_retrieve(query, time_point, top_k):
    results = mem.search(query, limit=top_k)
    seen, ranked = set(), []
    for r in results:
        ids = re.findall(r'\[ID:(E\d+)\]', r.get("text", ""))
        eid = ids[0] if ids else None
        if eid and eid not in seen:
            seen.add(eid)
            ranked.append({"event_id": eid, "score": r.get("score", 1.0)})
    return ranked
```

---

**No scores available?** Return `"score": 1.0` — MemPerf ranks by score but computes R@K purely on event_id presence.

**No clear/reset method?** Leave `handle_clear()` empty — MemPerf spawns a fresh process per seed so state is naturally isolated.

**REST API?** Describe it as "a REST API at http://localhost:8080 with POST /add and GET /search?q=…" — the bridge will use `urllib`.

**Dependency conflicts?** If your system requires packages that conflict with MemPerf's environment, run the bridge in its own virtualenv:

```bash
python -m venv .venv
.venv/bin/pip install your-memory-system

python benchmark.py --system ".venv/bin/python bridge.py"
```
