# MemPerf Bridge Generator Prompt

Copy the prompt below, fill in the blanks, and feed it to any LLM.
You'll get a working bridge script in under a minute.

---

```
I want to benchmark a memory system using MemPerf.
MemPerf communicates with external memory systems via a subprocess bridge:
a script that reads JSON lines from stdin and writes JSON responses to stdout.

Here is the protocol MemPerf uses:

  ingest  → {"op":"ingest",  "events":[{"id":"E001","title":"...","content":"...","time":"1943-06-01","characters":["Anne"],"locations":["Annex"]}]}
          ← {"ok": true}

  retrieve → {"op":"retrieve", "query":"Who was at the meeting?", "time_point":"1943-06-01", "top_k":10}
           ← {"results": [{"event_id":"E001","score":0.9}, ...]}

  clear   → {"op":"clear"}
          ← {"ok": true}

IMPORTANT: the event_id in retrieve results must exactly match the id from ingest.
Store it in metadata, embed it in text, or however your system allows.

My memory system is:
[DESCRIBE YOUR SYSTEM HERE — e.g. "mem0 with local Ollama backend",
 "a ChromaDB collection", "a custom vector store with add(text, meta) and search(query) methods",
 "an MCP server with tools memory_add and memory_search running on stdio"]

Installation / import:
[HOW TO IMPORT OR CONNECT — e.g. "from mem0 import Memory; mem = Memory()",
 "import chromadb; client = chromadb.Client()",
 "the MCP server is started separately; I connect via subprocess"]

The add/store method signature is:
[PASTE YOUR SYSTEM'S ADD METHOD SIGNATURE AND A USAGE EXAMPLE]

The search/retrieve method signature is:
[PASTE YOUR SYSTEM'S SEARCH METHOD SIGNATURE AND A USAGE EXAMPLE]

Please write a complete Python bridge script (bridge.py) that:
1. Implements the MemPerf stdin/stdout JSON protocol above
2. Uses my memory system for actual storage and retrieval
3. Handles the clear op (delete all stored memories)
4. Handles errors gracefully (print to stderr, respond with {"ok": false})
5. Uses only the main loop pattern from the template — no asyncio, no threads

The script will be run as:
    python benchmark.py --system "python bridge.py"
```

---

## Tips

**If your system doesn't return scores**, return `"score": 1.0` for all results — MemPerf uses score only for ranking, and will still compute R@K correctly based on event_id presence.

**If your system has no clear/reset method**, leave `handle_clear()` empty — MemPerf spawns a fresh process per seed, so state is naturally isolated.

**If your system is an MCP server on stdio**, describe it as: "an MCP server that I start with `node dist/index.js`. It has tools `memory_store` (args: text, metadata) and `memory_query` (args: query, limit)." The LLM will write the bridge to spawn and talk to the MCP process.

**If your system is a REST API**, describe it as: "a REST API running at http://localhost:8080 with POST /add and GET /search?q=..." The bridge will use `urllib` (no extra deps).
