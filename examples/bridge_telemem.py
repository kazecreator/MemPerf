"""
MemPerf bridge for TeleMem (https://github.com/TeleAI-UAGI/TeleMem)

Uses TeleMem's full pipeline: conversation ingestion → LLM fact extraction
→ FAISS storage → semantic search.

Ingestion uses TeleMem's add() method (LLM extracts and stores facts).
Retrieval queries the events FAISS index directly because TeleMem's search()
only queries person_1/person_2 and skips the events index.
The LLM preserves the event_id in extracted summaries (e.g. "事件编号E0001"),
so event_ids are recovered by regex from matched summaries.

Install:
    pip install faiss-cpu mem0ai
    cd /path/to/TeleMem && pip install -e .

Requires:
    ollama running with qwen2.5:7b (or any OpenAI-compatible model)

Usage:
    python benchmark.py --system "python examples/bridge_telemem.py"
"""

import sys
import json
import re
import uuid
import shutil
import numpy as np

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY  = "ollama"
LLM_MODEL       = "qwen2.5:7b"
EMBED_MODEL     = "nomic-embed-text"
FAISS_DIR       = "/tmp/telemem_memperf"

from telemem.main import TeleMemory
from telemem.configs import TeleMemoryConfig
from mem0.configs.base import LlmConfig, EmbedderConfig, VectorStoreConfig

def _make_mem():
    config = TeleMemoryConfig(
        llm=LlmConfig(provider="openai", config={
            "openai_base_url": OLLAMA_BASE_URL,
            "api_key":         OLLAMA_API_KEY,
            "model":           LLM_MODEL,
        }),
        embedder=EmbedderConfig(provider="openai", config={
            "openai_base_url": OLLAMA_BASE_URL,
            "api_key":         OLLAMA_API_KEY,
            "model":           EMBED_MODEL,
        }),
        vector_store=VectorStoreConfig(
            provider="faiss",
            config={"path": FAISS_DIR},
        ),
    )
    return TeleMemory(config=config)

mem     = _make_mem()
run_id  = str(uuid.uuid4())[:8]


def handle_ingest(events: list) -> None:
    import contextlib
    for ev in events:
        # TeleMem prints progress to stdout; redirect to stderr to keep
        # the JSON-lines protocol on stdout clean.
        with contextlib.redirect_stdout(sys.stderr):
            mem.add(
                ev["conversation"],
                metadata={"sample_id": run_id, "user": ["user", "assistant"]},
            )


def handle_retrieve(query: str, time_point, top_k: int) -> list:
    # Embed query
    resp = mem.emb_client.embeddings.create(model=EMBED_MODEL, input=query)
    q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(q_vec)
    if norm > 0:
        q_vec /= norm

    # Search events FAISS index directly (search() skips this index)
    mem._load_or_create_index(run_id, "events")
    index    = mem.faiss_store.get(run_id, {}).get("events")
    metadata = mem.metadata_store.get(run_id, {}).get("events", [])

    if index is None or index.ntotal == 0:
        return []

    k = min(top_k, index.ntotal)
    D, I = index.search(q_vec.reshape(1, -1), k)

    results = []
    seen = set()
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        # original_messages is stored verbatim; the assistant turn always
        # contains the event_id ("Got it. I've recorded event E0001: ...")
        orig = metadata[idx].get("original_messages", [])
        eid = None
        for msg in orig:
            if msg.get("role") == "assistant":
                ids = re.findall(r'\b(E\d+)\b', msg.get("content", ""))
                if ids:
                    eid = ids[0]
                    break
        if eid and eid not in seen:
            seen.add(eid)
            results.append({"event_id": eid, "score": float(score)})

    return results


def handle_clear() -> None:
    global mem, run_id
    run_id = str(uuid.uuid4())[:8]
    # wipe FAISS files so next seed starts clean
    try:
        shutil.rmtree(FAISS_DIR, ignore_errors=True)
    except Exception:
        pass
    mem = _make_mem()


# ── Main loop ─────────────────────────────────────────────────────────────────
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
