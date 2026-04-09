"""
MemPerf Bridge — ChromaDB
=========================
Connects MemPerf to a ChromaDB collection.

Install:
    pip install chromadb  # requires Python >= 3.11

Usage:
    python benchmark.py --system "python examples/bridge_chromadb.py"
"""

import sys
import json
import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("memperf_bench")


def handle_ingest(events):
    collection.add(
        ids=[ev["id"] for ev in events],
        documents=[f"{ev['title']}\n{ev['content']}" for ev in events],
        metadatas=[{"time": ev["time"],
                    "characters": ",".join(ev["characters"]),
                    "locations":  ",".join(ev["locations"])}
                   for ev in events],
    )


def handle_retrieve(query, time_point, top_k):
    n = min(top_k, collection.count())
    if n == 0:
        return []
    raw = collection.query(query_texts=[query], n_results=n,
                           include=["distances"])
    results = []
    for eid, dist in zip(raw["ids"][0], raw["distances"][0]):
        results.append({"event_id": eid, "score": max(0.0, 1.0 - dist)})
    return results


def handle_clear():
    global collection
    client.delete_collection("memperf_bench")
    collection = client.get_or_create_collection("memperf_bench")


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
