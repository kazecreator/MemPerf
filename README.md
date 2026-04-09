# MemPerf

A benchmark framework for AI memory systems — **no LLM required**.

Scoring is purely retrieval-based: given a question, does your memory system return the right events? This makes results reproducible, fast, and independent of any specific model.

## Why

Most memory benchmarks depend on LLM judges, making results noisy, expensive, and hard to reproduce. MemPerf evaluates memory systems directly on what they actually do: **retrieve the right events for a given question**.

The questions cover capabilities that matter for AI agents operating over long timelines:

- Who was present at an event?
- What happened before/after a given moment?
- How long between two events?
- Where did a character move over time?
- What caused a later event?

## Quick Start

```bash
pip install -r requirements.txt

python benchmark.py --system simple
python benchmark.py --system inverted
python benchmark.py --system oracle     # perfect-retrieval upper bound
```

## Benchmarking Your Own System

Works with any language, any framework — Python libraries, MCP servers, CLI tools, REST APIs.
MemPerf spawns your bridge script and communicates via JSON lines over stdio.

```bash
python benchmark.py --system "python bridge.py"
python benchmark.py --system "node bridge.js"
```

**Quickest way to get a bridge:** fill in [`examples/BRIDGE_PROMPT.md`](examples/BRIDGE_PROMPT.md) and feed it to any LLM. You'll have a working script in under a minute.

Ready-made examples in `examples/`:
- `bridge_template.py` — annotated template, fill in two functions
- `bridge_mem0.py` — mem0
- `bridge_chromadb.py` — ChromaDB

**Bridge protocol:**
```
ingest   → {"op":"ingest",   "events":[{"id","title","content","time","characters","locations"}]}
         ← {"ok": true}

retrieve → {"op":"retrieve", "query":"...", "time_point":"YYYY-MM-DD"|null, "top_k":10}
         ← {"results": [{"event_id":"E001", "score":0.9}, ...]}
```

The only requirement: `event_id` in retrieve results must match the `id` sent during ingest.

## Metrics

| Metric | What it measures |
|--------|-----------------|
| **R@1** | Is the top result the right event? |
| **R@3** | Is the right event in the top 3? |
| **R@5** | Is the right event in the top 5? |
| **R@K** | Is the right event anywhere in the returned list? |
| **MRR** | Mean Reciprocal Rank — average rank quality |

`--system oracle` gives the upper bound: what R@K looks like when retrieval is perfect.

## Question Types

13 types across 4 capability dimensions:

| Capability | Question Types |
|------------|---------------|
| Temporal ordering | `factual_when`, `sequential_before/after/order`, `temporal_duration` |
| Associative recall | `factual_who`, `character_appearance`, `character_state` |
| Spatial recall | `factual_where`, `location_tracking` |
| Detail retention | `factual_what`, `factual_count`, `causal_reasoning` |

## Architecture

```
benchmark.py              ← CLI entry point
engine/
  time_machine.py         ← event store + temporal graph
  event_generator.py      ← synthetic event generation (TMGenerator)
  question_bank.py        ← deterministic question generation (TemplateQuestionBank)
  memory_system.py        ← BaseMemorySystem interface + SimpleKeyword + InvertedIndex
  scorer.py               ← RetrievalScorer (R@K, MRR)
  event_graph.py          ← CharacterGraph, TemporalGraph, SpatialContext
```

## CLI Options

```
python benchmark.py [OPTIONS]

--system      simple | inverted | oracle | ./path.py:ClassName  (default: simple)
--events      synthetic events per seed   (default: 50)
--questions   questions per seed          (default: 100)
--seeds       comma-separated seeds       (default: 42,123,777)
--out         JSON output path
```
