# MemPerf

A fast integration test for AI memory systems — **no LLM required**, runs in under 5 minutes.

Point it at your memory system, get R@1/3/5/K and MRR scores broken down by question type. Useful for catching regressions during development and verifying that a new system actually retrieves the right events.

## What this is (and isn't)

**MemPerf is a developer diagnostic tool**, not a publication-quality benchmark.

It uses synthetically generated events and deterministic question templates. This makes it fast, reproducible, and dependency-free — but it is not a substitute for authoritative benchmarks (LoCoMo, LongMemEval, etc.) backed by real datasets and human annotation.

**Use MemPerf to:**
- Verify a new memory system retrieves relevant events
- Catch regressions when you change retrieval logic
- Compare retrieval approaches quickly during development
- Diagnose which question types your system struggles with

**Don't use MemPerf to:**
- Claim state-of-the-art performance in a paper
- Compare against systems tested on different benchmarks

## Why

Most memory benchmarks depend on LLM judges, making results noisy, expensive, and hard to reproduce. MemPerf evaluates memory systems directly on what they do: **retrieve the right events for a given question**.

Questions are generated from event attributes — not event titles — so keyword overlap alone cannot win. The benchmark exercises:

- Who was present at a given place and time?
- What happened before/after a specific event?
- How long between two events?
- Where was a character on a given date?

## Quick Start

```bash
pip install -r requirements.txt

python benchmark.py --system simple
python benchmark.py --system inverted
python benchmark.py --system oracle     # perfect-retrieval upper bound
```

## Benchmarking Your Own System

Works with any language, any framework — Python libraries, MCP servers, CLI tools, REST APIs.
MemPerf spawns your bridge script as a subprocess and communicates via JSON lines over stdio.

```bash
python benchmark.py --system "python bridge.py"
python benchmark.py --system "node bridge.js"
```

**Quickest way to get a bridge:** fill in [`examples/BRIDGE_PROMPT.md`](examples/BRIDGE_PROMPT.md) and feed it to any LLM. You'll have a working script in under a minute.

Ready-made examples in `examples/`:
- `bridge_template.py` — annotated template, fill in two functions
- `bridge_mem0.py` — mem0
- `bridge_chromadb.py` — ChromaDB

### Dependency isolation

If your bridge depends on packages that conflict with MemPerf's environment, run it in a separate virtual environment:

```bash
python -m venv .venv
.venv/bin/pip install your-memory-system

python benchmark.py --system ".venv/bin/python bridge.py"
```

The `--system` argument is passed directly to the shell, so any valid command works — including full paths to Python interpreters.

### Bridge protocol

```
ingest   → {"op":"ingest", "events":[{
               "id":           "E0001",
               "title":        "...",
               "content":      "...",
               "time":         "2024-03-15",
               "characters":   ["Alice", "Bob"],
               "locations":    ["Office A"],
               "conversation": [
                   {"role":"user",      "content":"ID: E0001\nDate: 2024-03-15\n..."},
                   {"role":"assistant", "content":"Stored. [ID:E0001] [DATE:2024-03-15] ..."}
               ]
           }]}
         ← {"ok": true}

retrieve → {"op":"retrieve", "query":"...", "time_point":"YYYY-MM-DD"|null, "top_k":10}
         ← {"results": [{"event_id":"E001", "score":0.9}, ...]}

clear    → {"op":"clear"}
         ← {"ok": true}
```

The only requirement: `event_id` in retrieve results must exactly match the `id` sent during ingest.

Each event includes two representations:
- **Structured fields** (`content`, `title`, `characters`, `locations`) — for vector stores, keyword search, graph databases
- **`conversation`** — a `[user, assistant]` pair for systems that ingest chat history; the user turn has explicit `ID:` / `Date:` / `Participants:` / `Location:` labels

## Metrics

| Metric | What it measures |
|--------|-----------------|
| **R@1** | Is the top result the right event? |
| **R@3** | Is the right event in the top 3? |
| **R@5** | Is the right event in the top 5? |
| **R@K** | Is the right event anywhere in the returned list? |
| **MRR** | Average reciprocal rank across all evidence events |

`R@1 ≤ R@3 ≤ R@5 ≤ R@K` always holds. R@K low → the system can't surface the event at all. R@K high but R@1 low → the system finds it but ranks it poorly.

`--system oracle` gives the upper bound: what scores look like when retrieval is perfect.

## Question Types

13 types across 4 capability dimensions:

| Capability | Question types |
|------------|---------------|
| Temporal ordering | `factual_when`, `sequential_before`, `sequential_after`, `sequential_order`, `temporal_duration` |
| Associative recall | `factual_who`, `character_appearance`, `character_state` |
| Spatial recall | `factual_where`, `location_tracking` |
| Detail retention | `factual_what`, `factual_count`, `causal_reasoning` |

Questions are generated from event attributes (character, location, date, kind) — never from event titles — so keyword matching alone does not win.

## Architecture

```
benchmark.py              ← CLI entry point + SubprocessAdapter
engine/
  time_machine.py         ← event store + temporal graph
  event_generator.py      ← synthetic event generation
  question_bank.py        ← deterministic, title-free question generation
  memory_system.py        ← SimpleKeyword + InvertedIndex built-in systems
  scorer.py               ← R@K, MRR
  event_graph.py          ← CharacterGraph, TemporalGraph, SpatialContext
examples/
  bridge_template.py      ← annotated starter template
  BRIDGE_PROMPT.md        ← prompt for generating a bridge with an LLM
```

## CLI Options

```
python benchmark.py [OPTIONS]

--system      simple | inverted | oracle | "python bridge.py"  (default: simple)
--events      synthetic events per seed   (default: 50)
--questions   questions per seed          (default: 100)
--seeds       comma-separated seeds       (default: 42,123,777)
--out         JSON output path
```
