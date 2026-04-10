# Contributing to MemPerf

## Ways to contribute

**Add a bridge example** (`examples/bridge_<system>.py`)
The most useful contributions. If you've benchmarked a memory system, share your bridge script so others don't have to write it from scratch.

**Add a question type** (`engine/question_bank.py`)
New question types should cover a capability not already represented. Each type needs:
- A `_gen_<type>()` method in `TemplateQuestionBank`
- Populated `evidence_event_ids` (the scorer depends on these)
- A corresponding entry in `QuestionType` enum

**Fix a scoring bug** (`engine/scorer.py`, `engine/question_bank.py`)
If a question type is generating wrong `evidence_event_ids`, or `RecallResult` metrics are miscalculated, open an issue with a minimal reproducer.

## Pull request checklist

- [ ] `python benchmark.py --system simple --events 20 --questions 30 --seeds 42` runs without errors
- [ ] New bridge examples include the install command at the top
- [ ] No new dependencies added to MemPerf itself (`requirements.txt` stays minimal)
- [ ] Commit message explains *why*, not just *what*

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

The test suite covers scorer arithmetic (R@K, MRR, multi-evidence partial credit) and runs in under a second with no external dependencies.

## Running the benchmark locally

```bash
pip install -r requirements.txt   # requires Python >= 3.11
python benchmark.py --system simple --events 20 --questions 30 --seeds 42
python benchmark.py --system oracle --events 20 --questions 30 --seeds 42
```

## Questions

Open an issue — no question is too small.
