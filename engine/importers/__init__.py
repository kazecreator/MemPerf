"""
Import Pipeline - Auto-detect format and import events from various sources.

Supported formats:
  - memoir.json / events.json  (list of events with time/content/characters/locations)
  - CSV                           (columns: time, title, content, characters, locations)
  - Obsidian Markdown vault        (directory of .md files with YAML frontmatter)
  - Notion CSV export             (exported from Notion database)
  - Telegram JSON export          (chat history JSON from Telegram export)
  - JSONL                         (one JSON object per line)

Usage:
  from engine.importers import detect_format, import_events

  # Auto-detect and import
  events, format_used, errors = import_events("/path/to/data")

  # Explicit format
  events, _, errors = import_events("/path/to/data", format="obsidian_md")
"""

import json
import csv
import re
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from engine.time_machine import TimePoint, MemoryEvent


# ─────────────────────────────────────────────────────────────────────────────
# Result Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImportResult:
    """Result of an import operation"""
    events: List[MemoryEvent]
    format_detected: str
    errors: List[str]           # non-fatal errors (skipped rows, etc.)
    warnings: List[str]         # warnings about data quality
    stats: Dict[str, Any]       # import statistics

    @property
    def success(self) -> bool:
        return len(self.events) > 0 and len(self.errors) == 0

    @property
    def partial_success(self) -> bool:
        return len(self.events) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Format Detectors
# ─────────────────────────────────────────────────────────────────────────────

class FormatDetector:
    """Auto-detect the format of an input file/directory"""

    @staticmethod
    def detect(path: str) -> str:
        """Detect format and return format name"""
        p = Path(path)

        if p.is_dir():
            # Check for Obsidian vault pattern (directory with .md files)
            md_files = list(p.rglob("*.md"))
            if md_files:
                # Check if files have frontmatter
                sample = md_files[0].read_text(errors="ignore")[:500]
                if re.search(r'^---\s*\n.*?\n---', sample, re.MULTILINE):
                    return "obsidian_md"
                return "markdown_dir"

            # Check for JSON files
            json_files = list(p.rglob("*.json"))
            if json_files:
                return "json_dir"

            return "unknown_dir"

        elif p.is_file():
            suffix = p.suffix.lower()

            if suffix == ".json":
                return FormatDetector._detect_json(p)
            elif suffix == ".csv":
                return "csv"
            elif suffix == ".md":
                return FormatDetector._detect_markdown(p)
            elif suffix == ".jsonl":
                return "jsonl"
            elif suffix in (".txt", ".log"):
                # Might be a text-based chat export
                content = p.read_text(errors="ignore")[:200]
                if any(kw in content.lower() for kw in ["message", "from", "text", "date"]):
                    return "text_chat"
                return "unknown"

        return "unknown"

    @staticmethod
    def _detect_json(p: Path) -> str:
        """Detect JSON subtype"""
        try:
            content = json.loads(p.read_text())
            if isinstance(content, list):
                if len(content) > 0:
                    first = content[0]
                    if "time" in first or "date" in first:
                        return "memoir_json"
                    if "role" in first or "from" in first:
                        return "telegram_json"
                    if "text" in first and "messages" not in first:
                        return "telegram_json"
            elif isinstance(content, dict):
                if "events" in content:
                    return "events_json"
                if "messages" in content or "chats" in content:
                    return "telegram_json"
        except Exception:
            pass
        return "unknown_json"

    @staticmethod
    def _detect_markdown(p: Path) -> str:
        """Detect markdown subtype"""
        content = p.read_text(errors="ignore")[:500]
        if re.search(r'^---\s*\n.*?\n---', content, re.MULTILINE):
            return "obsidian_md"
        return "plain_md"


# ─────────────────────────────────────────────────────────────────────────────
# Base Importer
# ─────────────────────────────────────────────────────────────────────────────

class BaseImporter:
    """Base class for format-specific importers"""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self._line_count = 0

    def import_file(self, path: str) -> ImportResult:
        """Import from file path — subclasses override"""
        raise NotImplementedError

    def _make_event(self, id: str, time_str: str, title: str,
                    content: str = "", characters: List[str] = None,
                    locations: List[str] = None, tags: List[str] = None,
                    emotional_valence: float = 0.0) -> Optional[MemoryEvent]:
        """Create a MemoryEvent with validation"""
        try:
            tp = TimePoint.from_string(time_str)
        except Exception:
            self.errors.append(f"[{id}] Invalid time format: {time_str}")
            return None

        if not title and not content:
            self.warnings.append(f"[{id}] Empty title and content, skipping")
            return None

        return MemoryEvent(
            id=id,
            time=tp,
            title=title or f"Event {id}",
            content=content or "",
            characters=set(characters or []),
            locations=set(locations or []),
            tags=set(tags or []),
            emotional_valence=emotional_valence,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Memoir JSON Importer
# ─────────────────────────────────────────────────────────────────────────────

class MemoirJsonImporter(BaseImporter):
    """Import from memoir.json format"""

    def import_file(self, path: str) -> ImportResult:
        events = []
        self.errors = []
        self.warnings = []

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = data.get("events", [])

        for i, item in enumerate(data):
            eid = item.get("id", f"E{i+1:04d}")
            time_str = item.get("time") or item.get("date", "")
            title = item.get("title", "")
            content = item.get("content", "") or item.get("text", "")
            chars = self._parse_list_field(item.get("characters", []))
            locs = self._parse_list_field(item.get("locations", []))
            tags = self._parse_list_field(item.get("tags", []))
            valence = float(item.get("valence", item.get("emotional_valence", 0.0)))

            ev = self._make_event(eid, time_str, title, content, chars, locs, tags, valence)
            if ev:
                events.append(ev)

        return ImportResult(
            events=events,
            format_detected="memoir_json",
            errors=self.errors,
            warnings=self.warnings,
            stats={"total_items": len(data), "imported": len(events), "failed": len(self.errors)}
        )

    def _parse_list_field(self, val) -> List[str]:
        if isinstance(val, list):
            return [str(v) for v in val]
        if isinstance(val, str):
            return [v.strip() for v in val.split(",") if v.strip()]
        return []


# ─────────────────────────────────────────────────────────────────────────────
# CSV Importer
# ─────────────────────────────────────────────────────────────────────────────

class CsvImporter(BaseImporter):
    """
    Import from CSV. Supports both Notion-style and custom CSV exports.

    Expected columns (flexible matching):
      - time, date, timestamp → time
      - title, name, event → title
      - content, body, text, description → content
      - characters, people, participants, roles → characters
      - locations, places, where → locations
      - tags, labels, categories → tags
    """

    COLUMN_ALIASES = {
        "time": ["time", "date", "timestamp", "datetime", "when", "day"],
        "title": ["title", "name", "event", "event_name", "summary"],
        "content": ["content", "body", "text", "description", "details", "note", "notes"],
        "characters": ["characters", "people", "participants", "roles", "person", "persons", "who"],
        "locations": ["locations", "places", "where", "location", "venue", "city"],
        "tags": ["tags", "labels", "categories", "type", "kind", "tag"],
    }

    def import_file(self, path: str) -> ImportResult:
        events = []
        self.errors = []
        self.warnings = []

        with open(path, encoding="utf-8", newline="") as f:
            # Detect delimiter
            sample = f.read(4096)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except Exception:
                dialect = csv.excel

            f.seek(0)
            reader = csv.DictReader(f, dialect=dialect)

            if not reader.fieldnames:
                self.errors.append("CSV has no headers")
                return ImportResult(events=[], format_detected="csv",
                                   errors=self.errors, warnings=self.warnings,
                                   stats={"total_rows": 0})

            # Map columns to canonical names
            col_map = self._map_columns(reader.fieldnames)
            missing = [c for c in ["time", "title"] if c not in col_map]
            if missing:
                self.warnings.append(f"CSV missing recommended columns: {missing} — will use defaults")

            for i, row in enumerate(reader):
                i += 1  # 1-indexed for user messages
                time_str = self._get_row_val(row, col_map, "time", "")
                title = self._get_row_val(row, col_map, "title", f"Row {i}")
                content = self._get_row_val(row, col_map, "content", "")
                chars = self._parse_list(self._get_row_val(row, col_map, "characters", ""))
                locs = self._parse_list(self._get_row_val(row, col_map, "locations", ""))
                tags = self._parse_list(self._get_row_val(row, col_map, "tags", ""))

                if not time_str:
                    self.warnings.append(f"[Row {i}] No time field, skipping")
                    continue

                ev = self._make_event(f"R{i:04d}", time_str, title, content, chars, locs, tags)
                if ev:
                    events.append(ev)

        return ImportResult(
            events=events,
            format_detected="csv",
            errors=self.errors,
            warnings=self.warnings,
            stats={"total_rows": i, "imported": len(events), "failed": len(self.warnings)}
        )

    def _map_columns(self, fieldnames: List[str]) -> Dict[str, str]:
        """Map CSV headers to canonical field names"""
        result = {}
        field_lower = {f.lower().strip(): f for f in fieldnames}

        for canon, aliases in self.COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in field_lower:
                    result[canon] = field_lower[alias]
                    break

        return result

    def _get_row_val(self, row: dict, col_map: dict, field: str, default: str) -> str:
        csv_col = col_map.get(field)
        if csv_col and csv_col in row:
            val = row[csv_col]
            return val.strip() if val else default
        return default

    def _parse_list(self, s: str) -> List[str]:
        if not s:
            return []
        # Split by comma, semicolon, or pipe
        parts = re.split(r'[,;|]\s*', s)
        return [p.strip() for p in parts if p.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Obsidian Markdown Importer
# ─────────────────────────────────────────────────────────────────────────────

class ObsidianImporter(BaseImporter):
    """
    Import from an Obsidian vault (directory of .md files).

    Expected frontmatter:
      ---
      date: 2024-01-15
      tags: [meeting, work]
      created: 2024-01-15
      ---

    Body is treated as content. Titles are extracted from the first H1/H2
    heading or from the filename.
    """

    FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.MULTILINE | re.DOTALL)
    TITLE_RE = re.compile(r'^#{1,2}\s+(.+)$', re.MULTILINE)

    def import_file(self, path: str) -> ImportResult:
        events = []
        self.errors = []
        self.warnings = []

        vault_path = Path(path)
        md_files = list(vault_path.rglob("*.md")) if vault_path.is_dir() else [vault_path]

        if not md_files:
            self.errors.append(f"No .md files found at {path}")
            return ImportResult(events=[], format_detected="obsidian_md",
                               errors=self.errors, warnings=self.warnings, stats={})

        for md_file in sorted(md_files):
            try:
                ev = self._parse_file(md_file)
                if ev:
                    events.append(ev)
            except Exception as e:
                self.errors.append(f"[{md_file.name}] Failed to parse: {e}")

        return ImportResult(
            events=events,
            format_detected="obsidian_md",
            errors=self.errors,
            warnings=self.warnings,
            stats={"files_found": len(md_files), "imported": len(events), "failed": len(self.errors)}
        )

    def _parse_file(self, path: Path) -> Optional[MemoryEvent]:
        content = path.read_text(encoding="utf-8", errors="ignore")

        # Extract frontmatter
        fm_match = self.FRONTMATTER_RE.match(content)
        frontmatter = {}
        if fm_match:
            fm_text = fm_match.group(1)
            frontmatter = self._parse_frontmatter(fm_text)

        # Extract title
        title = frontmatter.get("title", "")
        if not title:
            title_match = self.TITLE_RE.search(content[:500])
            title = title_match.group(1).strip() if title_match else path.stem

        # Extract date
        date_str = frontmatter.get("date") or frontmatter.get("created") or ""

        # Extract characters (custom frontmatter field)
        chars = self._parse_frontmatter_list(frontmatter.get("characters", ""))

        # Extract locations
        locs = self._parse_frontmatter_list(frontmatter.get("locations", ""))

        # Extract tags
        tags = self._parse_frontmatter_list(frontmatter.get("tags", ""))

        # Body content (everything after frontmatter)
        body = content[fm_match.end():].strip() if fm_match else content.strip()

        if not date_str:
            self.warnings.append(f"[{path.name}] No date in frontmatter, using filename date")
            # Try to extract date from filename
            date_str = self._extract_date_from_filename(path.stem)

        ev = self._make_event(
            id=f"F_{path.stem[:20]}",
            time_str=date_str,
            title=title,
            content=body[:500],  # truncate long content
            characters=chars,
            locations=locs,
            tags=tags,
        )
        return ev

    def _parse_frontmatter(self, text: str) -> Dict[str, str]:
        """Parse YAML-like frontmatter"""
        result = {}
        for line in text.split('\n'):
            line = line.strip()
            if ':' in line:
                key, _, val = line.partition(':')
                result[key.strip().lower()] = val.strip()
        return result

    def _parse_frontmatter_list(self, val: str) -> List[str]:
        """Parse frontmatter list field like '[meeting, work]' or 'meeting, work'"""
        val = val.strip()
        if not val:
            return []
        # Remove brackets
        val = re.sub(r'^\[|\]$', '', val)
        # Split by comma
        parts = re.split(r',\s*', val)
        return [p.strip('"\'') for p in parts if p.strip()]

    def _extract_date_from_filename(self, name: str) -> str:
        """Try to extract YYYY-MM-DD from filename like '2024-01-15 Daily Note'"""
        match = re.search(r'(\d{4}-\d{2}-\d{2})', name)
        if match:
            return match.group(1)
        return "2024-01-01"


# ─────────────────────────────────────────────────────────────────────────────
# Telegram JSON Importer
# ─────────────────────────────────────────────────────────────────────────────

class TelegramImporter(BaseImporter):
    """
    Import from Telegram chat export JSON.

    Expected format:
      [{"date": "2024-01-15T10:30:00", "from": "Alice", "text": "Hello!"}, ...]

    The TMGenerator will extract events from conversational content.
    """

    def import_file(self, path: str) -> ImportResult:
        events = []
        self.errors = []
        self.warnings = []

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Some exports wrap messages in {"messages": [...]}
            if "messages" in data:
                data = data["messages"]
            elif "chats" in data:
                # Telegram export format: {"chats": {"chats": [...]}}
                if "chats" in data.get("chats", {}):
                    data = data["chats"]["chats"]
                    if data and isinstance(data[0], dict) and "messages" in data[0]:
                        data = data[0]["messages"]

        if not isinstance(data, list):
            self.errors.append("Telegram export format not recognized")
            return ImportResult(events=[], format_detected="telegram_json",
                               errors=self.errors, warnings=self.warnings, stats={})

        # Group consecutive messages from same sender into events
        current_event = None
        event_messages = []
        last_sender = None
        event_counter = 0

        for msg in data:
            if not isinstance(msg, dict):
                continue

            text = msg.get("text") or msg.get("message", "")
            if not text or not isinstance(text, str):
                continue

            sender = msg.get("from") or msg.get("name", "Unknown")
            date_str = msg.get("date", "")[:10]  # YYYY-MM-DD

            if sender != last_sender or not event_messages:
                # Flush previous event
                if event_messages:
                    ev = self._make_event_from_messages(event_messages, event_counter)
                    if ev:
                        events.append(ev)
                    event_counter += 1

                last_sender = sender
                event_messages = [msg]
            else:
                event_messages.append(msg)

        # Flush last event
        if event_messages:
            ev = self._make_event_from_messages(event_messages, event_counter)
            if ev:
                events.append(ev)

        return ImportResult(
            events=events,
            format_detected="telegram_json",
            errors=self.errors,
            warnings=self.warnings,
            stats={"messages": len(data), "events_created": len(events)}
        )

    def _make_event_from_messages(self, messages: List[dict], counter: int) -> Optional[MemoryEvent]:
        if not messages:
            return None

        first = messages[0]
        date_str = first.get("date", "")[:10]
        sender = first.get("from") or first.get("name", "Unknown")
        texts = [m.get("text", m.get("message", "")) for m in messages if m.get("text")]
        content = " ".join(texts)

        title = f"Conversation with {sender}" if sender != "Unknown" else "Chat message"

        ev = self._make_event(
            id=f"CHAT{counter:04d}",
            time_str=date_str,
            title=title,
            content=content[:500],
            characters=[sender] if sender != "Unknown" else [],
        )
        return ev


# ─────────────────────────────────────────────────────────────────────────────
# JSONL Importer
# ─────────────────────────────────────────────────────────────────────────────

class JsonlImporter(BaseImporter):
    """Import from JSONL (one JSON object per line)"""

    def import_file(self, path: str) -> ImportResult:
        events = []
        self.errors = []
        self.warnings = []

        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    time_str = item.get("time") or item.get("date", "")
                    title = item.get("title", f"Line {i}")
                    content = item.get("content", "") or item.get("text", "")
                    chars = self._parse_list_field(item.get("characters", []))
                    locs = self._parse_list_field(item.get("locations", []))

                    ev = self._make_event(f"L{i:04d}", time_str, title, content, chars, locs)
                    if ev:
                        events.append(ev)
                except json.JSONDecodeError as e:
                    self.errors.append(f"[Line {i}] Invalid JSON: {e}")

        return ImportResult(
            events=events,
            format_detected="jsonl",
            errors=self.errors,
            warnings=self.warnings,
            stats={"lines": i, "imported": len(events), "failed": len(self.errors)}
        )

    def _parse_list_field(self, val) -> List[str]:
        if isinstance(val, list):
            return [str(v) for v in val]
        if isinstance(val, str):
            return [v.strip() for v in val.split(",") if v.strip()]
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Unified Import Function
# ─────────────────────────────────────────────────────────────────────────────

IMPORTERS = {
    "memoir_json": MemoirJsonImporter,
    "events_json": MemoirJsonImporter,
    "csv": CsvImporter,
    "obsidian_md": ObsidianImporter,
    "markdown_dir": ObsidianImporter,
    "telegram_json": TelegramImporter,
    "jsonl": JsonlImporter,
}


def import_events(path: str, format_hint: str = None) -> ImportResult:
    """
    Import events from a file or directory.

    Auto-detects format unless format_hint is provided.

    Returns ImportResult with events list, detected format, errors, and stats.
    """
    if not os.path.exists(path):
        return ImportResult(
            events=[],
            format_detected="unknown",
            errors=[f"Path not found: {path}"],
            warnings=[],
            stats={}
        )

    # Detect format
    detected = format_hint or FormatDetector.detect(path)

    # Get appropriate importer
    importer_class = IMPORTERS.get(detected)
    if not importer_class:
        return ImportResult(
            events=[],
            format_detected=detected,
            errors=[f"No importer available for format: {detected}"],
            warnings=[],
            stats={}
        )

    try:
        importer = importer_class()
        result = importer.import_file(path)
        result.format_detected = detected
        return result
    except Exception as e:
        return ImportResult(
            events=[],
            format_detected=detected,
            errors=[f"Import failed: {e}"],
            warnings=[],
            stats={}
        )
