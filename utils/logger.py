"""
utils/logger.py — Structured JSON-lines logger for agent traces.
Every agent run appends to logs/trace.jsonl.
In Phase 5 this feeds the observability dashboard.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_DIR = Path("logs")
TRACE_FILE = LOG_DIR / "trace.jsonl"


def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_event(event: str, data: dict[str, Any], run_id: str = ""):
    """Append a single structured event to logs/trace.jsonl."""
    _ensure_log_dir()
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "event": event,
        **data,
    }
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_traces(run_id: str = "") -> list[dict]:
    """Load all trace events, optionally filtered by run_id."""
    if not TRACE_FILE.exists():
        return []
    traces = []
    with open(TRACE_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if not run_id or entry.get("run_id") == run_id:
                    traces.append(entry)
            except json.JSONDecodeError:
                continue
    return traces


def print_trace(run_id: str = ""):
    """Pretty-print traces to stdout (useful in tests)."""
    traces = load_traces(run_id)
    for t in traces:
        ts = t.get("ts", "")
        event = t.get("event", "")
        data = {k: v for k, v in t.items() if k not in ("ts", "event", "run_id")}
        print(f"[{ts}] {event}: {json.dumps(data, indent=2)}")
