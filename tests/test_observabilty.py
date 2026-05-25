"""
tests/test_observability.py — Unit tests for observability/dashboard.py
"""

import json
import tempfile
from pathlib import Path
from observability.dashboard import load_runs, summarise_run, print_all_runs


def write_trace(events: list[dict]) -> Path:
    """Write events to a temp trace file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for e in events:
        tmp.write(json.dumps(e) + "\n")
    tmp.close()
    return Path(tmp.name)


SAMPLE_EVENTS = [
    {"ts": "2024-01-01T00:00:00", "run_id": "abc123", "event": "run_start",      "goal": "What is RAG?"},
    {"ts": "2024-01-01T00:00:01", "run_id": "abc123", "event": "tool_call",      "tool": "web_search", "iteration": 1},
    {"ts": "2024-01-01T00:00:02", "run_id": "abc123", "event": "tool_result",    "tool": "web_search", "result": "RAG is..."},
    {"ts": "2024-01-01T00:00:03", "run_id": "abc123", "event": "final_answer",   "answer": "RAG stands for Retrieval-Augmented Generation.", "iterations": 2},
]

ORCH_EVENTS = [
    {"ts": "2024-01-01T00:01:00", "run_id": "def456", "event": "orchestrator_start", "goal": "Research RAG"},
    {"ts": "2024-01-01T00:01:01", "run_id": "def456", "event": "subtask_start",      "agent": "researcher", "task": "find RAG info"},
    {"ts": "2024-01-01T00:01:02", "run_id": "def456", "event": "subtask_start",      "agent": "writer",     "task": "summarise"},
    {"ts": "2024-01-01T00:01:03", "run_id": "def456", "event": "critic_evaluation",  "score": 8, "passed": True},
    {"ts": "2024-01-01T00:01:04", "run_id": "def456", "event": "orchestrator_done",  "final": "RAG is a technique..."},
]


class TestLoadRuns:
    def test_loads_single_run(self):
        path = write_trace(SAMPLE_EVENTS)
        runs = load_runs(path)
        assert "abc123" in runs
        assert len(runs["abc123"]) == 4

    def test_loads_multiple_runs(self):
        path = write_trace(SAMPLE_EVENTS + ORCH_EVENTS)
        runs = load_runs(path)
        assert len(runs) == 2
        assert "abc123" in runs
        assert "def456" in runs

    def test_empty_file_returns_empty(self):
        path = write_trace([])
        runs = load_runs(path)
        assert runs == {}

    def test_missing_file_returns_empty(self):
        runs = load_runs(Path("nonexistent_file.jsonl"))
        assert runs == {}

    def test_malformed_lines_skipped(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        tmp.write("not json\n")
        tmp.write(json.dumps(SAMPLE_EVENTS[0]) + "\n")
        tmp.close()
        runs = load_runs(Path(tmp.name))
        assert len(runs) == 1


class TestSummariseRun:
    def test_extracts_goal(self):
        s = summarise_run(SAMPLE_EVENTS)
        assert s["goal"] == "What is RAG?"

    def test_extracts_final_answer(self):
        s = summarise_run(SAMPLE_EVENTS)
        assert "RAG" in s["final_answer"]

    def test_counts_tool_calls(self):
        s = summarise_run(SAMPLE_EVENTS)
        assert len(s["tool_calls"]) == 1
        assert s["tool_calls"][0]["tool"] == "web_search"

    def test_extracts_critic_scores(self):
        s = summarise_run(ORCH_EVENTS)
        assert s["critic_scores"] == [8]

    def test_extracts_agents_used(self):
        s = summarise_run(ORCH_EVENTS)
        assert "researcher" in s["agents_used"]
        assert "writer" in s["agents_used"]

    def test_counts_retries(self):
        events = ORCH_EVENTS + [
            {"ts": "T", "run_id": "def456", "event": "critic_retry", "attempt": 1, "score": 4}
        ]
        s = summarise_run(events)
        assert s["retries"] == 1

    def test_records_errors(self):
        events = SAMPLE_EVENTS + [
            {"ts": "T", "run_id": "abc123", "event": "max_iterations_reached"}
        ]
        s = summarise_run(events)
        assert "max_iterations_reached" in s["errors"]

    def test_empty_events_returns_defaults(self):
        s = summarise_run([])
        assert s["goal"] == ""
        assert s["tool_calls"] == []
        assert s["critic_scores"] == []