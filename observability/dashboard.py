"""
observability/dashboard.py — Trace replay and run analytics.

Reads logs/trace.jsonl and provides:
  - Per-run summaries
  - Agent call timelines
  - Token cost estimates
  - Failure analysis
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

TRACE_FILE = Path("logs/trace.jsonl")


def load_runs(trace_file: Path = TRACE_FILE) -> dict[str, list[dict]]:
    """Load all trace events grouped by run_id."""
    if not trace_file.exists():
        return {}

    runs: dict[str, list[dict]] = defaultdict(list)
    with open(trace_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                run_id = entry.get("run_id", "unknown")
                runs[run_id].append(entry)
            except json.JSONDecodeError:
                continue
    return dict(runs)


def summarise_run(events: list[dict]) -> dict:
    """Produce a summary dict for a single run."""
    summary = {
        "run_id": "",
        "goal": "",
        "final_answer": "",
        "iterations": 0,
        "tool_calls": [],
        "critic_scores": [],
        "retries": 0,
        "errors": [],
        "agents_used": [],
        "started_at": "",
        "finished_at": "",
    }

    for e in events:
        event = e.get("event", "")
        summary["run_id"] = e.get("run_id", "")
        summary["started_at"] = summary["started_at"] or e.get("ts", "")
        summary["finished_at"] = e.get("ts", "")

        if event == "run_start":
            summary["goal"] = e.get("goal", "")

        elif event == "orchestrator_start":
            summary["goal"] = e.get("goal", "")

        elif event == "final_answer":
            summary["final_answer"] = e.get("answer", "")
            summary["iterations"] = e.get("iterations", 0)

        elif event == "orchestrator_done":
            summary["final_answer"] = e.get("final", "")

        elif event == "tool_call":
            summary["tool_calls"].append({
                "tool": e.get("tool"),
                "iteration": e.get("iteration"),
            })

        elif event == "critic_evaluation":
            summary["critic_scores"].append(e.get("score", 0))

        elif event == "critic_retry":
            summary["retries"] += 1

        elif event == "subtask_start":
            agent = e.get("agent")
            if agent and agent not in summary["agents_used"]:
                summary["agents_used"].append(agent)

        elif event in ("max_iterations_reached", "critic_max_retries"):
            summary["errors"].append(event)

    return summary


def print_run_summary(run_id: str, events: list[dict]):
    """Pretty-print a single run summary to stdout."""
    s = summarise_run(events)
    print(f"\n{'─'*60}")
    print(f"Run ID      : {s['run_id']}")
    print(f"Goal        : {s['goal'][:80]}")
    print(f"Started     : {s['started_at']}")
    print(f"Iterations  : {s['iterations']}")
    print(f"Tool calls  : {len(s['tool_calls'])}")
    print(f"Agents used : {', '.join(s['agents_used']) or 'single-agent'}")

    if s["critic_scores"]:
        avg = sum(s["critic_scores"]) / len(s["critic_scores"])
        print(f"Critic scores: {s['critic_scores']}  (avg={avg:.1f})")
        print(f"Retries     : {s['retries']}")

    if s["errors"]:
        print(f"Errors      : {s['errors']}")

    print(f"Answer      : {s['final_answer'][:120]}...")


def print_all_runs(trace_file: Path = TRACE_FILE):
    """Print summaries for all runs in the trace file."""
    runs = load_runs(trace_file)
    if not runs:
        print(f"No traces found in {trace_file}")
        return

    print(f"\n{'='*60}")
    print(f"TRACE REPLAY — {len(runs)} run(s) found")
    print(f"{'='*60}")

    for run_id, events in runs.items():
        print_run_summary(run_id, events)

    # Aggregate stats
    all_summaries = [summarise_run(e) for e in runs.values()]
    total_tool_calls = sum(len(s["tool_calls"]) for s in all_summaries)
    total_errors = sum(len(s["errors"]) for s in all_summaries)
    all_scores = [sc for s in all_summaries for sc in s["critic_scores"]]

    print(f"\n{'='*60}")
    print(f"AGGREGATE STATS")
    print(f"{'='*60}")
    print(f"Total runs        : {len(runs)}")
    print(f"Total tool calls  : {total_tool_calls}")
    print(f"Total errors      : {total_errors}")
    print(f"Error rate        : {total_errors}/{len(runs)} runs")
    if all_scores:
        print(f"Avg critic score  : {sum(all_scores)/len(all_scores):.1f}")
    print(f"Avg tool calls/run: {total_tool_calls/len(runs):.1f}")