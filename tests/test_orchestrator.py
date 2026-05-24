"""
tests/test_orchestrator.py — Orchestrator unit tests (mock LLM + mock agents).
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from orchestrator import Orchestrator
from llm import LLMResponse


def make_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.chat.return_value = LLMResponse(
        content=content, tool_name=None, tool_args=None
    )
    return llm


def make_orchestrator(plan_json: str, verbose: bool = False) -> Orchestrator:
    llm = make_llm(plan_json)
    orch = Orchestrator(llm=llm, verbose=verbose)
    for agent in orch.agents.values():
        agent.run = MagicMock(return_value=f"{agent.role} result")
    return orch


VALID_PLAN = json.dumps([
    {"agent": "researcher", "task": "find info about RAG"},
    {"agent": "writer",     "task": "summarise the findings"},
])


class TestDecompose:
    def test_valid_plan_parsed(self):
        orch = make_orchestrator(VALID_PLAN)
        subtasks = orch._decompose("What is RAG?")
        assert len(subtasks) == 2
        assert subtasks[0]["agent"] == "researcher"
        assert subtasks[1]["agent"] == "writer"

    def test_markdown_fences_stripped(self):
        orch = make_orchestrator(f"```json\n{VALID_PLAN}\n```")
        subtasks = orch._decompose("What is RAG?")
        assert len(subtasks) == 2

    def test_unknown_agent_filtered(self):
        plan = json.dumps([
            {"agent": "hacker",     "task": "exploit"},
            {"agent": "researcher", "task": "find info"},
            {"agent": "writer",     "task": "write it"},
        ])
        orch = make_orchestrator(plan)
        subtasks = orch._decompose("test")
        assert all(t["agent"] != "hacker" for t in subtasks)
        assert len(subtasks) == 2

    def test_invalid_json_uses_fallback(self):
        orch = make_orchestrator("sorry I cannot do that")
        subtasks = orch._decompose("some goal")
        agents = [t["agent"] for t in subtasks]
        assert "researcher" in agents
        assert "writer" in agents

    def test_missing_fields_filtered(self):
        plan = json.dumps([
            {"agent": "researcher"},
            {"task": "do something"},
            {"agent": "writer", "task": "write"},
        ])
        orch = make_orchestrator(plan)
        subtasks = orch._decompose("test")
        assert len(subtasks) == 1
        assert subtasks[0]["agent"] == "writer"


class TestRun:
    def test_all_subtasks_executed(self):
        orch = make_orchestrator(VALID_PLAN)
        orch.run("What is RAG?")
        orch.agents["researcher"].run.assert_called_once()
        orch.agents["writer"].run.assert_called_once()

    def test_writer_result_is_final(self):
        orch = make_orchestrator(VALID_PLAN)
        result = orch.run("What is RAG?")
        assert result == "writer result"

    def test_writer_receives_previous_results(self):
        orch = make_orchestrator(VALID_PLAN)
        orch.run("tell me about RAG")
        writer_call = orch.agents["writer"].run.call_args[0][0]
        assert "RESEARCHER" in writer_call or "researcher result" in writer_call

    def test_writer_prompt_truncates_long_results(self):
        orch = make_orchestrator(VALID_PLAN)
        orch.agents["researcher"].run = MagicMock(return_value="x" * 2000)
        orch.run("test")
        writer_call = orch.agents["writer"].run.call_args[0][0]
        # Result should be truncated to 800 chars + "..."
        assert len(writer_call) < 2000

    def test_fallback_plan_runs_on_bad_json(self):
        orch = make_orchestrator("not valid json")
        result = orch.run("some goal")
        assert result is not None
        orch.agents["researcher"].run.assert_called()
        orch.agents["writer"].run.assert_called()

    def test_coder_included_when_needed(self):
        plan = json.dumps([
            {"agent": "coder",  "task": "write fibonacci code"},
            {"agent": "writer", "task": "explain the code"},
        ])
        orch = make_orchestrator(plan)
        orch.run("write fibonacci")
        orch.agents["coder"].run.assert_called_once()

    def test_rate_limit_retry(self):
        orch = make_orchestrator(VALID_PLAN)
        call_count = {"n": 0}

        def flaky_run(task):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception("413 Request too large TPM rate_limit")
            return "recovered result"

        orch.agents["researcher"].run = flaky_run
        # patch sleep to avoid waiting in tests
        with patch("orchestrator.time.sleep"):
            result = orch.run("test")
        assert call_count["n"] == 2  # failed once, succeeded on retry