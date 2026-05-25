"""
tests/test_critic.py — Unit tests for CriticAgent and CriticResult.
"""

import json
import pytest
from unittest.mock import MagicMock

from agents.critic import CriticAgent, CriticResult
from llm import LLMResponse


def make_critic(response_json: dict) -> CriticAgent:
    llm = MagicMock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(response_json),
        tool_name=None,
        tool_args=None,
    )
    return CriticAgent(llm=llm, verbose=False)


PASS_RESPONSE = {
    "score": 8,
    "issues": ["minor formatting"],
    "strengths": ["accurate", "clear"],
    "requeue_agent": "none",
    "instruction": "",
}

FAIL_RESPONSE = {
    "score": 4,
    "issues": ["missing citations", "code has bugs"],
    "strengths": ["good structure"],
    "requeue_agent": "coder",
    "instruction": "Fix the code so it runs without errors and add citations.",
}


class TestCriticResult:
    def test_passed_when_score_gte_7(self):
        r = CriticResult({"score": 7, "issues": [], "strengths": [],
                          "requeue_agent": "none", "instruction": ""})
        assert r.passed is True

    def test_failed_when_score_lt_7(self):
        r = CriticResult({"score": 6, "issues": [], "strengths": [],
                          "requeue_agent": "writer", "instruction": "fix it"})
        assert r.passed is False

    def test_score_clamped_to_1_10(self):
        r = CriticResult({"score": 15, "issues": [], "strengths": [],
                          "requeue_agent": "none", "instruction": ""})
        assert r.score == 10

        r2 = CriticResult({"score": -5, "issues": [], "strengths": [],
                           "requeue_agent": "none", "instruction": ""})
        assert r2.score == 1

    def test_invalid_requeue_agent_defaults_to_none(self):
        r = CriticResult({"score": 5, "issues": [], "strengths": [],
                          "requeue_agent": "hacker", "instruction": ""})
        assert r.requeue_agent == "none"


class TestCriticAgent:
    def test_returns_critic_result(self):
        critic = make_critic(PASS_RESPONSE)
        result = critic.evaluate("some goal", "some output")
        assert isinstance(result, CriticResult)
        assert result.score == 8
        assert result.passed is True

    def test_fail_response_parsed(self):
        critic = make_critic(FAIL_RESPONSE)
        result = critic.evaluate("some goal", "some output")
        assert result.score == 4
        assert result.passed is False
        assert result.requeue_agent == "coder"
        assert "citations" in result.instruction

    def test_markdown_fences_stripped(self):
        llm = MagicMock()
        llm.chat.return_value = LLMResponse(
            content=f"```json\n{json.dumps(PASS_RESPONSE)}\n```",
            tool_name=None, tool_args=None,
        )
        critic = CriticAgent(llm=llm, verbose=False)
        result = critic.evaluate("goal", "output")
        assert result.score == 8

    def test_llm_error_returns_default_pass(self):
        llm = MagicMock()
        llm.chat.side_effect = Exception("API error")
        critic = CriticAgent(llm=llm, verbose=False)
        result = critic.evaluate("goal", "output")
        assert result.passed is True  # safe default on error
        assert "failed" in result.issues[0].lower()

    def test_invalid_json_raises_handled(self):
        llm = MagicMock()
        llm.chat.return_value = LLMResponse(
            content="not json at all", tool_name=None, tool_args=None
        )
        critic = CriticAgent(llm=llm, verbose=False)
        result = critic.evaluate("goal", "output")
        # Should not crash — returns safe default
        assert isinstance(result, CriticResult)


class TestCriticLoopInOrchestrator:
    def _make_orchestrator(self, critic_responses: list[dict]):
        from orchestrator import Orchestrator
        from llm import LLMResponse
        import json

        plan = json.dumps([
            {"agent": "researcher", "task": "find info"},
            {"agent": "writer", "task": "summarise"},
        ])

        llm = MagicMock()
        llm.chat.return_value = LLMResponse(
            content=plan, tool_name=None, tool_args=None
        )

        orch = Orchestrator(llm=llm, verbose=False, max_retries=2)

        for agent in orch.agents.values():
            agent.run = MagicMock(return_value="agent output")

        # Set up critic to return responses in sequence
        orch.critic.llm = MagicMock()
        orch.critic.llm.chat.side_effect = [
            LLMResponse(content=json.dumps(r), tool_name=None, tool_args=None)
            for r in critic_responses
        ]
        return orch

    def test_passes_on_first_attempt(self):
        orch = self._make_orchestrator([PASS_RESPONSE])
        result = orch.run("test goal")
        assert result == "agent output"

    def test_retries_on_fail_then_passes(self):
        orch = self._make_orchestrator([FAIL_RESPONSE, PASS_RESPONSE])
        result = orch.run("test goal")
        # Writer should have been called more than once (initial + retry)
        assert orch.agents["writer"].run.call_count >= 2

    def test_returns_after_max_retries(self):
        # Always fail — should stop at max_retries
        orch = self._make_orchestrator([FAIL_RESPONSE] * 5)
        result = orch.run("test goal")
        assert result is not None
        assert orch.agents["writer"].run.call_count <= 4