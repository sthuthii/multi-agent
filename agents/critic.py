"""
agents/critic.py — Critic specialist agent.
Scores output 1-10 with structured JSON feedback and triggers reruns.
"""

import json
import re
from typing import Optional

from llm import LLMWrapper
from utils.logger import log_event


CRITIC_PROMPT = """You are a Critic Agent. You evaluate the quality of AI-generated responses.

Score the response on a scale of 1-10 based on:
- Accuracy: is the information correct?
- Completeness: does it fully address the goal?
- Clarity: is it well-structured and easy to understand?
- Code quality (if code present): does it run, is it correct, is it readable?

You MUST respond with valid JSON only. No explanation outside the JSON.

Format:
{
  "score": <integer 1-10>,
  "issues": ["issue 1", "issue 2"],
  "strengths": ["strength 1", "strength 2"],
  "requeue_agent": "<researcher|coder|writer|none>",
  "instruction": "<specific instruction for improvement, or empty string if score >= 7>"
}

Rules:
- score >= 7: good enough, set requeue_agent to "none"
- score < 7: set requeue_agent to the agent most responsible for the issue
- instruction must be specific and actionable, not vague
- issues and strengths must each have at least one item
"""


class CriticResult:
    """Structured result from the Critic agent."""

    def __init__(self, raw: dict):
        self.score: int = int(raw.get("score", 5))
        self.issues: list[str] = raw.get("issues", [])
        self.strengths: list[str] = raw.get("strengths", [])
        self.requeue_agent: str = raw.get("requeue_agent", "none")
        self.instruction: str = raw.get("instruction", "")

    @property
    def passed(self) -> bool:
        return self.score >= 7

    def __repr__(self):
        return (
            f"<CriticResult score={self.score} "
            f"passed={self.passed} "
            f"requeue={self.requeue_agent}>"
        )


class CriticAgent:
    """
    Evaluates agent output and returns a structured CriticResult.
    Not a subclass of Agent — has its own simple request/response loop
    since it never needs tools.
    """

    def __init__(
        self,
        llm: LLMWrapper,
        verbose: bool = True,
        save_trace: bool = False,
    ):
        self.llm = llm
        self.verbose = verbose
        self.save_trace = save_trace

    def evaluate(self, goal: str, output: str, run_id: str = "") -> CriticResult:
        """
        Evaluate the output against the original goal.
        Returns a CriticResult with score, issues, and retry instruction.
        """
        messages = [
            {"role": "system", "content": CRITIC_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original goal:\n{goal}\n\n"
                    f"Response to evaluate:\n{output[:2000]}"
                ),
            },
        ]

        try:
            response = self.llm.chat(messages=messages, tools=None)
            raw_text = response.content or ""
            result = self._parse(raw_text)
        except Exception as e:
            self._print(f"[critic] Error during evaluation: {e}")
            # Default: pass with a warning
            result = CriticResult({
                "score": 7,
                "issues": [f"Critic evaluation failed: {e}"],
                "strengths": ["Could not evaluate"],
                "requeue_agent": "none",
                "instruction": "",
            })

        self._print(
            f"[critic] Score: {result.score}/10  "
            f"{'✓ PASS' if result.passed else '✗ FAIL'}  "
            f"requeue={result.requeue_agent}"
        )
        if not result.passed:
            self._print(f"[critic] Issues: {result.issues}")
            self._print(f"[critic] Instruction: {result.instruction}")

        if self.save_trace:
            log_event("critic_evaluation", {
                "score": result.score,
                "passed": result.passed,
                "issues": result.issues,
                "requeue_agent": result.requeue_agent,
                "instruction": result.instruction,
            }, run_id=run_id)

        return result

    def _parse(self, raw: str) -> CriticResult:
        """Parse JSON from LLM response robustly."""
        # Strip markdown fences
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

        # Find JSON object
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in critic response: {clean[:100]}")

        data = json.loads(clean[start:end])

        # Validate score range
        score = int(data.get("score", 5))
        data["score"] = max(1, min(10, score))

        # Ensure requeue_agent is valid
        valid_agents = {"researcher", "coder", "writer", "none"}
        if data.get("requeue_agent") not in valid_agents:
            data["requeue_agent"] = "none"

        return CriticResult(data)

    def _print(self, msg: str):
        if self.verbose:
            print(msg)