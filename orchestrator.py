"""
orchestrator.py — Phase 3 Orchestrator.

Decomposes a user goal into subtasks, routes each to the right
specialist agent, collects results, and passes everything to the
Writer agent for final synthesis.

Routing schema (JSON the LLM must produce):
[
  {"agent": "researcher", "task": "find X"},
  {"agent": "coder",      "task": "write code to do Y"},
  {"agent": "writer",     "task": "synthesise the above into Z"}
]
"""

import json
import re
import time
import uuid
from typing import Optional

from llm import LLMWrapper
from agents.researcher import ResearcherAgent
from agents.coder import CoderAgent
from agents.writer import WriterAgent
from tools import PythonREPL, WebSearchTool, FileWriteTool, CalculatorTool
from memory.chroma import MemoryManager
from utils.logger import log_event


ORCHESTRATOR_PROMPT = """You are an Orchestrator. You decompose user goals into subtasks
and assign each to the right specialist agent.

Available agents:
  researcher — finds information, searches the web, summarises sources
  coder      — writes and executes Python code, does computation
  writer     — synthesises research + code output into a final response

Your response must be a valid JSON array of subtasks. Nothing else — no explanation, no markdown.

Rules:
1. Always end with a writer task that synthesises all previous results.
2. Only include agents that are actually needed. Simple factual questions need only researcher + writer.
3. Keep task descriptions specific and self-contained.
4. Maximum 4 subtasks.

Example output:
[
  {"agent": "researcher", "task": "find the top 3 use cases of RAG in enterprise LLM applications"},
  {"agent": "writer", "task": "combine the research findings into a clear explanation"}
]
"""


class Orchestrator:
    """
    Coordinates specialist agents to complete a complex goal.

    Flow:
      1. Decompose goal into subtasks (LLM → JSON)
      2. Execute each subtask with the right agent
      3. Accumulate results
      4. Writer agent synthesises final output
    """

    def __init__(
        self,
        llm: LLMWrapper,
        agent_llm: Optional[LLMWrapper] = None,
        verbose: bool = True,
        save_trace: bool = False,
        long_term_memory: Optional[MemoryManager] = None,
    ):
        self.llm = llm                    # used for decomposition
        self.verbose = verbose
        self.save_trace = save_trace
        self.long_term_memory = long_term_memory
        self.run_id = ""

        # Sub-agents use agent_llm (can be a faster/smaller model)
        _agent_llm = agent_llm or llm

        self.agents = {
            "researcher": ResearcherAgent(
                llm=_agent_llm,
                tools=[WebSearchTool()],
                verbose=verbose,
                save_trace=save_trace,
                long_term_memory=long_term_memory,
            ),
            "coder": CoderAgent(
                llm=_agent_llm,
                tools=[PythonREPL(), CalculatorTool(), FileWriteTool()],
                verbose=verbose,
                save_trace=save_trace,
            ),
            "writer": WriterAgent(
                llm=_agent_llm,
                tools=[],
                verbose=verbose,
                save_trace=save_trace,
            ),
        }

    def run(self, goal: str) -> str:
        self.run_id = str(uuid.uuid4())[:8]
        self._print(f"\n{'='*60}")
        self._print(f"ORCHESTRATOR  run_id={self.run_id}")
        self._print(f"GOAL: {goal}")
        self._print(f"{'='*60}")
        self._log("orchestrator_start", {"goal": goal})

        # Step 1: Decompose
        subtasks = self._decompose(goal)
        if not subtasks:
            return "Orchestrator failed to decompose the goal into subtasks."

        self._print(f"\n[plan] {len(subtasks)} subtask(s):")
        for i, t in enumerate(subtasks, 1):
            self._print(f"  {i}. [{t['agent']}] {t['task']}")

        # Step 2: Execute subtasks, accumulate results
        results: list[dict] = []
        for i, subtask in enumerate(subtasks):
            agent_name = subtask.get("agent", "").lower()
            task = subtask.get("task", "")

            if agent_name not in self.agents:
                self._print(f"\n[skip] Unknown agent '{agent_name}' — skipping.")
                continue

            # Writer gets all previous results injected into its task
            if agent_name == "writer":
                task = self._build_writer_prompt(goal, results, task)

            self._print(f"\n[subtask {i+1}/{len(subtasks)}] → {agent_name}: {task[:80]}")
            self._log("subtask_start", {"agent": agent_name, "task": task[:200]})

            answer = self._run_with_retry(agent_name, task)
            results.append({"agent": agent_name, "task": task, "result": answer})
            self._log("subtask_done", {"agent": agent_name, "result": answer[:200]})

        # Step 3: Return writer's output, or last result if no writer ran
        writer_results = [r for r in results if r["agent"] == "writer"]
        final = (
            writer_results[-1]["result"] if writer_results
            else results[-1]["result"] if results
            else "No results produced."
        )

        self._print(f"\n{'='*60}")
        self._print(f"FINAL ANSWER:\n{final}")
        self._print(f"{'='*60}")
        self._log("orchestrator_done", {"final": final[:300]})
        return final

    # ── Private ───────────────────────────────────────────────────────────────

    def _run_with_retry(self, agent_name: str, task: str, max_retries: int = 2) -> str:
        """Run an agent with exponential backoff on rate limit errors."""
        agent = self.agents[agent_name]
        for attempt in range(max_retries + 1):
            try:
                return agent.run(task)
            except Exception as e:
                err = str(e)
                if "413" in err or "rate_limit" in err or "TPM" in err or "429" in err:
                    if attempt < max_retries:
                        wait = 15 * (attempt + 1)
                        self._print(f"[rate limit] Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                        time.sleep(wait)
                    else:
                        return f"Error: rate limit exceeded after {max_retries} retries. Try a smaller model or reduce max_iter."
                else:
                    raise
        return "Error: max retries exceeded."

    def _decompose(self, goal: str) -> list[dict]:
        """Ask the LLM to decompose the goal into a JSON subtask list."""
        self._print("\n[orchestrator] Decomposing goal...")

        messages = [
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user",   "content": f"Decompose this goal into subtasks:\n{goal}"},
        ]

        try:
            response = self.llm.chat(messages=messages, tools=None)
            raw = response.content or ""
            subtasks = self._parse_subtasks(raw)

            if subtasks:
                self._print(f"[orchestrator] Plan: {len(subtasks)} subtask(s).")
                self._log("decompose_success", {"subtasks": subtasks})
                return subtasks
            else:
                self._print(f"[orchestrator] Failed to parse plan, using fallback.")
                self._log("decompose_failed", {"raw": raw[:300]})
                return self._fallback_plan(goal)

        except Exception as e:
            self._print(f"[orchestrator] Decompose error: {e}")
            return self._fallback_plan(goal)

    def _parse_subtasks(self, raw: str) -> list[dict]:
        """Extract and validate JSON subtask array from LLM response."""
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []

        try:
            subtasks = json.loads(raw[start:end])
        except json.JSONDecodeError:
            return []

        valid = []
        for t in subtasks:
            if isinstance(t, dict) and "agent" in t and "task" in t:
                if t["agent"] in self.agents:
                    valid.append(t)
        return valid

    def _fallback_plan(self, goal: str) -> list[dict]:
        """Safe default: researcher → writer."""
        self._print("[orchestrator] Using fallback plan: researcher → writer.")
        return [
            {"agent": "researcher", "task": goal},
            {"agent": "writer",     "task": f"Summarise the research findings for: {goal}"},
        ]

    def _build_writer_prompt(
        self, goal: str, results: list[dict], writer_task: str
    ) -> str:
        """Build a concise writer prompt from all prior results.
        Truncates each result to 800 chars to stay within token limits."""
        parts = [
            f"Original user goal: {goal}",
            "",
            "Results from other agents:",
        ]
        for r in results:
            if r["agent"] != "writer":
                truncated = r["result"][:800] + ("..." if len(r["result"]) > 800 else "")
                parts.append(f"\n[{r['agent'].upper()}]\n{truncated}")

        parts += ["", f"Your task: {writer_task}"]
        return "\n".join(parts)

    def _print(self, msg: str):
        if self.verbose:
            print(msg)

    def _log(self, event: str, data: dict):
        if self.save_trace:
            log_event(event, data, run_id=self.run_id)