"""
orchestrator.py — Phase 4 Orchestrator with Critic loop.

Flow:
  1. Decompose goal into subtasks (JSON)
  2. Execute each subtask with the right agent
  3. Critic evaluates the writer's final output
  4. If score < 7, requeue the failing agent with feedback
  5. Writer re-synthesises and critic re-evaluates
  6. Repeat up to max_retries times
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
from agents.critic import CriticAgent, CriticResult
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
2. Only include agents that are actually needed.
3. Keep task descriptions specific — always spell out acronyms fully.
4. Maximum 4 subtasks.

Example:
[
  {"agent": "researcher", "task": "find how Retrieval-Augmented Generation (RAG) works in AI"},
  {"agent": "writer", "task": "summarise the research into a clear explanation"}
]
"""


class Orchestrator:
    def __init__(
        self,
        llm: LLMWrapper,
        agent_llm: Optional[LLMWrapper] = None,
        verbose: bool = True,
        save_trace: bool = False,
        long_term_memory: Optional[MemoryManager] = None,
        critic_threshold: int = 7,
        max_retries: int = 2,
    ):
        self.llm = llm
        self.verbose = verbose
        self.save_trace = save_trace
        self.long_term_memory = long_term_memory
        self.critic_threshold = critic_threshold
        self.max_retries = max_retries
        self.run_id = ""

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

        self.critic = CriticAgent(
            llm=_agent_llm,
            verbose=verbose,
            save_trace=save_trace,
        )

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
            return "Orchestrator failed to decompose the goal."

        self._print(f"\n[plan] {len(subtasks)} subtask(s):")
        for i, t in enumerate(subtasks, 1):
            self._print(f"  {i}. [{t['agent']}] {t['task']}")

        # Step 2: Execute subtasks
        results: list[dict] = []
        for i, subtask in enumerate(subtasks):
            agent_name = subtask.get("agent", "").lower()
            task = subtask.get("task", "")

            if agent_name not in self.agents:
                self._print(f"\n[skip] Unknown agent '{agent_name}'.")
                continue

            if agent_name == "writer":
                task = self._build_writer_prompt(goal, results, task)

            self._print(f"\n[subtask {i+1}/{len(subtasks)}] → {agent_name}: {task[:80]}")
            self._log("subtask_start", {"agent": agent_name, "task": task[:200]})

            answer = self._run_agent_safe(agent_name, task)
            results.append({"agent": agent_name, "task": task, "result": answer})
            self._log("subtask_done", {"agent": agent_name, "result": answer[:200]})

        # Get writer output
        writer_results = [r for r in results if r["agent"] == "writer"]
        current_output = (
            writer_results[-1]["result"] if writer_results
            else results[-1]["result"] if results
            else "No results produced."
        )

        # Step 3: Critic loop
        current_output = self._critic_loop(goal, results, current_output)

        self._print(f"\n{'='*60}")
        self._print(f"FINAL ANSWER:\n{current_output}")
        self._print(f"{'='*60}")
        self._log("orchestrator_done", {"final": current_output[:300]})
        return current_output

    # ── Critic loop ───────────────────────────────────────────────────────────

    def _critic_loop(
        self, goal: str, results: list[dict], current_output: str
    ) -> str:
        """
        Evaluate output with the critic. If score < threshold,
        requeue the failing agent with feedback and re-synthesise.
        Repeat up to max_retries times.
        """
        self._print(f"\n[critic loop] Evaluating output (threshold={self.critic_threshold})...")

        for attempt in range(self.max_retries + 1):
            critique = self.critic.evaluate(
                goal=goal,
                output=current_output,
                run_id=self.run_id,
            )

            if critique.passed:
                self._print(f"[critic loop] ✓ Passed on attempt {attempt + 1}.")
                self._log("critic_passed", {
                    "attempt": attempt + 1,
                    "score": critique.score,
                })
                break

            if attempt >= self.max_retries:
                self._print(
                    f"[critic loop] Max retries ({self.max_retries}) reached. "
                    f"Returning best output (score={critique.score})."
                )
                self._log("critic_max_retries", {"score": critique.score})
                break

            # Requeue the failing agent with critic feedback
            self._print(
                f"\n[critic loop] Attempt {attempt + 1} failed "
                f"(score={critique.score}). Requeuing {critique.requeue_agent}..."
            )
            self._log("critic_retry", {
                "attempt": attempt + 1,
                "score": critique.score,
                "requeue": critique.requeue_agent,
            })

            current_output = self._requeue(
                goal=goal,
                results=results,
                critique=critique,
            )

        return current_output

    def _requeue(
        self, goal: str, results: list[dict], critique: CriticResult
    ) -> str:
        """
        Rerun the failing agent with critic feedback, then re-synthesise
        with the writer.
        """
        requeue_agent = critique.requeue_agent

        if requeue_agent == "none" or requeue_agent not in self.agents:
            # Just rerun the writer with the critique as additional context
            requeue_agent = "writer"

        self._print(f"[requeue] Running {requeue_agent} with critic feedback...")

        # Build improved task with critic instruction
        original_task = self._find_last_task(results, requeue_agent) or goal
        improved_task = (
            f"{original_task}\n\n"
            f"[Critic feedback — score was {critique.score}/10]\n"
            f"Issues to fix: {'; '.join(critique.issues)}\n"
            f"Instruction: {critique.instruction}"
        )

        improved_result = self._run_agent_safe(requeue_agent, improved_task)

        # Update results with the improved output
        for r in results:
            if r["agent"] == requeue_agent:
                r["result"] = improved_result
                break
        else:
            results.append({
                "agent": requeue_agent,
                "task": improved_task,
                "result": improved_result,
            })

        # Re-synthesise with writer if the requeued agent wasn't writer
        if requeue_agent != "writer":
            writer_task = self._build_writer_prompt(
                goal, results,
                "Re-synthesise all findings into a complete, improved response."
            )
            return self._run_agent_safe("writer", writer_task)

        return improved_result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_agent_safe(
        self, agent_name: str, task: str, max_retries: int = 2
    ) -> str:
        """Run an agent with exponential backoff on rate limit errors."""
        agent = self.agents[agent_name]
        for attempt in range(max_retries + 1):
            try:
                return agent.run(task)
            except Exception as e:
                err = str(e)
                if any(code in err for code in ("413", "429", "rate_limit", "TPM")):
                    if attempt < max_retries:
                        wait = 15 * (attempt + 1)
                        self._print(f"[rate limit] Waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        return (
                            f"Error: rate limit exceeded after {max_retries} retries."
                        )
                else:
                    raise
        return "Error: max retries exceeded."

    def _decompose(self, goal: str) -> list[dict]:
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
            self._print("[orchestrator] Parse failed — using fallback.")
            return self._fallback_plan(goal)
        except Exception as e:
            self._print(f"[orchestrator] Decompose error: {e}")
            return self._fallback_plan(goal)

    def _parse_subtasks(self, raw: str) -> list[dict]:
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        try:
            subtasks = json.loads(raw[start:end])
        except json.JSONDecodeError:
            return []
        return [
            t for t in subtasks
            if isinstance(t, dict)
            and "agent" in t and "task" in t
            and t["agent"] in self.agents
        ]

    def _fallback_plan(self, goal: str) -> list[dict]:
        self._print("[orchestrator] Using fallback: researcher → writer.")
        return [
            {"agent": "researcher", "task": goal},
            {"agent": "writer",     "task": f"Summarise findings for: {goal}"},
        ]

    def _build_writer_prompt(
        self, goal: str, results: list[dict], writer_task: str
    ) -> str:
        parts = [f"Original user goal: {goal}", "", "Results from other agents:"]
        for r in results:
            if r["agent"] != "writer":
                truncated = r["result"][:800] + (
                    "..." if len(r["result"]) > 800 else ""
                )
                parts.append(f"\n[{r['agent'].upper()}]\n{truncated}")
        parts += ["", f"Your task: {writer_task}"]
        return "\n".join(parts)

    def _find_last_task(self, results: list[dict], agent_name: str) -> Optional[str]:
        """Find the most recent task assigned to an agent."""
        for r in reversed(results):
            if r["agent"] == agent_name:
                return r["task"]
        return None

    def _print(self, msg: str):
        if self.verbose:
            print(msg)

    def _log(self, event: str, data: dict):
        if self.save_trace:
            log_event(event, data, run_id=self.run_id)