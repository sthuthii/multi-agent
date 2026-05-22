"""
agent.py — Core single-agent loop (Phase 1).
The plan → act → observe → reflect loop lives here.
Phase 2 will extend this with an Orchestrator that creates
and coordinates multiple Agent instances.
"""

import uuid
from typing import Optional

from llm import LLMWrapper, LLMResponse
from memory.buffer import ConversationBuffer
from tools.base import BaseTool
from utils.logger import log_event


SYSTEM_PROMPT = """You are a capable AI agent with access to tools.

When given a goal:
1. Think step by step about what information or computation you need.
2. Call a tool if you need external information or to run code.
3. After observing the tool result, decide:
   - If you have enough to answer → reply directly in plain text (no tool call).
   - If you need more → call another tool.
4. Never guess facts you can verify with a tool.
5. Give a clear, concise final answer when done.

Important: when you are ready to give the final answer, respond with plain text ONLY.
Do NOT call a tool just to format or summarise — write the answer directly.
"""


class Agent:
    """
    A single ReAct-style agent.

    Loop:
      for each iteration:
        1. PLAN  — ask LLM what to do next (tool call or final answer)
        2. ACT   — if tool call, execute the tool
        3. OBSERVE — append tool result to context
        4. REFLECT — LLM decides whether to continue or answer
    """

    def __init__(
        self,
        llm: LLMWrapper,
        tools: list[BaseTool],
        max_iterations: int = 10,
        verbose: bool = True,
        save_trace: bool = False,
    ):
        self.llm = llm
        self.tools: dict[str, BaseTool] = {t.name: t for t in tools}
        self.tool_schemas: list[dict] = [t.get_schema() for t in tools]
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.save_trace = save_trace

        # Set per run
        self.run_id: str = ""
        self.iteration_count: int = 0
        self.memory: ConversationBuffer = ConversationBuffer()

    # ── Public ───────────────────────────────────────────────────────────────

    def run(self, goal: str) -> str:
        """Execute the agent loop for a given goal. Returns the final answer."""
        self._reset(goal)

        for i in range(self.max_iterations):
            self.iteration_count = i + 1
            self._print(f"\n[iter {self.iteration_count}] Thinking...")

            response: LLMResponse = self.llm.chat(
                messages=self.memory.get(),
                tools=self.tool_schemas,
            )

            if response.wants_tool:
                self._handle_tool_call(response, i)
            else:
                return self._handle_final_answer(response)

        return self._handle_max_iterations()

    # ── Private — loop steps ─────────────────────────────────────────────────

    def _handle_tool_call(self, response: LLMResponse, iteration: int):
        """ACT + OBSERVE."""
        tool_name = response.tool_name
        tool_args = response.tool_args or {}

        self._print(f"[iter {self.iteration_count}] → {tool_name}({tool_args})")
        self._log("tool_call", {"iteration": self.iteration_count, "tool": tool_name, "args": tool_args})

        # ACT
        if tool_name not in self.tools:
            tool_result = f"Error: unknown tool '{tool_name}'. Available: {list(self.tools)}"
        else:
            try:
                tool_result = self.tools[tool_name].run(**tool_args)
            except TypeError as e:
                tool_result = f"Error: bad arguments for tool '{tool_name}': {e}"
            except Exception as e:
                tool_result = f"Tool error: {e}"

        preview = tool_result[:300] + ("..." if len(tool_result) > 300 else "")
        self._print(f"[iter {self.iteration_count}] ← {preview}")
        self._log("tool_result", {"iteration": self.iteration_count, "tool": tool_name, "result": tool_result})

        # OBSERVE — feed result back into context
        self.memory.add("assistant", f"[Called {tool_name} with args: {tool_args}]")
        self.memory.add("user", f"[Tool result from {tool_name}]:\n{tool_result}")

    def _handle_final_answer(self, response: LLMResponse) -> str:
        """REFLECT — LLM produced a final answer."""
        answer = response.content or "(empty response)"
        self._print(f"\n✓ Done in {self.iteration_count} iteration(s).\n")
        self._log("final_answer", {"answer": answer, "iterations": self.iteration_count})
        return answer

    def _handle_max_iterations(self) -> str:
        msg = (
            f"Reached max iterations ({self.max_iterations}) without a final answer. "
            "Consider increasing --max-iter or simplifying the goal."
        )
        self._log("max_iterations_reached", {"limit": self.max_iterations})
        return msg

    # ── Private — helpers ─────────────────────────────────────────────────────

    def _reset(self, goal: str):
        self.run_id = str(uuid.uuid4())[:8]
        self.iteration_count = 0
        self.memory = ConversationBuffer()
        self.memory.add("system", SYSTEM_PROMPT)
        self.memory.add("user", goal)
        self._print(f"\n{'='*60}")
        self._print(f"GOAL: {goal}")
        self._print(f"run_id={self.run_id}  max_iter={self.max_iterations}  tools={list(self.tools)}")
        self._print(f"{'='*60}")
        self._log("run_start", {"goal": goal})

    def _print(self, msg: str):
        if self.verbose:
            print(msg)

    def _log(self, event: str, data: dict):
        if self.save_trace:
            log_event(event, data, run_id=self.run_id)

    def __repr__(self):
        return (
            f"<Agent llm={self.llm} tools={list(self.tools)} "
            f"max_iter={self.max_iterations}>"
        )
