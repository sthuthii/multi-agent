"""
agent.py — Memory-aware single agent loop (Phase 2).
"""

import uuid
from typing import Optional

from llm import LLMWrapper, LLMResponse
from memory.buffer import ConversationBuffer
from memory.chroma import MemoryManager
from tools.base import BaseTool
from utils.logger import log_event


SYSTEM_PROMPT = """You are a helpful AI agent with access to tools.

Rules:
1. Use tools to answer — call web_search for facts, python_repl for computation, calculator for math.
2. After seeing a tool result, give your final answer in plain text.
3. If search returns off-topic results, retry with a more specific query including domain keywords.
4. If the conversation contains prior exchanges about past interactions, use them to answer recall questions.
5. For questions like "what did I ask before?" — look at the conversation history above and answer directly. No tools needed.

Your final answer must be plain text. Never output JSON, XML, or function call syntax as your answer.
"""



class Agent:

    # Patterns that indicate hallucinated tool syntax
    _HALLUCINATED_TOOL_PATTERNS = (
        "[Called ",
        "[Tool result",
        "<tool_call>",
        "<function_calls>",
        "<web_search>",
        '{"name":',
        "```json",
    )

    def __init__(
        self,
        llm: LLMWrapper,
        tools: list[BaseTool],
        max_iterations: int = 10,
        verbose: bool = True,
        save_trace: bool = False,
        long_term_memory: Optional[MemoryManager] = None,
        memory_top_k: int = 3,
        memory_min_score: float = 0.3,
    ):

        self.llm = llm
        self.tools: dict[str, BaseTool] = {t.name: t for t in tools}
        self.tool_schemas: list[dict] = [t.get_schema() for t in tools]

        self.max_iterations = max_iterations
        self.verbose = verbose
        self.save_trace = save_trace

        self.long_term_memory = long_term_memory
        self.memory_top_k = memory_top_k
        self.memory_min_score = memory_min_score

        self.run_id: str = ""
        self.iteration_count: int = 0
        self.memory: ConversationBuffer = ConversationBuffer()

    # ─────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────

    def run(self, goal: str) -> str:

        self._reset(goal)

        # Inject relevant long-term memory
        self._inject_long_term_memory(goal)

        for i in range(self.max_iterations):

            self.iteration_count = i + 1

            self._print(f"\n[iter {self.iteration_count}] Thinking...")

            response: LLMResponse = self.llm.chat(
                messages=self.memory.get(),
                tools=self.tool_schemas,
            )

            if response.wants_tool:
                self._handle_tool_call(response)
            else:
                return self._handle_final_answer(response, goal)

        return self._handle_max_iterations()

    # ─────────────────────────────────────────────
    # Memory
    # ─────────────────────────────────────────────

    def _inject_long_term_memory(self, goal: str):

        if self.long_term_memory is None:
            return

        context = self.long_term_memory.search_as_context(
            query=goal,
            n_results=self.memory_top_k,
            min_score=self.memory_min_score,
        )

        if context:
            self.memory.inject_context(context)

            self._print("[memory] Injected relevant memories.")

            self._log(
                "memory_injected",
                {"context_preview": context[:200]},
            )

    def _store_to_long_term_memory(self, goal: str, answer: str):
        if self.long_term_memory is None:
            return
        memory_text = (
            f"User asked: {goal[:200]}\n"
            f"Agent answered: {answer[:400]}"
        )
        doc_id = self.long_term_memory.store(
            text=memory_text,
            metadata={"run_id": self.run_id, "type": "qa_pair"},
        )
        self._print(f"[memory] Stored answer to long-term memory (id={doc_id}).")
        self._log("memory_stored", {"doc_id": doc_id})

    # ─────────────────────────────────────────────
    # Tool Calls
    # ─────────────────────────────────────────────

    def _handle_tool_call(self, response: LLMResponse):

        tool_name = response.tool_name
        tool_args = response.tool_args or {}

        self._print(
            f"[iter {self.iteration_count}] → {tool_name}({tool_args})"
        )

        self._log(
            "tool_call",
            {
                "iteration": self.iteration_count,
                "tool": tool_name,
                "args": tool_args,
            },
        )

        if tool_name not in self.tools:

            tool_result = (
                f"Error: unknown tool '{tool_name}'. "
                f"Available: {list(self.tools)}"
            )

        else:

            try:
                tool_result = self.tools[tool_name].run(**tool_args)

            except TypeError as e:
                tool_result = f"Error: bad arguments: {e}"

            except Exception as e:
                tool_result = f"Tool error: {e}"

        preview = tool_result[:300]

        if len(tool_result) > 300:
            preview += "..."

        self._print(
            f"[iter {self.iteration_count}] ← {preview}"
        )

        self._log(
            "tool_result",
            {
                "iteration": self.iteration_count,
                "tool": tool_name,
                "result": tool_result,
            },
        )

        self.memory.add(
            "assistant",
            f"[Called {tool_name} with args: {tool_args}]",
        )

        self.memory.add(
            "user",
            f"[Tool result from {tool_name}]:\n{tool_result}",
        )

    # ─────────────────────────────────────────────
    # Final Answer
    # ─────────────────────────────────────────────

    def _handle_final_answer(
        self,
        response: LLMResponse,
        goal: str,
    ) -> str:

        answer = response.content or "(empty response)"

        # Detect hallucinated raw tool syntax
        if any(
            answer.strip().startswith(p)
            for p in self._HALLUCINATED_TOOL_PATTERNS
        ):

            self._print(
                f"[iter {self.iteration_count}] ⚠ Hallucinated tool call detected."
            )

            self.memory.add(
                "user",
                (
                    "Please answer in plain English only. "
                    "No JSON, no XML, no tool syntax."
                ),
            )

            retry = self.llm.chat(
                messages=self.memory.get(),
                tools=self.tool_schemas,
            )

            answer = retry.content or "(no answer produced)"

        self._print(
            f"\n✓ Done in {self.iteration_count} iteration(s).\n"
        )

        self._log(
            "final_answer",
            {
                "answer": answer,
                "iterations": self.iteration_count,
            },
        )

        RECALL_PHRASES = ("what did i ask", "what did you say", "what did we discuss",
                  "what do you remember", "previous conversation", "asked before")

        if not any(p in goal.lower() for p in RECALL_PHRASES):
            self._store_to_long_term_memory(goal, answer)
        else:
            self._print("[memory] Skipping storage — recall question.")

        return answer

    # ─────────────────────────────────────────────
    # Max Iterations
    # ─────────────────────────────────────────────

    def _handle_max_iterations(self) -> str:

        msg = (
            f"Reached max iterations ({self.max_iterations}) "
            f"without a final answer."
        )

        self._log(
            "max_iterations_reached",
            {"limit": self.max_iterations},
        )

        return msg

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _reset(self, goal: str):

        self.run_id = str(uuid.uuid4())[:8]

        self.iteration_count = 0

        self.memory = ConversationBuffer()

        self.memory.add("system", SYSTEM_PROMPT)

        self.memory.add("user", goal)

        self._print(f"\n{'=' * 60}")

        self._print(f"GOAL: {goal}")

        self._print(
            f"run_id={self.run_id}  "
            f"max_iter={self.max_iterations}  "
            f"tools={list(self.tools)}  "
            f"ltm={'on' if self.long_term_memory else 'off'}"
        )

        self._print(f"{'=' * 60}")

        self._log("run_start", {"goal": goal})

    def _print(self, msg: str):

        if self.verbose:
            print(msg)

    def _log(self, event: str, data: dict):

        if self.save_trace:
            log_event(event, data, run_id=self.run_id)

    def __repr__(self):

        return (
            f"<Agent llm={self.llm} "
            f"tools={list(self.tools)} "
            f"max_iter={self.max_iterations} "
            f"ltm={'on' if self.long_term_memory else 'off'}>"
        )