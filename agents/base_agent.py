"""
agents/base_agent.py — Specialist agent base class.
Extends Agent with a role name and role-specific system prompt.
"""

import uuid
from typing import Optional

from agent import Agent
from llm import LLMWrapper
from memory.buffer import ConversationBuffer
from memory.chroma import MemoryManager
from tools.base import BaseTool


class BaseSpecialistAgent(Agent):
    role: str = "specialist"
    role_prompt: str = ""

    def __init__(
        self,
        llm: LLMWrapper,
        tools: list[BaseTool],
        max_iterations: int = 6,
        verbose: bool = True,
        save_trace: bool = False,
        long_term_memory: Optional[MemoryManager] = None,
    ):
        super().__init__(
            llm=llm,
            tools=tools,
            max_iterations=max_iterations,
            verbose=verbose,
            save_trace=save_trace,
            long_term_memory=long_term_memory,
        )

    def _reset(self, goal: str):
        """Use role-specific system prompt instead of generic one."""
        self.run_id = str(uuid.uuid4())[:8]
        self.iteration_count = 0
        self.memory = ConversationBuffer()
        self.memory.add("system", self.role_prompt)
        self.memory.add("user", goal)
        self._print(f"\n  [{self.role.upper()}] {goal[:100]}")
        self._log("run_start", {"role": self.role, "goal": goal})

    def __repr__(self):
        return f"<{self.role.capitalize()}Agent tools={list(self.tools)}>"