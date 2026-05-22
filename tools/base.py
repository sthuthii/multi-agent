"""
tools/base.py — Abstract base class for all tools.
Every tool in Phase 1+ must inherit from BaseTool.
"""

from abc import ABC, abstractmethod


class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs) -> str:
        """Execute the tool and return a string result."""
        pass

    def get_schema(self) -> dict:
        """
        Returns an OpenAI-compatible function schema.
        Override in each subclass with the correct parameters.
        """
        raise NotImplementedError(
            f"Tool '{self.name}' must implement get_schema()."
        )

    def __repr__(self):
        return f"<Tool name={self.name}>"
