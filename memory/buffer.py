"""
memory/buffer.py — Rolling conversation buffer with a rough token cap.
Phase 2 will add a long-term ChromaDB layer on top of this.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str          # "system" | "user" | "assistant"
    content: str
    tool_name: Optional[str] = None   # set when role == "tool_result"


class ConversationBuffer:
    """
    Keeps a rolling window of messages that fits within max_tokens.
    Eviction policy: drop oldest non-system messages first.
    The system prompt at index 0 is never evicted.
    """

    def __init__(self, max_tokens: int = 6000):
        self.max_tokens = max_tokens
        self._messages: list[Message] = []

    def add(self, role: str, content: str, tool_name: Optional[str] = None):
        self._messages.append(Message(role=role, content=content, tool_name=tool_name))
        self._trim()

    def get(self) -> list[dict]:
        """Returns messages in OpenAI API format (role + content only)."""
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def clear(self):
        """Reset buffer, keeping only the system prompt if present."""
        system = [m for m in self._messages if m.role == "system"]
        self._messages = system

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def estimated_tokens(self) -> int:
        return self._estimate_tokens()

    # ── Private ──────────────────────────────────────────────────────────────

    def _trim(self):
        while self._estimate_tokens() > self.max_tokens and len(self._messages) > 2:
            # Find the first non-system message and remove it
            for i, m in enumerate(self._messages):
                if m.role != "system":
                    self._messages.pop(i)
                    break

    def _estimate_tokens(self) -> int:
        # Rough estimate: 1 token ≈ 4 characters
        return sum(len(m.content) // 4 for m in self._messages)

    def __repr__(self):
        return (
            f"<ConversationBuffer messages={self.message_count} "
            f"~tokens={self.estimated_tokens}>"
        )
