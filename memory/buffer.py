"""
memory/buffer.py — Rolling conversation buffer with token cap + compression.

Phase 2 addition: compress_old_messages() summarises old messages into a
single summary message instead of hard-dropping them, so context is preserved.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Message:
    role: str          # "system" | "user" | "assistant"
    content: str
    tool_name: Optional[str] = None


class ConversationBuffer:
    """
    Keeps a rolling window of messages that fits within max_tokens.

    Eviction policy (Phase 1): drop oldest non-system messages first.
    Compression (Phase 2):     when over 80% of token budget, summarise
                               old messages into a single [Summary] message
                               before falling back to hard eviction.

    The system prompt at index 0 is never evicted or compressed.
    """

    def __init__(self, max_tokens: int = 6000, compress_at: float = 0.8):
        """
        max_tokens   : hard token budget for the buffer
        compress_at  : fraction of max_tokens at which compression triggers
                       (default 0.8 = compress when 80% full)
        """
        self.max_tokens = max_tokens
        self.compress_at = compress_at
        self._messages: list[Message] = []
        self._compression_threshold = int(max_tokens * compress_at)

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, role: str, content: str, tool_name: Optional[str] = None):
        self._messages.append(
            Message(role=role, content=content, tool_name=tool_name)
        )
        if self._estimate_tokens() > self._compression_threshold:
            self._compress()
        self._trim()

    def get(self) -> list[dict]:
        """Returns messages in OpenAI API format (role + content only)."""
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def get_recent(self, n: int) -> list[dict]:
        """Returns the n most recent messages (excluding system prompt)."""
        non_system = [m for m in self._messages if m.role != "system"]
        return [{"role": m.role, "content": m.content} for m in non_system[-n:]]

    def inject_context(self, context: str):
        """
        Inject a context string (e.g. long-term memory results) as a
        system-level message just after the main system prompt.
        Replaces any existing injected context to avoid duplication.
        """
        if not context:
            return

        # Remove previous injected context if present
        self._messages = [
            m for m in self._messages
            if not (m.role == "system" and m.content.startswith("[Memory context]"))
        ]

        # Insert after system prompt (index 0), or at start if no system prompt
        insert_at = 1 if self._messages and self._messages[0].role == "system" else 0
        self._messages.insert(
            insert_at,
            Message(role="system", content=f"[Memory context]\n{context}"),
        )

    def clear(self):
        """Reset buffer, keeping only the system prompt if present."""
        system = [m for m in self._messages if m.role == "system"]
        self._messages = system[:1]  # keep only the first system message

    def summarise(self) -> str:
        """
        Return a plain-text summary of all non-system messages.
        Used by the agent before storing a run into long-term memory.
        """
        lines = []
        for m in self._messages:
            if m.role == "system":
                continue
            prefix = m.role.upper()
            lines.append(f"{prefix}: {m.content[:300]}")
        return "\n".join(lines)

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def estimated_tokens(self) -> int:
        return self._estimate_tokens()

    # ── Private ───────────────────────────────────────────────────────────────

    def _compress(self):
        """
        Summarise the oldest 40% of non-system messages into a single
        [Summary] message. Preserves meaning without hard-dropping content.
        Falls back to _trim() if compression doesn't free enough space.
        """
        non_system_indices = [
            i for i, m in enumerate(self._messages)
            if m.role != "system" and not m.content.startswith("[Summary]")
            and not m.content.startswith("[Memory context]")
        ]

        if len(non_system_indices) < 4:
            return  # not enough messages to compress meaningfully

        # Compress oldest 40% of non-system messages
        n_to_compress = max(2, int(len(non_system_indices) * 0.4))
        to_compress = non_system_indices[:n_to_compress]

        summary_parts = []
        for i in to_compress:
            m = self._messages[i]
            summary_parts.append(f"{m.role}: {m.content[:200]}")

        summary_text = "[Summary of earlier conversation]\n" + "\n".join(summary_parts)

        # Replace the compressed messages with a single summary
        for i in sorted(to_compress, reverse=True):
            self._messages.pop(i)

        insert_at = next(
            (i for i, m in enumerate(self._messages) if m.role != "system"), 1
        )
        self._messages.insert(
            insert_at,
            Message(role="user", content=summary_text),
        )

    def _trim(self):
        """Hard eviction: drop oldest non-system messages until under budget."""
        while self._estimate_tokens() > self.max_tokens and len(self._messages) > 2:
            for i, m in enumerate(self._messages):
                if m.role != "system":
                    self._messages.pop(i)
                    break

    def _estimate_tokens(self) -> int:
        return sum(len(m.content) // 4 for m in self._messages)

    def __repr__(self):
        return (
            f"<ConversationBuffer messages={self.message_count} "
            f"~tokens={self.estimated_tokens}/{self.max_tokens}>"
        )
