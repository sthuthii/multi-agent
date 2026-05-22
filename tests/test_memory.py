"""
tests/test_memory.py — Unit tests for ConversationBuffer.
"""

from memory.buffer import ConversationBuffer


class TestConversationBuffer:
    def test_add_and_get(self):
        buf = ConversationBuffer()
        buf.add("user", "hello")
        messages = buf.get()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"

    def test_preserves_order(self):
        buf = ConversationBuffer()
        buf.add("user", "first")
        buf.add("assistant", "second")
        buf.add("user", "third")
        msgs = buf.get()
        assert msgs[0]["content"] == "first"
        assert msgs[1]["content"] == "second"
        assert msgs[2]["content"] == "third"

    def test_system_prompt_never_evicted(self):
        buf = ConversationBuffer(max_tokens=50)
        buf.add("system", "You are an agent.")
        # flood with user messages
        for i in range(30):
            buf.add("user", "x" * 20)
        msgs = buf.get()
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are an agent."

    def test_trims_when_over_limit(self):
        buf = ConversationBuffer(max_tokens=100)
        for i in range(50):
            buf.add("user", "x" * 20)
        tokens = buf.estimated_tokens
        assert tokens <= 110  # small tolerance for trim granularity

    def test_clear_keeps_system_prompt(self):
        buf = ConversationBuffer()
        buf.add("system", "system prompt")
        buf.add("user", "hello")
        buf.add("assistant", "hi")
        buf.clear()
        msgs = buf.get()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_message_count(self):
        buf = ConversationBuffer()
        assert buf.message_count == 0
        buf.add("user", "hello")
        assert buf.message_count == 1

    def test_get_returns_copy(self):
        buf = ConversationBuffer()
        buf.add("user", "hello")
        msgs = buf.get()
        msgs.append({"role": "user", "content": "injected"})
        assert buf.message_count == 1  # internal state unchanged
