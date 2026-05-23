"""
tests/test_memory.py — Unit tests for ConversationBuffer and MemoryManager.
ChromaDB tests use an in-memory client to avoid disk I/O.
"""

import pytest
from unittest.mock import patch, MagicMock
from memory.buffer import ConversationBuffer


# ── ConversationBuffer ────────────────────────────────────────────────────────

class TestConversationBuffer:
    def test_add_and_get(self):
        buf = ConversationBuffer()
        buf.add("user", "hello")
        msgs = buf.get()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"

    def test_preserves_order(self):
        buf = ConversationBuffer()
        buf.add("user", "first")
        buf.add("assistant", "second")
        buf.add("user", "third")
        msgs = buf.get()
        assert msgs[0]["content"] == "first"
        assert msgs[2]["content"] == "third"

    def test_system_prompt_never_evicted(self):
        buf = ConversationBuffer(max_tokens=50)
        buf.add("system", "You are an agent.")
        for i in range(30):
            buf.add("user", "x" * 20)
        msgs = buf.get()
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are an agent."

    def test_trims_when_over_limit(self):
        buf = ConversationBuffer(max_tokens=100)
        for i in range(50):
            buf.add("user", "x" * 20)
        assert buf.estimated_tokens <= 120

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
        assert buf.message_count == 1

    def test_get_recent(self):
        buf = ConversationBuffer()
        buf.add("system", "sys")
        for i in range(5):
            buf.add("user", f"msg {i}")
        recent = buf.get_recent(3)
        assert len(recent) == 3
        assert recent[-1]["content"] == "msg 4"

    def test_inject_context_added_after_system(self):
        buf = ConversationBuffer()
        buf.add("system", "You are an agent.")
        buf.add("user", "hello")
        buf.inject_context("Relevant: RAG means Retrieval-Augmented Generation")
        msgs = buf.get()
        assert msgs[1]["content"].startswith("[Memory context]")
        assert "RAG" in msgs[1]["content"]

    def test_inject_context_replaces_previous(self):
        buf = ConversationBuffer()
        buf.add("system", "sys")
        buf.inject_context("first context")
        buf.inject_context("second context")
        context_msgs = [m for m in buf.get() if "[Memory context]" in m["content"]]
        assert len(context_msgs) == 1
        assert "second" in context_msgs[0]["content"]

    def test_inject_empty_context_is_noop(self):
        buf = ConversationBuffer()
        buf.add("system", "sys")
        buf.add("user", "hello")
        count_before = buf.message_count
        buf.inject_context("")
        assert buf.message_count == count_before

    def test_compression_triggered_at_threshold(self):
        buf = ConversationBuffer(max_tokens=200, compress_at=0.8)
        buf.add("system", "sys")
        # Add enough messages to trigger compression
        for i in range(20):
            buf.add("user", f"message number {i} with some content here")
        # After compression, tokens should be under budget
        assert buf.estimated_tokens <= 250  # allow small overshoot before trim

    def test_summarise_returns_string(self):
        buf = ConversationBuffer()
        buf.add("system", "sys")
        buf.add("user", "question")
        buf.add("assistant", "answer")
        summary = buf.summarise()
        assert isinstance(summary, str)
        assert "USER" in summary or "user" in summary.lower()
        assert "question" in summary


# ── MemoryManager (mocked ChromaDB) ──────────────────────────────────────────

class TestMemoryManager:
    """
    Tests use a mock ChromaDB collection to avoid disk I/O and
    the sentence-transformers dependency.
    """

    def _make_manager(self):
        from memory.chroma import MemoryManager
        mgr = MemoryManager(agent_name="test_agent")
        # Mock the collection directly
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mgr._collection = mock_collection
        mgr._client = MagicMock()
        return mgr, mock_collection

    def test_store_calls_upsert(self):
        mgr, col = self._make_manager()
        col.count.return_value = 0
        mgr.store("RAG stands for Retrieval-Augmented Generation")
        assert col.upsert.called
        call_kwargs = col.upsert.call_args
        assert "documents" in call_kwargs.kwargs or len(call_kwargs.args) > 0

    def test_search_returns_empty_when_no_memories(self):
        mgr, col = self._make_manager()
        col.count.return_value = 0
        results = mgr.search("what is RAG")
        assert results == []

    def test_search_returns_results(self):
        mgr, col = self._make_manager()
        col.count.return_value = 2
        col.query.return_value = {
            "documents": [["RAG is Retrieval-Augmented Generation", "LLM stands for Large Language Model"]],
            "distances": [[0.1, 0.4]],
            "metadatas": [[{"agent": "test"}, {"agent": "test"}]],
        }
        results = mgr.search("what is RAG", n_results=2)
        assert len(results) == 2
        assert results[0]["text"] == "RAG is Retrieval-Augmented Generation"
        assert results[0]["score"] > results[1]["score"]  # sorted by similarity

    def test_search_as_context_empty_when_no_results(self):
        mgr, col = self._make_manager()
        col.count.return_value = 0
        context = mgr.search_as_context("something")
        assert context == ""

    def test_search_as_context_filters_by_min_score(self):
        mgr, col = self._make_manager()
        col.count.return_value = 1
        col.query.return_value = {
            "documents": [["low relevance memory"]],
            "distances": [[0.95]],  # distance 0.95 → similarity 0.05, below min_score
            "metadatas": [[{"agent": "test"}]],
        }
        context = mgr.search_as_context("query", min_score=0.3)
        assert context == ""

    def test_safe_collection_name(self):
        from memory.chroma import MemoryManager
        assert MemoryManager._safe_collection_name("researcher") == "researcher"
        assert MemoryManager._safe_collection_name("My Agent!") == "my-agent"
        assert len(MemoryManager._safe_collection_name("a" * 100)) <= 63

    def test_count_property(self):
        mgr, col = self._make_manager()
        col.count.return_value = 42
        assert mgr.count == 42
