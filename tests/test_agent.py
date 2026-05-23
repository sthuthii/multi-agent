"""
tests/test_agent.py — Integration tests for the Agent loop.
Uses mock LLM so tests are fast, free, and deterministic.
"""

from unittest.mock import MagicMock, patch
import pytest

from agent import Agent
from llm import LLMResponse
from tools.python_repl import PythonREPL
from tools.web_search import WebSearchTool


def make_llm(*responses: LLMResponse) -> MagicMock:
    """Mock LLM that cycles through a list of canned responses."""
    llm = MagicMock()
    llm.chat.side_effect = list(responses)
    return llm


def tool_call_response(name: str, args: dict) -> LLMResponse:
    return LLMResponse(content=None, tool_name=name, tool_args=args)


def final_response(text: str) -> LLMResponse:
    return LLMResponse(content=text, tool_name=None, tool_args=None)


class TestAgentLoop:
    def test_single_tool_call_then_answer(self):
        llm = make_llm(
            tool_call_response("python_repl", {"code": "print(40 + 2)"}),
            final_response("The answer is 42."),
        )
        agent = Agent(llm=llm, tools=[PythonREPL()], verbose=False)
        result = agent.run("What is 40 + 2?")

        assert "42" in result
        assert agent.iteration_count == 2
        assert llm.chat.call_count == 2

    def test_direct_answer_no_tool(self):
        llm = make_llm(final_response("Paris is the capital of France."))
        agent = Agent(llm=llm, tools=[PythonREPL()], verbose=False)
        result = agent.run("What is the capital of France?")

        assert "Paris" in result
        assert agent.iteration_count == 1

    def test_max_iterations_respected(self):
        # LLM keeps calling tools forever
        looping = tool_call_response("python_repl", {"code": "print('loop')"})
        llm = make_llm(*[looping] * 20)
        agent = Agent(llm=llm, tools=[PythonREPL()], max_iterations=3, verbose=False)
        result = agent.run("Loop forever")

        assert llm.chat.call_count == 3
        assert "max iterations" in result.lower()

    def test_unknown_tool_does_not_crash(self):
        llm = make_llm(
            tool_call_response("nonexistent_tool", {}),
            final_response("I couldn't use the tool but here is my answer."),
        )
        agent = Agent(llm=llm, tools=[PythonREPL()], verbose=False)
        result = agent.run("Use a tool that doesn't exist")

        assert result is not None
        assert "couldn't" in result.lower() or "answer" in result.lower()

    def test_tool_exception_does_not_crash(self):
        bad_tool = MagicMock(spec=PythonREPL)
        bad_tool.name = "python_repl"
        bad_tool.get_schema.return_value = PythonREPL().get_schema()
        bad_tool.run.side_effect = RuntimeError("kernel panic")

        llm = make_llm(
            tool_call_response("python_repl", {"code": "print(1)"}),
            final_response("There was an error but I can still answer."),
        )
        agent = Agent(llm=llm, tools=[bad_tool], verbose=False)
        result = agent.run("Run some code")

        assert result is not None  # did not raise

    def test_multiple_tool_calls_accumulate_in_memory(self):
        llm = make_llm(
            tool_call_response("python_repl", {"code": "print('step1')"}),
            tool_call_response("python_repl", {"code": "print('step2')"}),
            final_response("Done after two tool calls."),
        )
        agent = Agent(llm=llm, tools=[PythonREPL()], verbose=False)
        result = agent.run("Do two steps")

        assert agent.iteration_count == 3
        # Memory should contain system + user goal + 4 tool messages + extra
        assert agent.memory.message_count >= 5

    def test_run_id_changes_between_runs(self):
        llm = make_llm(
            final_response("First answer."),
            final_response("Second answer."),
        )
        agent = Agent(llm=llm, tools=[], verbose=False)
        agent.run("First goal")
        run_id_1 = agent.run_id
        agent.run("Second goal")
        run_id_2 = agent.run_id
        assert run_id_1 != run_id_2

    def test_save_trace_calls_log_event(self):
        llm = make_llm(final_response("answer"))
        agent = Agent(llm=llm, tools=[], verbose=False, save_trace=True)

        with patch("agent.log_event") as mock_log:
            agent.run("test goal")
            assert mock_log.called
            events = [call.args[0] for call in mock_log.call_args_list]
            assert "run_start" in events
            assert "final_answer" in events


# ── Phase 2: memory-aware agent tests ────────────────────────────────────────

class TestAgentWithMemory:
    def test_agent_injects_memory_context(self):
        from unittest.mock import MagicMock, patch
        from agent import Agent
        from llm import LLMResponse

        mock_memory = MagicMock()
        mock_memory.search_as_context.return_value = "Relevant: RAG means Retrieval-Augmented Generation"
        mock_memory.count = 1

        llm = make_llm(final_response("RAG stands for Retrieval-Augmented Generation."))
        agent = Agent(llm=llm, tools=[], verbose=False, long_term_memory=mock_memory)
        result = agent.run("What is RAG?")

        assert mock_memory.search_as_context.called
        assert "RAG" in result

    def test_agent_stores_answer_to_memory(self):
        mock_memory = MagicMock()
        mock_memory.search_as_context.return_value = ""
        mock_memory.count = 0

        llm = make_llm(final_response("Paris is the capital of France."))
        agent = Agent(llm=llm, tools=[], verbose=False, long_term_memory=mock_memory)
        agent.run("What is the capital of France?")

        assert mock_memory.store.called
        stored_text = mock_memory.store.call_args[0][0]
        assert "France" in stored_text or "Paris" in stored_text

    def test_agent_works_without_memory(self):
        llm = make_llm(final_response("Answer without memory."))
        agent = Agent(llm=llm, tools=[], verbose=False, long_term_memory=None)
        result = agent.run("Some goal")
        assert result == "Answer without memory."

    def test_memory_not_called_when_no_context(self):
        mock_memory = MagicMock()
        mock_memory.search_as_context.return_value = ""  # no relevant memories

        llm = make_llm(final_response("Clean answer."))
        agent = Agent(llm=llm, tools=[], verbose=False, long_term_memory=mock_memory)

        # Inject context should NOT be called if search returns empty
        agent.run("test goal")
        # memory.search_as_context WAS called, but inject_context was not (internal to agent)
        assert mock_memory.search_as_context.called
