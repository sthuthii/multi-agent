"""
tests/test_tools.py — Unit tests for tools.
These run offline (no LLM calls).
"""

import pytest
from tools.python_repl import PythonREPL
from tools.web_search import WebSearchTool


class TestPythonREPL:
    def setup_method(self):
        self.tool = PythonREPL(timeout=5)

    def test_basic_arithmetic(self):
        result = self.tool.run(code="print(2 + 2)")
        assert "4" in result

    def test_multiline_code(self):
        code = "x = [i**2 for i in range(5)]\nprint(x)"
        result = self.tool.run(code=code)
        assert "16" in result  # 4^2 = 16

    def test_timeout_enforced(self):
        result = self.tool.run(code="while True: pass")
        assert "timed out" in result.lower()

    def test_syntax_error_returned_not_raised(self):
        result = self.tool.run(code="def broken(")
        assert result  # returns something
        assert "Error" in result or "SyntaxError" in result

    def test_runtime_error_returned_not_raised(self):
        result = self.tool.run(code="raise ValueError('test error')")
        assert "ValueError" in result

    def test_zero_division_returned_not_raised(self):
        result = self.tool.run(code="print(1 / 0)")
        assert "ZeroDivision" in result

    def test_schema_has_required_fields(self):
        schema = self.tool.get_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "python_repl"
        assert "code" in schema["function"]["parameters"]["properties"]

    def test_import_works(self):
        result = self.tool.run(code="import math\nprint(math.sqrt(144))")
        assert "12" in result


class TestWebSearchTool:
    def setup_method(self):
        self.tool = WebSearchTool()

    def test_schema_has_required_fields(self):
        schema = self.tool.get_schema()
        assert schema["function"]["name"] == "web_search"
        assert "query" in schema["function"]["parameters"]["properties"]

    def test_returns_string(self):
        # We just test it returns a non-empty string
        # (may fail if no network, that's ok — integration concern)
        result = self.tool.run(query="Python programming language")
        assert isinstance(result, str)
        assert len(result) > 0
