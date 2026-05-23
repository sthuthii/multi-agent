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


# ── Phase 2 tool tests ────────────────────────────────────────────────────────

class TestCalculatorTool:
    def setup_method(self):
        from tools.calculator import CalculatorTool
        self.tool = CalculatorTool()

    def test_basic_arithmetic(self):
        assert self.tool.run("2 + 2") == "4"
        assert self.tool.run("10 - 3") == "7"
        assert self.tool.run("6 * 7") == "42"
        assert self.tool.run("10 / 4") == "2.5"

    def test_power(self):
        assert self.tool.run("2 ** 10") == "1024"

    def test_sqrt(self):
        assert self.tool.run("sqrt(144)") == "12"

    def test_floor_div_and_mod(self):
        assert self.tool.run("17 // 5") == "3"
        assert self.tool.run("17 % 5") == "2"

    def test_pi_constant(self):
        result = float(self.tool.run("pi"))
        assert abs(result - 3.14159) < 0.001

    def test_zero_division(self):
        result = self.tool.run("1 / 0")
        assert "zero" in result.lower()

    def test_disallows_dangerous_expressions(self):
        result = self.tool.run("__import__('os').system('ls')")
        assert "Error" in result or "not allowed" in result.lower()

    def test_disallows_exec(self):
        result = self.tool.run("exec('print(1)')")
        assert "Error" in result

    def test_schema_valid(self):
        schema = self.tool.get_schema()
        assert schema["function"]["name"] == "calculator"
        assert "expression" in schema["function"]["parameters"]["properties"]


class TestFileWriteTool:
    def setup_method(self):
        from tools.file_write import FileWriteTool
        self.tool = FileWriteTool()

    def test_writes_file_successfully(self):
        result = self.tool.run("test_output.txt", "hello world")
        assert "successfully" in result.lower()
        assert "test_output.txt" in result

    def test_returns_path_in_result(self):
        result = self.tool.run("myfile.md", "# Title\nContent here")
        assert "/tmp/agent_outputs/" in result

    def test_sanitises_dangerous_filename(self):
        result = self.tool.run("../../etc/passwd", "malicious")
        assert "/tmp/agent_outputs/" in result
        assert "etc" not in result or "passwd" not in result

    def test_schema_valid(self):
        schema = self.tool.get_schema()
        assert schema["function"]["name"] == "file_write"
        props = schema["function"]["parameters"]["properties"]
        assert "filename" in props
        assert "content" in props
