"""
tools/python_repl.py — Safe Python code execution via subprocess.
Uses a temp file + timeout to sandbox execution.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from tools.base import BaseTool


class PythonREPL(BaseTool):
    name = "python_repl"
    description = (
        "Executes Python 3 code in an isolated subprocess. "
        "Returns stdout on success or the error message on failure. "
        "Use for calculations, data processing, string manipulation, "
        "generating sequences, or any deterministic logic. "
        "Do NOT use for tasks requiring network access or file I/O outside /tmp."
    )

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def run(self, code: str) -> str:
        with tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write(code)
            fname = f.name

        try:
            result = subprocess.run(
                [sys.executable, fname],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = result.stdout.strip()
            error = result.stderr.strip()

            if output and error:
                return f"{output}\n[stderr]: {error}"
            return output or error or "(no output)"

        except subprocess.TimeoutExpired:
            return f"Error: execution timed out after {self.timeout}s."
        except Exception as e:
            return f"Error running code: {e}"
        finally:
            Path(fname).unlink(missing_ok=True)

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Valid Python 3 source code to execute.",
                        }
                    },
                    "required": ["code"],
                },
            },
        }
