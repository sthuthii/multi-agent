"""
tools/file_write.py — Write text content to a file in /tmp.
Returns the absolute path so other tools (or the user) can reference it.

Safe by design: writes only to /tmp/agent_outputs/ to prevent path traversal.
"""

import os
import re
from pathlib import Path

from tools.base import BaseTool

OUTPUT_DIR = Path("/tmp/agent_outputs")


class FileWriteTool(BaseTool):
    name = "file_write"
    description = (
        "Writes text content to a file and returns the file path. "
        "Use when you need to save a report, code file, summary, or any "
        "text output that should persist beyond this conversation. "
        "Filename should include an appropriate extension (.txt, .md, .py, .json, etc.)."
    )

    def run(self, filename: str, content: str) -> str:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Sanitise filename — strip path components, allow only safe chars
        safe_name = re.sub(r"[^\w.\-]", "_", Path(filename).name)
        if not safe_name or safe_name.startswith("."):
            safe_name = "output.txt"

        file_path = OUTPUT_DIR / safe_name

        try:
            file_path.write_text(content, encoding="utf-8")
            size_kb = round(file_path.stat().st_size / 1024, 2)
            return (
                f"File written successfully.\n"
                f"Path: {file_path}\n"
                f"Size: {size_kb} KB\n"
                f"Lines: {content.count(chr(10)) + 1}"
            )
        except OSError as e:
            return f"Error writing file: {e}"

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": (
                                "Name of the file to create, including extension. "
                                "Example: 'summary.md', 'analysis.py', 'results.json'"
                            ),
                        },
                        "content": {
                            "type": "string",
                            "description": "The full text content to write to the file.",
                        },
                    },
                    "required": ["filename", "content"],
                },
            },
        }
