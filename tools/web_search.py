"""
tools/web_search.py — Web search via DuckDuckGo (no API key required).
Swap DDG for Tavily by setting TAVILY_API_KEY in your .env — the interface
stays identical so the agent loop doesn't change.
"""

import os

from tools.base import BaseTool


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Searches the web and returns titles, URLs, and snippets for the top results. "
        "Use when you need current information, facts, recent events, documentation, "
        "or anything you cannot compute directly."
    )

    def __init__(self, max_results: int = 5, provider: str = "auto"):
        """
        provider: "ddg" | "tavily" | "auto"
            auto → uses Tavily if TAVILY_API_KEY is set, else DuckDuckGo
        """
        self.max_results = max_results
        self.provider = provider

    def run(self, query: str) -> str:
        provider = self.provider
        if provider == "auto":
            provider = "tavily" if os.getenv("TAVILY_API_KEY") else "ddg"

        if provider == "tavily":
            return self._tavily(query)
        return self._ddg(query)

    def _ddg(self, query: str) -> str:
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))

            if not results:
                return "No results found."

            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] {r['title']}\n"
                    f"    URL: {r['href']}\n"
                    f"    {r['body']}"
                )
            return "\n\n".join(lines)

        except ImportError:
            return (
                "Error: duckduckgo-search not installed. "
                "Run: pip install duckduckgo-search"
            )
        except Exception as e:
            return f"DuckDuckGo search error: {e}"

    def _tavily(self, query: str) -> str:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            response = client.search(query, max_results=self.max_results)

            results = response.get("results", [])
            if not results:
                return "No results found."

            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] {r.get('title', 'No title')}\n"
                    f"    URL: {r.get('url', '')}\n"
                    f"    {r.get('content', '')}"
                )
            return "\n\n".join(lines)

        except ImportError:
            return (
                "Error: tavily-python not installed. "
                "Run: pip install tavily-python"
            )
        except Exception as e:
            return f"Tavily search error: {e}"

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query string.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
