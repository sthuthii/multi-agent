"""
tools/web_search.py — Web search via DDGS (formerly duckduckgo-search).

Install: pip install ddgs
Optional: pip install tavily-python  (better quality, free tier at tavily.com)

If TAVILY_API_KEY is set in .env, Tavily is used automatically.
Otherwise falls back to DDGS (free, no key needed).
"""

import os
from tools.base import BaseTool


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Searches the web and returns titles, URLs, and snippets for the top results. "
        "Use when you need current information, facts, recent events, documentation, "
        "or anything you cannot compute directly. "
        "If a search returns no results, try a shorter or different query."
    )

    def __init__(self, max_results: int = 5, provider: str = "auto"):
        self.max_results = max_results
        self.provider = provider

    def run(self, query: str) -> str:
        provider = self.provider
        if provider == "auto":
            provider = "tavily" if os.getenv("TAVILY_API_KEY") else "ddgs"

        if provider == "tavily":
            return self._tavily(query)
        return self._ddgs(query)

    def _ddgs(self, query: str) -> str:
        try:
            from ddgs import DDGS
        except ImportError:
            # fallback to old package name
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                return (
                    "Error: search package not installed. "
                    "Run: pip install ddgs"
                )

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))

            if not results:
                return (
                    f"No results found for '{query}'. "
                    "Try a shorter or more general search query."
                )

            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] {r['title']}\n"
                    f"    URL: {r['href']}\n"
                    f"    {r['body']}"
                )
            return "\n\n".join(lines)

        except Exception as e:
            return (
                f"Search failed: {e}. "
                "Try a different query or use your own knowledge to answer."
            )

    def _tavily(self, query: str) -> str:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            response = client.search(query, max_results=self.max_results)
            results = response.get("results", [])

            if not results:
                return (
                    f"No results found for '{query}'. "
                    "Try a shorter or more general search query."
                )

            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] {r.get('title', 'No title')}\n"
                    f"    URL: {r.get('url', '')}\n"
                    f"    {r.get('content', '')}"
                )
            return "\n\n".join(lines)

        except ImportError:
            return "Error: tavily-python not installed. Run: pip install tavily-python"
        except Exception as e:
            return f"Tavily search failed: {e}"

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
                            "description": (
                                "The search query string. "
                                "Keep it short and specific for best results."
                            ),
                        }
                    },
                    "required": ["query"],
                },
            },
        }
