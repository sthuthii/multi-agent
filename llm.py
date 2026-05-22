"""
llm.py — LLM wrapper supporting OpenAI, Groq, and Ollama.

Supported providers:
  openai  → requires OPENAI_API_KEY in .env
  groq    → requires GROQ_API_KEY in .env  (free at console.groq.com)
  ollama  → requires Ollama running locally (ollama serve)

Usage:
  llm = LLMWrapper(provider="groq", model="llama-3.1-8b-instant")
  llm = LLMWrapper(provider="ollama", model="mistral")
  llm = LLMWrapper(provider="openai", model="gpt-4o-mini")
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LLMResponse:
    """Normalised response from any LLM provider."""
    content: Optional[str]      # Final text answer (None if tool call)
    tool_name: Optional[str]    # Tool the LLM wants to call
    tool_args: Optional[dict]   # Arguments for the tool call
    raw: Any = None             # Original API response (for debugging)

    @property
    def wants_tool(self) -> bool:
        return self.tool_name is not None


# ── Provider defaults ─────────────────────────────────────────────────────────

PROVIDER_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "groq":   "llama-3.1-8b-instant",
    "ollama": "mistral",
}

GROQ_MODELS = {
    "llama-3.1-8b-instant",     # fastest, good tool calling
    "llama-3.3-70b-versatile",  # best quality on free tier
    "mixtral-8x7b-32768",       # long context
    "gemma2-9b-it",             # lightweight
}


class LLMWrapper:
    """
    Unified wrapper around OpenAI, Groq, and Ollama APIs.
    The chat() interface is identical for all three providers.

    Groq tip: use llama-3.1-8b-instant for speed during development,
    llama-3.3-70b-versatile for evals and demos.
    """

    OLLAMA_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        provider: str = "groq",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        if provider not in PROVIDER_DEFAULTS:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose from: {list(PROVIDER_DEFAULTS)}"
            )

        self.provider = provider
        self.model = model or PROVIDER_DEFAULTS[provider]
        self.temperature = temperature
        self.client = self._build_client(provider, base_url)

    def _build_client(self, provider: str, base_url: Optional[str]):
        if provider == "groq":
            try:
                from groq import Groq
            except ImportError:
                raise ImportError(
                    "groq package not installed. Run: pip install groq"
                )
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set.\n"
                    "  1. Go to https://console.groq.com\n"
                    "  2. Create a free account\n"
                    "  3. Generate an API key\n"
                    "  4. Add to .env: GROQ_API_KEY=gsk_..."
                )
            return Groq(api_key=api_key)

        elif provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package not installed. Run: pip install openai"
                )
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY is not set. Add it to your .env file."
                )
            return OpenAI(api_key=api_key)

        elif provider == "ollama":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package not installed. Run: pip install openai"
                )
            return OpenAI(
                api_key="ollama",
                base_url=base_url or self.OLLAMA_BASE_URL,
            )

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> LLMResponse:
        """
        Send messages to the LLM and return a normalised LLMResponse.
        Works identically across all three providers.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"raw": tc.function.arguments}

            return LLMResponse(
                content=None,
                tool_name=tc.function.name,
                tool_args=args,
                raw=response,
            )

        return LLMResponse(
            content=msg.content,
            tool_name=None,
            tool_args=None,
            raw=response,
        )

    def __repr__(self):
        return f"<LLMWrapper provider={self.provider} model={self.model}>"