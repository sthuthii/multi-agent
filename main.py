"""
main.py — Phase 1 entry point.

Default provider is now Groq (free, fast, no credit card needed).
Get a free key at https://console.groq.com

Usage:
  python main.py                                      # Groq, default model
  python main.py --goal "Your goal here"
  python main.py --provider groq   --model llama-3.3-70b-versatile
  python main.py --provider ollama --model mistral
  python main.py --provider openai --model gpt-4o-mini
  python main.py --list-models                        # show all model options
  python main.py --save-trace                         # write logs/trace.jsonl
  python main.py --quiet                              # suppress iteration logs
"""

import argparse
import sys

from dotenv import load_dotenv

from agent import Agent
from llm import LLMWrapper, PROVIDER_DEFAULTS, GROQ_MODELS
from tools import PythonREPL, WebSearchTool
from tools.base import BaseTool

load_dotenv()


# ── Default eval goals ────────────────────────────────────────────────────────

DEFAULT_GOALS = [
    "What is the square root of 1764? Verify by running Python code.",
    "Search for what RAG stands for in the context of LLMs, then summarise it in 2 sentences.",
    "Write and run Python code that generates the first 10 Fibonacci numbers.",
    "Search for the creator of the Python programming language.",
    "Run this broken code and tell me the exact error: x = 1 / 0",
]

# ── Model reference ───────────────────────────────────────────────────────────

MODEL_REFERENCE = {
    "groq": [
        ("llama-3.1-8b-instant",    "Fastest — best for development & testing"),
        ("llama-3.3-70b-versatile", "Best quality — use for evals & demos"),
        ("mixtral-8x7b-32768",      "Long context (32k tokens)"),
        ("gemma2-9b-it",            "Lightweight alternative"),
    ],
    "ollama": [
        ("mistral",     "7B — good tool calling, fast"),
        ("qwen2.5",     "Strong at coding tasks"),
        ("llama3.2",    "3B — very fast, lighter reasoning"),
        ("llama3.1",    "8B — closest to GPT-4o-mini behaviour"),
    ],
    "openai": [
        ("gpt-4o-mini", "Fast, cheap, reliable tool calling"),
        ("gpt-4o",      "Best quality, higher cost"),
    ],
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 — Single Agent (default provider: Groq, free)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--goal", type=str, default=None,
        help="Goal for the agent. Runs default eval suite if omitted.",
    )
    p.add_argument(
        "--provider", type=str, default="groq",
        choices=["groq", "openai", "ollama"],
        help="LLM provider (default: groq — free at console.groq.com)",
    )
    p.add_argument(
        "--model", type=str, default=None,
        help=(
            "Model name. Defaults per provider:\n"
            "  groq:   llama-3.1-8b-instant\n"
            "  ollama: mistral\n"
            "  openai: gpt-4o-mini\n"
            "Run --list-models to see all options."
        ),
    )
    p.add_argument(
        "--list-models", action="store_true",
        help="Print available models for each provider and exit.",
    )
    p.add_argument(
        "--max-iter", type=int, default=10,
        help="Max agent iterations per goal (default: 10)",
    )
    p.add_argument(
        "--no-search", action="store_true",
        help="Disable web search tool",
    )
    p.add_argument(
        "--no-repl", action="store_true",
        help="Disable Python REPL tool",
    )
    p.add_argument(
        "--save-trace", action="store_true",
        help="Save run trace to logs/trace.jsonl",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-iteration logs",
    )
    return p.parse_args()


def list_models():
    print("\nAvailable models by provider:\n")
    for provider, models in MODEL_REFERENCE.items():
        free_tag = "  [FREE]" if provider in ("groq", "ollama") else ""
        print(f"  {provider}{free_tag}")
        for name, desc in models:
            default = " (default)" if name == PROVIDER_DEFAULTS.get(provider) else ""
            print(f"    --model {name:<32} {desc}{default}")
        print()
    print("Groq signup (free, no credit card): https://console.groq.com")
    print("Ollama install (local):             https://ollama.com\n")


def build_tools(args: argparse.Namespace) -> list[BaseTool]:
    tools: list[BaseTool] = []
    if not args.no_repl:
        tools.append(PythonREPL())
    if not args.no_search:
        tools.append(WebSearchTool())
    return tools


def run_goals(agent: Agent, goals: list[str]):
    for idx, goal in enumerate(goals, 1):
        print(f"\n[Goal {idx}/{len(goals)}]")
        answer = agent.run(goal)
        print(f"\nANSWER:\n{answer}")
        print(f"(iterations used: {agent.iteration_count})")

    print(f"\n{'='*60}")
    print(f"Completed {len(goals)} goal(s).")
    if agent.save_trace:
        print("Trace saved → logs/trace.jsonl")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.list_models:
        list_models()
        sys.exit(0)

    tools = build_tools(args)
    if not tools:
        print("Error: at least one tool must be enabled.")
        sys.exit(1)

    try:
        llm = LLMWrapper(
            provider=args.provider,
            model=args.model,  # None → uses provider default
        )
    except (EnvironmentError, ImportError) as e:
        print(f"\nError: {e}\n")
        sys.exit(1)

    print(f"\nUsing: {llm}")

    agent = Agent(
        llm=llm,
        tools=tools,
        max_iterations=args.max_iter,
        verbose=not args.quiet,
        save_trace=args.save_trace,
    )

    goals = [args.goal] if args.goal else DEFAULT_GOALS
    run_goals(agent, goals)


if __name__ == "__main__":
    main()