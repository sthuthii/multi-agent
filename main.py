"""
main.py — Phase 2 entry point.

New in Phase 2:
  --memory          enables long-term ChromaDB memory
  --memory-dir      where ChromaDB persists data (default: .chroma)
  --memory-agent    agent namespace for memory (default: "default")
  --clear-memory    wipe the agent's memory before running
  --no-calc         disable calculator tool
  --no-filewrite    disable file write tool

Usage:
  python main.py                                      # Groq, no memory
  python main.py --memory                             # enable long-term memory
  python main.py --memory --goal "What is RAG?"       # memory + custom goal
  python main.py --provider ollama --model mistral
  python main.py --list-models
"""

import argparse
import sys

from dotenv import load_dotenv

from agent import Agent
from llm import LLMWrapper, PROVIDER_DEFAULTS
from memory.chroma import MemoryManager
from tools import PythonREPL, WebSearchTool, FileWriteTool, CalculatorTool
from tools.base import BaseTool

load_dotenv()


# ── Default eval goals ────────────────────────────────────────────────────────

DEFAULT_GOALS = [
    "What is the square root of 1764? Use the calculator tool.",
    "Search for what RAG stands for in the context of LLMs, then summarise it in 2 sentences.",
    "Write Python code that generates the first 10 Fibonacci numbers, then save the code to a file called fibonacci.py.",
    "Search for the creator of the Python programming language.",
    "Run this broken code and tell me the exact error: x = 1 / 0",
]

MODEL_REFERENCE = {
    "groq": [
        ("llama-3.1-8b-instant",    "Fastest — best for development & testing"),
        ("llama-3.3-70b-versatile", "Best quality — use for evals & demos"),
        ("mixtral-8x7b-32768",      "Long context (32k tokens)"),
        ("gemma2-9b-it",            "Lightweight alternative"),
    ],
    "ollama": [
        ("mistral",  "7B — good tool calling, fast"),
        ("qwen2.5",  "Strong at coding tasks"),
        ("llama3.2", "3B — very fast, lighter reasoning"),
        ("llama3.1", "8B — closest to GPT-4o-mini behaviour"),
    ],
    "openai": [
        ("gpt-4o-mini", "Fast, cheap, reliable tool calling"),
        ("gpt-4o",      "Best quality, higher cost"),
    ],
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 — Memory-aware Single Agent",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Core
    p.add_argument("--goal",     type=str, default=None,
                   help="Goal for the agent. Runs default suite if omitted.")
    p.add_argument("--provider", type=str, default="groq",
                   choices=["groq", "openai", "ollama"],
                   help="LLM provider (default: groq — free at console.groq.com)")
    p.add_argument("--model",    type=str, default=None,
                   help="Model name. Run --list-models to see options.")
    p.add_argument("--list-models", action="store_true",
                   help="Print available models and exit.")
    p.add_argument("--max-iter", type=int, default=10,
                   help="Max agent iterations per goal (default: 10)")

    # Phase 2: memory flags
    p.add_argument("--memory",       action="store_true",
                   help="Enable long-term ChromaDB memory (Phase 2)")
    p.add_argument("--memory-dir",   type=str, default=".chroma",
                   help="Directory for ChromaDB persistence (default: .chroma)")
    p.add_argument("--memory-agent", type=str, default="default",
                   help="Agent namespace for memory (default: 'default')")
    p.add_argument("--clear-memory", action="store_true",
                   help="Wipe agent memory before running")

    # Tool toggles
    p.add_argument("--no-search",    action="store_true", help="Disable web search")
    p.add_argument("--no-repl",      action="store_true", help="Disable Python REPL")
    p.add_argument("--no-calc",      action="store_true", help="Disable calculator")
    p.add_argument("--no-filewrite", action="store_true", help="Disable file write")

    # Output
    p.add_argument("--save-trace", action="store_true",
                   help="Save run trace to logs/trace.jsonl")
    p.add_argument("--quiet",      action="store_true",
                   help="Suppress per-iteration logs")
    return p.parse_args()


def list_models():
    print("\nAvailable models by provider:\n")
    for provider, models in MODEL_REFERENCE.items():
        free_tag = "  [FREE]" if provider in ("groq", "ollama") else ""
        print(f"  {provider}{free_tag}")
        for name, desc in models:
            default = " (default)" if name == PROVIDER_DEFAULTS.get(provider) else ""
            print(f"    --model {name:<34} {desc}{default}")
        print()
    print("Groq signup (free, no credit card): https://console.groq.com")
    print("Ollama install (local):             https://ollama.com\n")


def build_tools(args: argparse.Namespace) -> list[BaseTool]:
    tools: list[BaseTool] = []
    if not args.no_repl:
        tools.append(PythonREPL())
    if not args.no_search:
        tools.append(WebSearchTool())
    if not args.no_calc:
        tools.append(CalculatorTool())
    if not args.no_filewrite:
        tools.append(FileWriteTool())
    return tools


def build_memory(args: argparse.Namespace) -> MemoryManager | None:
    if not args.memory:
        return None

    mem = MemoryManager(
        agent_name=args.memory_agent,
        persist_dir=args.memory_dir,
    )

    if args.clear_memory:
        mem.clear()
        print(f"[memory] Cleared memory for agent '{args.memory_agent}'.")

    print(f"[memory] Long-term memory ON  (namespace={args.memory_agent}, dir={args.memory_dir})")
    return mem


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
    if agent.long_term_memory:
        print(f"Memory entries stored: {agent.long_term_memory.count}")


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
        llm = LLMWrapper(provider=args.provider, model=args.model)
    except (EnvironmentError, ImportError) as e:
        print(f"\nError: {e}\n")
        sys.exit(1)

    long_term_memory = build_memory(args)

    print(f"\nUsing: {llm}")
    print(f"Tools: {[t.name for t in tools]}")

    agent = Agent(
        llm=llm,
        tools=tools,
        max_iterations=args.max_iter,
        verbose=not args.quiet,
        save_trace=args.save_trace,
        long_term_memory=long_term_memory,
    )

    goals = [args.goal] if args.goal else DEFAULT_GOALS
    run_goals(agent, goals)


if __name__ == "__main__":
    main()
