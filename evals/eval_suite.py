"""
evals/eval_suite.py — Phase 5 eval suite.

Tests both single-agent and orchestrator modes.
Saves results to evals/results.json.

Usage:
  python evals/eval_suite.py
  python evals/eval_suite.py --provider groq --model llama-3.3-70b-versatile
  python evals/eval_suite.py --mode orchestrator
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agent import Agent
from llm import LLMWrapper
from orchestrator import Orchestrator
from tools import PythonREPL, WebSearchTool, CalculatorTool, FileWriteTool


# ── Eval goals ────────────────────────────────────────────────────────────────

SINGLE_AGENT_GOALS = [
    # Computation
    {"id": 1,  "goal": "What is the square root of 1764? Use the calculator.",
               "must_contain": ["42"]},
    {"id": 2,  "goal": "Write Python to check if 97 is prime and print the result.",
               "must_contain": ["prime"]},
    {"id": 3,  "goal": "Use python_repl to write and run Python code that generates and prints the first 10 Fibonacci numbers.",
                "must_contain": ["34"]},
    {"id": 4,  "goal": "How many days are in 17 weeks? Use the calculator.",
               "must_contain": ["119"]},
    {"id": 5,  "goal": "Use Python to reverse the string 'intelligence' and print it.",
               "must_contain": ["ecnegilletni"]},

    # Web search
    {"id": 6,  "goal": "What does Retrieval-Augmented Generation (RAG) stand for in AI?",
               "must_contain": ["retrieval"]},
    {"id": 7,  "goal": "Who created the Python programming language?",
               "must_contain": ["guido"]},
    {"id": 8,  "goal": "What is ChromaDB used for in AI applications?",
               "must_contain": ["vector"]},
    {"id": 9,  "goal": "What year was the Transformer paper 'Attention Is All You Need' published?",
               "must_contain": ["2017"]},
    {"id": 10, "goal": "What is the difference between fine-tuning and prompt engineering in LLMs?",
               "must_contain": []},  # manual

    # Error resilience
    {"id": 11, "goal": "Run this broken code and tell me the error: x = 1 / 0",
               "must_contain": ["zerodivision"]},
    {"id": 12,"goal": "Use python_repl to run this code and tell me the exact error type: print('hello'",
                "must_contain": ["syntaxerror"]},
    {"id": 13,
 "goal": "Use python_repl to run: import nonexistentlib999 — tell me the exact Python exception class name.",
 "must_contain": ["modulenotfounderror"]},

    # Multi-step
    {"id": 14,
 "goal": "Use web_search to find what FAISS stands for and what company made it.",
 "must_contain": ["facebook", "meta", "similarity"]},
    {"id": 15,
 "goal": "What is gradient descent in machine learning? Explain in 2 sentences.",
 "must_contain": ["gradient", "optim"]},
]

ORCHESTRATOR_GOALS = [
    {"id": 16, "goal": "Research what a transformer neural network is and write a Python class skeleton for one.",
               "must_contain": ["attention", "class"]},
    {"id": 17, "goal": "Research what vector embeddings are and write Python code to compute cosine similarity between two vectors.",
               "must_contain": ["cosine", "def"]},
    {"id": 18, "goal": "Research the difference between LSTM and Transformer architectures.",
               "must_contain": ["lstm", "transformer"]},
    {"id": 19, "goal": "Research what fine-tuning a language model means and write a Python pseudocode example.",
               "must_contain": ["fine-tun", "def"]},
    {"id": 20, "goal": "Research what tokenization is in NLP and write Python code to tokenize a sentence.",
               "must_contain": ["token", "split"]},
]


# ── Eval runner ───────────────────────────────────────────────────────────────

def run_single_agent_evals(
    provider: str, model: str, max_iter: int
) -> list[dict]:
    llm = LLMWrapper(provider=provider, model=model)
    agent = Agent(
        llm=llm,
        tools=[PythonREPL(), WebSearchTool(), CalculatorTool(), FileWriteTool()],
        max_iterations=max_iter,
        verbose=False,
        save_trace=True,
    )

    results = []
    print(f"\n── Single-agent evals ({len(SINGLE_AGENT_GOALS)} goals) ──")

    for case in SINGLE_AGENT_GOALS:
        start = time.time()
        try:
            answer = agent.run(case["goal"])
            elapsed = round(time.time() - start, 2)
            crashed = False
        except Exception as e:
            answer = f"CRASHED: {e}"
            elapsed = round(time.time() - start, 2)
            crashed = True

        must = case.get("must_contain", [])
        if must:
            passed = all(kw.lower() in answer.lower() for kw in must)
        else:
            passed = None  # manual review

        status = "PASS" if passed is True else ("MANUAL" if passed is None else "FAIL")
        print(f"  [{status:6}] #{case['id']:2} ({agent.iteration_count} iter, {elapsed}s): {case['goal'][:55]}")
        if passed is False:
            print(f"           Answer: {answer[:100]}")

        results.append({
            "id": case["id"],
            "mode": "single",
            "goal": case["goal"],
            "answer": answer,
            "passed": passed,
            "crashed": crashed,
            "iterations": agent.iteration_count,
            "elapsed_s": elapsed,
        })

    return results


def run_orchestrator_evals(
    provider: str, model: str, agent_model: str
) -> list[dict]:
    llm = LLMWrapper(provider=provider, model=model)
    agent_llm = LLMWrapper(provider=provider, model=agent_model)
    orch = Orchestrator(
        llm=llm,
        agent_llm=agent_llm,
        verbose=False,
        save_trace=True,
        max_retries=1,
    )

    results = []
    print(f"\n── Orchestrator evals ({len(ORCHESTRATOR_GOALS)} goals) ──")

    for case in ORCHESTRATOR_GOALS:
        start = time.time()
        try:
            answer = orch.run(case["goal"])
            elapsed = round(time.time() - start, 2)
            crashed = False
        except Exception as e:
            answer = f"CRASHED: {e}"
            elapsed = round(time.time() - start, 2)
            crashed = True

        must = case.get("must_contain", [])
        if must:
            passed = all(kw.lower() in answer.lower() for kw in must)
        else:
            passed = None

        status = "PASS" if passed is True else ("MANUAL" if passed is None else "FAIL")
        print(f"  [{status:6}] #{case['id']:2} ({elapsed}s): {case['goal'][:55]}")
        if passed is False:
            print(f"           Answer: {answer[:100]}")

        results.append({
            "id": case["id"],
            "mode": "orchestrator",
            "goal": case["goal"],
            "answer": answer,
            "passed": passed,
            "crashed": crashed,
            "elapsed_s": elapsed,
        })

    return results


def print_summary(results: list[dict]):
    auto = [r for r in results if r["passed"] is not None]
    manual = [r for r in results if r["passed"] is None]
    passed = sum(1 for r in auto if r["passed"])
    crashed = sum(1 for r in results if r["crashed"])
    avg_elapsed = sum(r["elapsed_s"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"EVAL RESULTS")
    print(f"{'='*60}")
    print(f"Auto-graded : {passed}/{len(auto)} passed")
    print(f"Manual review needed: {len(manual)} goals")
    print(f"Crash rate  : {crashed}/{len(results)}")
    print(f"Avg latency : {avg_elapsed:.1f}s per goal")

    single = [r for r in results if r["mode"] == "single"]
    orch = [r for r in results if r["mode"] == "orchestrator"]
    if single:
        s_auto = [r for r in single if r["passed"] is not None]
        s_pass = sum(1 for r in s_auto if r["passed"])
        print(f"\nSingle-agent: {s_pass}/{len(s_auto)} auto-graded passed")
    if orch:
        o_auto = [r for r in orch if r["passed"] is not None]
        o_pass = sum(1 for r in o_auto if r["passed"])
        print(f"Orchestrator: {o_pass}/{len(o_auto)} auto-graded passed")


def main():
    p = argparse.ArgumentParser(description="Phase 5 Eval Suite")
    p.add_argument("--provider",    default="groq",    choices=["groq", "openai", "ollama"])
    p.add_argument("--model",       default="llama-3.3-70b-versatile")
    p.add_argument("--agent-model", default="llama-3.1-8b-instant")
    p.add_argument("--max-iter",    type=int, default=8)
    p.add_argument("--mode",        default="both", choices=["single", "orchestrator", "both"])
    args = p.parse_args()

    print(f"Provider : {args.provider}")
    print(f"Model    : {args.model}")
    print(f"Mode     : {args.mode}")

    all_results = []

    if args.mode in ("single", "both"):
        all_results += run_single_agent_evals(args.provider, args.model, args.max_iter)

    if args.mode in ("orchestrator", "both"):
        all_results += run_orchestrator_evals(args.provider, args.model, args.agent_model)

    print_summary(all_results)

    out = Path("evals/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()