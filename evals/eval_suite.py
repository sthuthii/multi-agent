"""
evals/eval_suite.py — Phase 1 eval suite.
Runs 20 goals against a real LLM and reports pass/fail + metrics.
Results are saved to evals/results.json.

Usage:
  python evals/eval_suite.py
  python evals/eval_suite.py --provider ollama --model mistral
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agent import Agent
from llm import LLMWrapper
from tools import PythonREPL, WebSearchTool


EVAL_GOALS = [
    # ── Computation (REPL) ─────────────────────────────────────────────────
    {"id": 1,  "goal": "What is the square root of 1764? Use Python to verify.",
               "must_contain": ["42"]},
    {"id": 2,  "goal": "Write Python to check if 97 is a prime number and print the result.",
               "must_contain": ["prime", "97"]},
    {"id": 3,  "goal": "Generate the first 10 Fibonacci numbers using Python and print them.",
               "must_contain": ["55"]},  # 10th Fibonacci number
    {"id": 4,  "goal": "How many days are in 17 weeks? Compute with Python.",
               "must_contain": ["119"]},
    {"id": 5,  "goal": "Use Python to reverse the string 'intelligence' and print it.",
               "must_contain": ["ecnegilletni"]},

    # ── Web search ─────────────────────────────────────────────────────────
    {"id": 6,  "goal": "What does RAG stand for in the context of LLMs?",
               "must_contain": ["retrieval"]},
    {"id": 7,  "goal": "Who created the Python programming language?",
               "must_contain": ["Guido"]},
    {"id": 8,  "goal": "What is the Hugging Face Transformers library used for?",
               "must_contain": []},  # manual review
    {"id": 9,  "goal": "What is ChromaDB?",
               "must_contain": ["vector"]},
    {"id": 10, "goal": "What year was the Transformer architecture introduced in 'Attention Is All You Need'?",
               "must_contain": ["2017"]},

    # ── Multi-step (search + compute) ──────────────────────────────────────
    {"id": 11, "goal": "Search for what LangChain is, then write Python to count the characters in the word 'LangChain'.",
               "must_contain": ["9"]},
    {"id": 12, "goal": "Find out what DPO stands for in LLM fine-tuning and summarise it in 1 sentence.",
               "must_contain": ["preference"]},

    # ── Error resilience ───────────────────────────────────────────────────
    {"id": 13, "goal": "Run this broken code and tell me the exact error type: x = 1 / 0",
               "must_contain": ["ZeroDivision"]},
    {"id": 14, "goal": "Run this broken code and tell me the error: print('hello'",
               "must_contain": ["SyntaxError", "EOF", "parenthes"]},
    {"id": 15, "goal": "Try to import a library called 'nonexistentlib123' and tell me what happens.",
               "must_contain": ["ModuleNotFoundError", "No module"]},

    # ── Factual (direct answer, no tool needed) ────────────────────────────
    {"id": 16, "goal": "What does API stand for?",
               "must_contain": ["Application Programming Interface"]},
    {"id": 17, "goal": "What is the time complexity of binary search?",
               "must_contain": ["O(log"]},

    # ── Compound ───────────────────────────────────────────────────────────
    {"id": 18, "goal": "Find the latest version of the 'requests' Python library and then write code that prints the version string.",
               "must_contain": []},  # manual review
    {"id": 19, "goal": "Search for what FAISS is and then compute: if FAISS can search 1 million vectors in 10ms, how many per second? Use Python.",
               "must_contain": ["100000000", "100,000,000", "1e8", "100M"]},
    {"id": 20, "goal": "Explain what a vector embedding is in 2 sentences.",
               "must_contain": []},  # manual review
]


def run_evals(provider: str = "openai", model: str = "gpt-4o-mini", max_iter: int = 8):
    llm = LLMWrapper(provider=provider, model=model)
    agent = Agent(
        llm=llm,
        tools=[PythonREPL(), WebSearchTool()],
        max_iterations=max_iter,
        verbose=False,
        save_trace=True,
    )

    results = []
    passed = 0
    total_iterations = 0

    print(f"\nRunning Phase 1 eval suite ({len(EVAL_GOALS)} goals)")
    print(f"Provider: {provider}  Model: {model}  max_iter: {max_iter}")
    print("=" * 60)

    for case in EVAL_GOALS:
        start = time.time()
        try:
            answer = agent.run(case["goal"])
            elapsed = round(time.time() - start, 2)
            iterations = agent.iteration_count
            crashed = False
        except Exception as e:
            answer = f"CRASHED: {e}"
            elapsed = round(time.time() - start, 2)
            iterations = agent.iteration_count
            crashed = True

        must_contain = case.get("must_contain", [])
        if must_contain:
            passed_case = all(kw.lower() in answer.lower() for kw in must_contain)
        else:
            passed_case = None  # manual review required

        total_iterations += iterations
        if passed_case is True:
            passed += 1

        status = "PASS" if passed_case is True else ("MANUAL" if passed_case is None else "FAIL")
        print(f"[{status:6}] #{case['id']:2} ({iterations} iter, {elapsed}s): {case['goal'][:55]}")
        if passed_case is False:
            print(f"         Answer: {answer[:120]}")

        results.append({
            "id": case["id"],
            "goal": case["goal"],
            "answer": answer,
            "passed": passed_case,
            "crashed": crashed,
            "iterations": iterations,
            "elapsed_s": elapsed,
        })

    auto_graded = [r for r in results if r["passed"] is not None]
    manual = [r for r in results if r["passed"] is None]

    print("\n" + "=" * 60)
    print(f"Auto-graded:  {passed}/{len(auto_graded)} passed")
    print(f"Manual review needed: {len(manual)} goals")
    print(f"Avg iterations/goal: {total_iterations / len(EVAL_GOALS):.1f}")
    print(f"Crash rate: {sum(r['crashed'] for r in results)}/{len(results)}")

    out_path = Path("evals/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="openai", choices=["openai", "ollama"])
    p.add_argument("--model",    default="gpt-4o-mini")
    p.add_argument("--max-iter", type=int, default=8)
    args = p.parse_args()
    run_evals(args.provider, args.model, args.max_iter)
