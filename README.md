# Multi-Agent System — Phase 1: Single Agent Foundation

A clean, from-scratch ReAct agent with tools, memory, structured logging, and a full eval suite.
Built as Phase 1 of a multi-phase multi-agent system project.

## Project structure

```
multi_agent_phase1/
├── main.py              # Entry point — CLI runner
├── agent.py             # Core Agent class (plan → act → observe → reflect)
├── llm.py               # LLM wrapper (OpenAI + Ollama)
├── tools/
│   ├── base.py          # BaseTool abstract class
│   ├── python_repl.py   # Safe Python execution via subprocess
│   └── web_search.py    # DuckDuckGo / Tavily search
├── memory/
│   └── buffer.py        # Rolling conversation buffer
├── utils/
│   └── logger.py        # Structured JSON-lines trace logger
├── tests/
│   ├── test_tools.py    # Tool unit tests (offline)
│   ├── test_memory.py   # Memory unit tests (offline)
│   └── test_agent.py    # Agent integration tests (mock LLM)
├── evals/
│   └── eval_suite.py    # 20-goal eval suite (real LLM)
├── logs/                # trace.jsonl written here
├── .env.example
├── requirements.txt
└── Makefile
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add OPENAI_API_KEY=sk-...

# 3. Run with default eval goals
python main.py

# 4. Run with your own goal
python main.py --goal "Search for what RLHF is and summarise it"

# 5. Use Ollama locally
ollama pull mistral
python main.py --provider ollama --model mistral --goal "What is 17 * 43?"
```

## CLI options

| Flag | Default | Description |
|---|---|---|
| `--goal` | None | Custom goal; runs default suite if omitted |
| `--provider` | openai | `openai` or `ollama` |
| `--model` | gpt-4o-mini | Any model name |
| `--max-iter` | 10 | Max iterations per goal |
| `--no-search` | False | Disable web search tool |
| `--no-repl` | False | Disable Python REPL tool |
| `--save-trace` | False | Write trace to `logs/trace.jsonl` |
| `--quiet` | False | Suppress per-iteration logs |

## Run tests

```bash
# Unit + integration tests (no LLM, no API key needed)
make test

# Full eval suite (requires API key, costs ~$0.05 with gpt-4o-mini)
make eval
```

## Phase 1 definition of done

- [ ] Completes a 3-step task without crashing
- [ ] Tool error doesn't kill the loop
- [ ] Max iteration cap fires correctly
- [ ] Buffer doesn't overflow on long runs
- [ ] Logs are readable in `logs/trace.jsonl`
- [ ] Eval suite: ≥16/17 auto-graded goals pass

## What's coming in Phase 2

- Long-term memory with ChromaDB
- Additional tools (file I/O, calculator)
- Orchestrator agent
- Specialist agents (Researcher, Coder, Writer)
