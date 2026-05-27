# Multi-Agent AI System

A production-grade multi-agent system built from scratch — no AutoGen, no CrewAI.

## Architecture

```
User Goal
    ↓
Orchestrator  (decomposes → routes → tracks)
    ↓              ↓               ↓
Researcher      Coder          Writer
(web search)  (REPL + files)  (synthesis)
    ↓
Critic  (scores 1-10, triggers reruns if < 7)
    ↓
Final Output + Trace Log
```

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Groq (free) / Ollama (local) / OpenAI |
| Agent framework | Custom from scratch |
| Tools | Web search, Python REPL, Calculator, File write |
| Long-term memory | ChromaDB + sentence-transformers |
| API | FastAPI |
| UI | Gradio |
| Testing | pytest (35+ tests) |

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API key
cp .env.example .env
# Add GROQ_API_KEY=gsk_... (free at console.groq.com)

# 3. Run single agent
python main.py --goal "What is gradient descent?"

# 4. Run orchestrator
python main.py --orchestrator --model llama-3.3-70b-versatile \
  --goal "Research RAG and write a Python implementation"

# 5. Launch UI
python ui/app.py
# Open http://localhost:7860

# 6. Run API
uvicorn api.server:app --reload --port 8000
# Open http://localhost:8000/docs

# 7. Run tests
pytest tests/ -v

# 8. Run evals
python evals/eval_suite.py --mode single --model llama-3.1-8b-instant
```

## CLI flags

| Flag | Description |
|---|---|
| `--goal` | Goal for the agent |
| `--orchestrator` | Use multi-agent mode |
| `--provider` | groq / openai / ollama |
| `--model` | Model name |
| `--agent-model` | Sub-agent model (orchestrator) |
| `--memory` | Enable long-term memory |
| `--save-trace` | Write trace to logs/trace.jsonl |
| `--trace` | Replay all run traces |
| `--critic-threshold` | Min score to accept (default: 7) |
| `--max-retries` | Critic retry attempts (default: 2) |

## Eval results

```
Single-agent (llama-3.1-8b-instant, 15 goals):
  Auto-graded : 14/14 (100%)
  Crash rate  : 0%
  Avg latency : 21.7s/goal
  Avg iters   : 2.1
```

## Deploy to HuggingFace Spaces

1. Create a new Space → SDK: Gradio
2. Push this repo to the Space
3. Add `GROQ_API_KEY` in Space Settings → Secrets
4. App goes live at `huggingface.co/spaces/yourusername/multi-agent`

## Project structure

```
multi_agent/
├── main.py              # CLI entry point
├── agent.py             # Core ReAct loop
├── orchestrator.py      # Multi-agent coordinator
├── llm.py               # LLM wrapper (Groq/OpenAI/Ollama)
├── agents/
│   ├── researcher.py    # Web search specialist
│   ├── coder.py         # Code execution specialist
│   ├── writer.py        # Synthesis specialist
│   └── critic.py        # Quality evaluator + retry trigger
├── tools/               # Calculator, REPL, search, file write
├── memory/              # ChromaDB long-term memory
├── observability/       # Trace replay + analytics
├── api/                 # FastAPI backend
├── ui/                  # Gradio frontend
├── tests/               # 35+ unit + integration tests
└── evals/               # 20-goal eval suite
```