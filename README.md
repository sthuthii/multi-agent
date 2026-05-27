# Multi-Agent AI System

A production-grade multi-agent LLM system built from scratch — no AutoGen, no CrewAI, no LangChain abstractions. Every component is implemented directly, demonstrating deep understanding of how multi-agent systems actually work.

---

## What it does

A user gives a high-level goal. A team of specialised AI agents autonomously decomposes it, executes subtasks using real tools, critiques the output, and delivers a final result with a full audit trail.

```
User Goal
    ↓
Orchestrator  ← decomposes goal into subtasks, routes to agents
    ↓              ↓               ↓
Researcher      Coder          Writer
(web search)  (REPL + files)  (synthesis)
    ↓
Critic  ← scores 1-10, triggers reruns if score < 7
    ↓
Final Output + Trace Log
```

---

## Architecture

### Agents

| Agent | Role | Tools |
|---|---|---|
| Orchestrator | Decomposes goals, routes subtasks, tracks state | None (pure reasoning) |
| Researcher | Finds and summarises information | Web search |
| Coder | Writes and executes Python code | Python REPL, Calculator, File write |
| Writer | Synthesises findings into final output | None |
| Critic | Scores output 1-10, triggers retry if < 7 | None |

### Agent loop (ReAct pattern)

Every agent runs the same underlying loop:

```
PLAN  → LLM decides: call a tool or produce final answer
ACT   → execute tool call, catch errors
OBSERVE → append tool result to context
REFLECT → decide: done or continue
```

### Tech stack

| Layer | Technology |
|---|---|
| LLM | Groq (free) / Ollama (local) / OpenAI |
| Agent framework | Custom from scratch |
| Tool calling | OpenAI function calling with Pydantic schemas |
| Web search | DDGS (free, no key) / Tavily (optional) |
| Code execution | Python subprocess sandbox with timeout |
| Short-term memory | Rolling conversation buffer with compression |
| Long-term memory | ChromaDB + sentence-transformers |
| Observability | Structured JSON-lines trace logger |
| API | FastAPI |
| UI | Gradio (mounted inside FastAPI) |
| Deployment | Docker + HuggingFace Spaces |
| Testing | pytest (35+ tests) |

---

## Eval results

```
Single-agent (llama-3.1-8b-instant, Groq, 15 goals):
  Auto-graded  : 14/14 (100%)
  Manual review: 1/15 (qualitative — passed on review)
  Crash rate   : 0/15 (0%)
  Avg latency  : 21.7s/goal
  Avg iters    : 2.1
```

---

## Quick start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd multi_agent_phase1
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
```

Edit `.env` and add:
```
GROQ_API_KEY=gsk_...
```

Get a free key at [console.groq.com](https://console.groq.com) — no credit card required.

### 3. Run

```bash
# Single agent
python main.py --goal "What is gradient descent in machine learning?"

# Multi-agent orchestrator
python main.py --orchestrator \
  --model llama-3.3-70b-versatile \
  --goal "Research RAG and write a Python implementation"

# Launch full UI + API (recommended)
uvicorn api.server:app --reload --port 8000
# → http://localhost:8000        (Gradio UI)
# → http://localhost:8000/docs   (API explorer)
```

### 4. Run with Docker

```bash
docker compose up --build
# → http://localhost:8000
```

---

## CLI reference

```bash
python main.py [options]
```

| Flag | Default | Description |
|---|---|---|
| `--goal` | — | Goal for the agent |
| `--orchestrator` | False | Use multi-agent mode |
| `--provider` | groq | `groq` / `openai` / `ollama` |
| `--model` | llama-3.1-8b-instant | LLM model name |
| `--agent-model` | llama-3.1-8b-instant | Sub-agent model (orchestrator) |
| `--memory` | False | Enable long-term ChromaDB memory |
| `--clear-memory` | False | Wipe memory before run |
| `--save-trace` | False | Write trace to `logs/trace.jsonl` |
| `--trace` | False | Replay all run traces |
| `--max-iter` | 10 | Max agent iterations |
| `--critic-threshold` | 7 | Min critic score to accept |
| `--max-retries` | 2 | Critic retry attempts |
| `--quiet` | False | Suppress iteration logs |
| `--list-models` | False | Print available models |

---

## Model recommendations

| Use case | Model | Provider |
|---|---|---|
| Development / testing | `llama-3.1-8b-instant` | Groq (free) |
| Orchestrator decomposition | `llama-3.3-70b-versatile` | Groq (free) |
| Local / offline | `mistral` or `qwen2.5` | Ollama |
| Best reliability | `gpt-4o-mini` | OpenAI |

---

## Project structure

```
multi_agent/
├── main.py                   # CLI entry point
├── agent.py                  # Core ReAct agent loop
├── orchestrator.py           # Multi-agent coordinator + critic loop
├── llm.py                    # LLM wrapper (Groq / OpenAI / Ollama)
│
├── agents/
│   ├── base_agent.py         # Specialist agent base class
│   ├── researcher.py         # Web search specialist
│   ├── coder.py              # Code execution specialist
│   ├── writer.py             # Synthesis specialist
│   └── critic.py             # Quality evaluator + retry trigger
│
├── tools/
│   ├── base.py               # BaseTool abstract class
│   ├── python_repl.py        # Subprocess sandbox (10s timeout)
│   ├── web_search.py         # DDGS + Tavily
│   ├── calculator.py         # Safe AST math evaluator
│   └── file_write.py         # Sandboxed file output
│
├── memory/
│   ├── buffer.py             # Rolling buffer with compression
│   └── chroma.py             # ChromaDB long-term memory
│
├── observability/
│   └── dashboard.py          # Trace replay + run analytics
│
├── api/
│   └── server.py             # FastAPI + Gradio (single server)
│
├── ui/
│   └── app.py                # Gradio standalone UI
│
├── tests/                    # 35+ unit + integration tests
│   ├── test_tools.py
│   ├── test_memory.py
│   ├── test_agent.py
│   ├── test_orchestrator.py
│   ├── test_critic.py
│   └── test_observability.py
│
├── evals/
│   └── eval_suite.py         # 20-goal eval suite
│
├── logs/                     # trace.jsonl (gitignored)
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

---

## Testing

```bash
# All tests (no API key needed — uses mock LLM)
make test
# or
pytest tests/ -v

# Specific test file
pytest tests/test_orchestrator.py -v

# Full eval suite (requires API key, ~$0.00 on Groq free tier)
make eval
# or
python evals/eval_suite.py --mode single --model llama-3.1-8b-instant
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/run` | Run a goal |
| `GET` | `/traces` | List all run summaries |
| `GET` | `/trace/{run_id}` | Get events for a specific run |
| `GET` | `/docs` | Swagger UI |

### Example request

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "What is the square root of 1764?",
    "mode": "single",
    "provider": "groq",
    "model": "llama-3.1-8b-instant"
  }'
```

---

## Deployment

### HuggingFace Spaces (recommended — free)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Gradio**
   - Name: `multi-agent-system`

2. Add your API key in Space Settings → Secrets:
   - Key: `GROQ_API_KEY`
   - Value: `gsk_...`

3. Add `app.py` at root (HuggingFace looks for this file):
   ```python
   # app.py
   from ui.app import demo
   demo.launch()
   ```

4. Push your code:
   ```bash
   git init
   git add .
   git commit -m "multi-agent system"
   git remote add space https://huggingface.co/spaces/yourusername/multi-agent-system
   git push space main
   ```

5. Your app goes live at:
   `https://huggingface.co/spaces/yourusername/multi-agent-system`

### Railway (full FastAPI + Gradio)

1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Connect your repo
3. Set environment variable: `GROQ_API_KEY=gsk_...`
4. Railway auto-detects the `Dockerfile` and deploys
5. Your app gets a public URL with both UI and API

### Docker (self-hosted)

```bash
# Build and run
docker compose up --build -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

---

## Known limitations

- **Cross-session memory recall** is imperfect with small hosted models (Llama 3.1 8B). The retrieval pipeline works correctly (verified at 0.67 cosine similarity) but hosted models resist overriding their "no prior history" training instinct. Solvable with fine-tuning or a dedicated memory model.

- **Groq free tier latency** averages 21.7s/goal due to rate limiting. On a paid tier or local Ollama this drops to 3-5s.

- **Llama 3.1 8B tool calling** is occasionally unreliable — it sometimes answers from knowledge instead of calling tools. The 70B model or GPT-4o-mini are significantly more reliable for tool-heavy tasks.

---

## What was built (phase by phase)

| Phase | What | Key outcome |
|---|---|---|
| 1 | Single agent, ReAct loop, tools | Working plan→act→observe→reflect loop |
| 2 | ChromaDB memory, calculator, file write | Persistent memory across sessions |
| 3 | Orchestrator, Researcher, Coder, Writer | Multi-agent goal decomposition |
| 4 | Critic loop, structured scoring, retry | Quality-controlled output |
| 5 | Trace logger, eval suite | 14/14 auto-graded, 0% crash rate |
| 6 | FastAPI, Gradio UI, Docker, deploy | Production-ready, publicly accessible |