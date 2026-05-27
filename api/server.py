"""
api/server.py — FastAPI backend with Gradio UI mounted at root.

Run:
  uvicorn api.server:app --reload --port 8000

Then visit:
  http://localhost:8000/       ← Gradio UI
  http://localhost:8000/docs   ← API docs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

from llm import LLMWrapper
from agent import Agent
from orchestrator import Orchestrator
from tools import PythonREPL, WebSearchTool, CalculatorTool, FileWriteTool
from observability.dashboard import load_runs, summarise_run

app = FastAPI(
    title="Multi-Agent System",
    description="Multi-agent system with Critic loop",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    goal: str
    mode: str = "single"
    provider: str = "groq"
    model: Optional[str] = None
    agent_model: Optional[str] = None
    max_iterations: int = 10
    critic_threshold: int = 7
    max_retries: int = 2
    save_trace: bool = True


class RunResponse(BaseModel):
    run_id: str
    goal: str
    answer: str
    mode: str
    iterations: Optional[int] = None


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/run", response_model=RunResponse)
def run_goal(req: RunRequest):
    try:
        llm = LLMWrapper(provider=req.provider, model=req.model)
    except (EnvironmentError, ImportError) as e:
        raise HTTPException(status_code=500, detail=str(e))

    if req.mode == "orchestrator":
        agent_llm = LLMWrapper(
            provider=req.provider,
            model=req.agent_model or "llama-3.1-8b-instant",
        )
        runner = Orchestrator(
            llm=llm,
            agent_llm=agent_llm,
            verbose=False,
            save_trace=req.save_trace,
            critic_threshold=req.critic_threshold,
            max_retries=req.max_retries,
        )
        answer = runner.run(req.goal)
        return RunResponse(
            run_id=runner.run_id,
            goal=req.goal,
            answer=answer,
            mode="orchestrator",
        )
    else:
        agent = Agent(
            llm=llm,
            tools=[PythonREPL(), WebSearchTool(), CalculatorTool(), FileWriteTool()],
            max_iterations=req.max_iterations,
            verbose=False,
            save_trace=req.save_trace,
        )
        answer = agent.run(req.goal)
        return RunResponse(
            run_id=agent.run_id,
            goal=req.goal,
            answer=answer,
            mode="single",
            iterations=agent.iteration_count,
        )


@app.get("/traces")
def list_traces():
    runs = load_runs()
    summaries = [summarise_run(events) for events in runs.values()]
    return {"count": len(summaries), "runs": summaries}


@app.get("/trace/{run_id}")
def get_trace(run_id: str):
    runs = load_runs()
    if run_id not in runs:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return {"run_id": run_id, "events": runs[run_id]}


# ── Gradio UI (calls FastAPI /run endpoint) ───────────────────────────────────

import httpx

EXAMPLE_GOALS = [
    "What is gradient descent? Explain in 2 sentences.",
    "Use python_repl to generate the first 10 Fibonacci numbers.",
    "Search for what ChromaDB is used for in AI.",
    "What is the square root of 1764? Use the calculator.",
    "Research transformer self-attention and write a Python implementation.",
]


def run_from_ui(
    goal: str,
    mode: str,
    provider: str,
    model: str,
    agent_model: str,
    max_iter: int,
    critic_threshold: int,
) -> tuple[str, str]:
    """Calls the FastAPI /run endpoint and returns answer + run info."""
    if not goal.strip():
        return "Please enter a goal.", ""

    payload = {
        "goal": goal,
        "mode": "orchestrator" if "Orchestrator" in mode else "single",
        "provider": provider,
        "model": model or None,
        "agent_model": agent_model or None,
        "max_iterations": int(max_iter),
        "critic_threshold": int(critic_threshold),
        "save_trace": True,
    }

    try:
        # Call our own FastAPI endpoint
        response = httpx.post(
            "http://localhost:8000/run",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        answer = data.get("answer", "No answer returned.")
        info_lines = [
            f"run_id    : {data.get('run_id')}",
            f"mode      : {data.get('mode')}",
            f"iterations: {data.get('iterations', 'N/A')}",
        ]
        return answer, "\n".join(info_lines)

    except httpx.TimeoutException:
        return "Request timed out. Try a simpler goal or increase timeout.", ""
    except httpx.HTTPStatusError as e:
        return f"API error {e.response.status_code}: {e.response.text}", ""
    except Exception as e:
        return f"Error: {e}", ""


gradio_app = gr.Blocks(title="Multi-Agent System")

with gradio_app:
    gr.Markdown("""
    # Multi-Agent System
    **Phase 6** — Single agent or multi-agent orchestrator with Critic loop
    """)

    with gr.Row():
        with gr.Column(scale=2):
            goal_input = gr.Textbox(
                label="Goal",
                placeholder="What do you want the agent to do?",
                lines=3,
            )
            gr.Examples(
                examples=EXAMPLE_GOALS,
                inputs=goal_input,
                label="Example goals",
            )

        with gr.Column(scale=1):
            mode = gr.Radio(
                choices=["Single agent", "Orchestrator (multi-agent)"],
                value="Single agent",
                label="Mode",
            )
            provider = gr.Dropdown(
                choices=["groq", "openai", "ollama"],
                value="groq",
                label="Provider",
            )
            model = gr.Textbox(
                value="llama-3.1-8b-instant",
                label="Model",
            )
            agent_model = gr.Textbox(
                value="llama-3.1-8b-instant",
                label="Sub-agent model (orchestrator only)",
            )

    with gr.Row():
        max_iter = gr.Slider(
            minimum=1, maximum=20, value=10, step=1,
            label="Max iterations",
        )
        critic_threshold = gr.Slider(
            minimum=1, maximum=10, value=7, step=1,
            label="Critic threshold",
        )

    run_btn = gr.Button("▶ Run", variant="primary", size="lg")

    with gr.Row():
        answer_box = gr.Textbox(label="Answer", lines=15)
        log_box = gr.Textbox(label="Run info", lines=15)

    run_btn.click(
        fn=run_from_ui,
        inputs=[goal_input, mode, provider, model, agent_model, max_iter, critic_threshold],
        outputs=[answer_box, log_box],
    )


# Mount Gradio at root — single server, single port
app = gr.mount_gradio_app(app, gradio_app, path="/")