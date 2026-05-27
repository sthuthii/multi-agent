"""
api/server.py — FastAPI backend for the multi-agent system.

Endpoints:
  POST /run          — run a goal (single agent or orchestrator)
  GET  /trace/{run_id} — get trace events for a run
  GET  /traces       — list all run summaries
  GET  /health       — health check

Run locally:
  uvicorn api.server:app --reload --port 8000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from llm import LLMWrapper
from agent import Agent
from orchestrator import Orchestrator
from tools import PythonREPL, WebSearchTool, CalculatorTool, FileWriteTool
from observability.dashboard import load_runs, summarise_run

app = FastAPI(
    title="Multi-Agent System API",
    description="Phase 6 — FastAPI backend for the multi-agent system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    goal: str
    mode: str = "single"           # "single" or "orchestrator"
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


# ── Routes ────────────────────────────────────────────────────────────────────

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