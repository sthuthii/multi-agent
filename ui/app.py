"""
ui/app.py — Gradio frontend for the multi-agent system.

Run locally:
  python ui/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from dotenv import load_dotenv
load_dotenv()

from llm import LLMWrapper
from agent import Agent
from orchestrator import Orchestrator
from tools import PythonREPL, WebSearchTool, CalculatorTool, FileWriteTool


EXAMPLE_GOALS = [
    "What is gradient descent in machine learning? Explain in 2 sentences.",
    "Use python_repl to generate the first 10 Fibonacci numbers.",
    "Search for what ChromaDB is used for in AI applications.",
    "What is the square root of 1764? Use the calculator.",
    "Research transformer self-attention and write a Python implementation.",
]


def run_agent(
    goal: str,
    mode: str,
    provider: str,
    model: str,
    agent_model: str,
    max_iter: int,
    critic_threshold: int,
) -> tuple[str, str]:
    if not goal.strip():
        return "Please enter a goal.", ""

    log_lines = []

    try:
        llm = LLMWrapper(provider=provider, model=model or None)
    except EnvironmentError as e:
        return f"Error: {e}", ""

    if mode == "Orchestrator (multi-agent)":
        try:
            agent_llm = LLMWrapper(provider=provider, model=agent_model or None)
        except EnvironmentError as e:
            return f"Error: {e}", ""

        runner = Orchestrator(
            llm=llm,
            agent_llm=agent_llm,
            verbose=True,
            save_trace=True,
            critic_threshold=int(critic_threshold),
            max_retries=2,
        )
        answer = runner.run(goal)
        log_lines.append(f"run_id: {runner.run_id}")
        log_lines.append(f"mode: orchestrator")

    else:
        agent = Agent(
            llm=llm,
            tools=[PythonREPL(), WebSearchTool(), CalculatorTool(), FileWriteTool()],
            max_iterations=int(max_iter),
            verbose=True,
            save_trace=True,
        )
        answer = agent.run(goal)
        log_lines.append(f"run_id: {agent.run_id}")
        log_lines.append(f"iterations: {agent.iteration_count}")
        log_lines.append(f"mode: single-agent")

    return answer, "\n".join(log_lines)


# ── UI layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Multi-Agent System") as demo:

    gr.Markdown("""
    # 🤖 Multi-Agent System
    **Phase 6 Demo** — Single agent or multi-agent orchestrator with Critic loop
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
            label="Max iterations (single agent)",
        )
        critic_threshold = gr.Slider(
            minimum=1, maximum=10, value=7, step=1,
            label="Critic threshold (orchestrator)",
        )

    run_btn = gr.Button("▶ Run", variant="primary", size="lg")

    with gr.Row():
        answer_box = gr.Textbox(
            label="Answer",
            lines=15,
        )
        log_box = gr.Textbox(
            label="Run info",
            lines=15,
        )

    run_btn.click(
        fn=run_agent,
        inputs=[goal_input, mode, provider, model, agent_model, max_iter, critic_threshold],
        outputs=[answer_box, log_box],
    )


if __name__ == "__main__":
    demo.launch(
        share=False,
        server_port=7860,
        theme=gr.themes.Soft(),   # moved here in Gradio 6.0
    )