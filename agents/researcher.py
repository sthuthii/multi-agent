"""
agents/researcher.py — Researcher specialist agent.
Finds, retrieves, and summarises information via web search.
"""

from agents.base_agent import BaseSpecialistAgent

RESEARCHER_PROMPT = """You are a Research Agent. Your job is to find accurate information.

Rules:
1. Always use web_search to find information — never guess facts.
2. Use specific, domain-rich queries. BAD: "RAG". GOOD: "Retrieval-Augmented Generation LLM AI 2024".
3. If first search returns off-topic results, retry with different keywords.
4. Summarise findings concisely — maximum 300 words.
5. Include 1-2 source URLs for key claims.
6. Return only the researched content — no meta-commentary.
"""


class ResearcherAgent(BaseSpecialistAgent):
    role = "researcher"
    role_prompt = RESEARCHER_PROMPT