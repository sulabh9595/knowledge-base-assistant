from __future__ import annotations

from typing import Optional

from graph.langgraph_agent import LangGraphAgent


class LangGraphAgentService:
    def __init__(self, agent: Optional[LangGraphAgent] = None) -> None:
        self.agent = agent or LangGraphAgent()

    def ingest_documents(self, documents: list[dict]) -> None:
        self.agent.ingest_documents(documents)

    def ask_question(self, question: str, top_k: int = 3) -> dict:
        return self.agent.ask(question, top_k=top_k)


langgraph_service = LangGraphAgentService()
