from app.services.langgraph_agent_service import LangGraphAgentService


class DummyLLMService:
    def generate(self, prompt: str) -> str:
        assert "official LangGraph reasoning agent" in prompt
        return "A dummy graph-based answer."


def test_langgraph_agent_uses_graph_nodes_and_citations() -> None:
    service = LangGraphAgentService()
    service.agent.llm_service = DummyLLMService()

    documents = [
        {
            "page_id": "1",
            "title": "Test Graph Page",
            "text": "This document describes graph-based questions and their answers.",
            "source_url": "https://example.com/page/1",
            "metadata": {"space_key": "TEST"},
        },
        {
            "page_id": "2",
            "title": "Related Graph Topic",
            "text": "Graph theory and knowledge graph agents are connected through reasoning.",
            "source_url": "https://example.com/page/2",
            "metadata": {"space_key": "TEST"},
        },
    ]

    service.ingest_documents(documents)
    result = service.ask_question("What is the document about?", top_k=2)

    assert result["question"] == "What is the document about?"
    assert result["answer"] == "A dummy graph-based answer."
    assert len(result["nodes"]) == 2
    assert len(result["citations"]) == 2
    assert result["citations"][0]["page_id"] == "1"
    assert "https://example.com/page/1" in result["citations"][0]["source_url"]
