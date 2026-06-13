from app.services.rag_service import RAGService


class DummyLLMService:
    def generate(self, prompt: str) -> str:
        return "This is a dummy answer generated from the prompt."


def test_rag_service_returns_answer_for_query() -> None:
    service = RAGService()
    service.pipeline.llm_service = DummyLLMService()

    documents = [
        {
            "page_id": "1",
            "title": "Example Page",
            "source_url": "https://example.com/page/1",
            "text": "This page contains a sample knowledge base article about testing.",
            "metadata": {"space_key": "TEST"},
        }
    ]

    service.ingest_documents(documents)
    result = service.query("What is this page about?", top_k=1)

    assert result["question"] == "What is this page about?"
    assert result["answer"] == "This is a dummy answer generated from the prompt."
    assert len(result["retrieved_documents"]) == 1
