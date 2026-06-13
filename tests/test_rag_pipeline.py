from app.rag.pipeline import RAGPipeline
from app.services.llm_service import OllamaService
from langchain.docstore.document import Document
from app.vectorstore.interfaces import VectorStoreRepository


class DummyStore(VectorStoreRepository):
    def __init__(self) -> None:
        self.documents = []

    def add_documents(self, documents):
        self.documents.extend(documents)

    def similarity_search_with_score(self, query: str, k: int = 5):
        return [(self.documents[0], 0.95)] if self.documents else []

    def as_retriever(self, search_kwargs=None):
        class DummyRetriever:
            def get_relevant_documents(self, query: str):
                return [self.documents[0]] if self.documents else []

        retriever = DummyRetriever()
        retriever.documents = self.documents
        return retriever

    def delete(self, ids: list[str]) -> None:
        pass

    def update(self, documents):
        self.documents = documents

    def reindex(self, documents):
        self.documents = documents


class DummyLLMService(OllamaService):
    def __init__(self) -> None:
        pass

    def generate(self, prompt: str) -> str:
        return "dummy answer"


def test_rag_pipeline_preserves_metadata_in_chunks():
    pipeline = RAGPipeline(
        llm_service=DummyLLMService(),
        vector_store=DummyStore(),
    )

    pipeline.ingest_documents([
        {
            "page_id": "p1",
            "title": "Title",
            "source_url": "https://example.com/page/1",
            "text": "This is a simple test document that should be chunked for ingestion.",
            "metadata": {"author": "tester"},
        }
    ])

    assert pipeline.vector_store.documents
    first_chunk = pipeline.vector_store.documents[0]
    assert first_chunk.metadata["page_id"] == "p1"
    assert first_chunk.metadata["title"] == "Title"
    assert first_chunk.metadata["source_url"] == "https://example.com/page/1"
    assert first_chunk.metadata["author"] == "tester"
    assert "chunk_id" in first_chunk.metadata


def test_rag_pipeline_returns_answer_with_scores():
    pipeline = RAGPipeline(
        llm_service=DummyLLMService(),
        vector_store=DummyStore(),
    )

    pipeline.ingest_documents([
        {
            "page_id": "p1",
            "title": "Title",
            "source_url": "https://example.com/page/1",
            "text": "This is a simple test document.",
            "metadata": {"author": "tester"},
        }
    ])

    result = pipeline.answer_question("What is this document about?", top_k=1)
    assert result["question"] == "What is this document about?"
    assert result["answer"] == "dummy answer"
    assert result["retrieved_documents"]
    assert result["retrieved_documents"][0]["similarity_score"] == 0.95
