from pathlib import Path
from tempfile import TemporaryDirectory

from langchain.docstore.document import Document

from app.vectorstore.chroma_store import ChromaStore


class DummyEmbeddingProvider:
    def __init__(self) -> None:
        self.client = self

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def test_chroma_store_add_and_similarity_search():
    with TemporaryDirectory() as tmpdir:
        provider = DummyEmbeddingProvider()
        store = ChromaStore(
            embedding_provider=provider,
            persist_directory=tmpdir,
            collection_name="test_collection",
        )

        document = Document(
            page_content="Hello world",
            metadata={"page_id": "1", "title": "Test", "source_url": "https://example.com"},
        )
        store.add_documents([document])

        results = store.similarity_search_with_score("Hello", k=1)
        assert len(results) == 1
        returned_doc, score = results[0]
        assert returned_doc.page_content == "Hello world"
        assert returned_doc.metadata["page_id"] == "1"
        assert isinstance(score, float)

        assert Path(tmpdir).exists()
