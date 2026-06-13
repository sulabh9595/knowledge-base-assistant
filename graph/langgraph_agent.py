from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from app.config.settings import settings
from app.embeddings.embeddings import EmbeddingProvider
from app.services.llm_service import OllamaService


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools: Dict[str, Tool] = {tool.name: tool for tool in tools}

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def available_descriptions(self) -> str:
        return "\n".join(f"{tool.name}: {tool.description}" for tool in self._tools.values())


@dataclass
class KnowledgeGraphCitation:
    page_id: str
    title: str
    source_url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return {
            "page_id": self.page_id,
            "title": self.title,
            "source_url": self.source_url,
            "snippet": self.snippet,
        }


@dataclass
class KnowledgeGraphNode:
    page_id: str
    title: str
    text: str
    source_url: str
    metadata: dict[str, Any]
    edges: set[str]
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "title": self.title,
            "source_url": self.source_url,
            "edges": sorted(self.edges),
        }

    def keyword_set(self) -> set[str]:
        tokens = re.findall(r"\w+", f"{self.title} {self.text}".lower())
        return {token for token in tokens if len(token) > 3}


class KnowledgeGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, KnowledgeGraphNode] = {}

    def add_document(self, document: dict[str, Any]) -> None:
        page_id = document["page_id"]
        node = KnowledgeGraphNode(
            page_id=page_id,
            title=document.get("title", ""),
            text=document.get("text", ""),
            source_url=document.get("source_url", ""),
            metadata=document.get("metadata", {}),
            edges=set(),
        )
        self.nodes[page_id] = node
        self._link_node(node)

    def _link_node(self, node: KnowledgeGraphNode) -> None:
        for other_id, other in self.nodes.items():
            if other_id == node.page_id:
                continue

            if self._should_connect(node, other):
                node.edges.add(other.page_id)
                other.edges.add(node.page_id)

    def _should_connect(self, node: KnowledgeGraphNode, other: KnowledgeGraphNode) -> bool:
        return bool(node.keyword_set() & other.keyword_set())

    def query(self, question: str, top_k: int = 3, query_embedding: Optional[list[float]] = None) -> list[KnowledgeGraphNode]:
        question_tokens = set(re.findall(r"\w+", question.lower()))
        node_scores: dict[str, float] = {}

        for node in self.nodes.values():
            score = float(len(question_tokens & node.keyword_set()))
            score += len(node.edges) * 0.1

            if query_embedding is not None and node.embedding is not None:
                score += self._cosine_similarity(query_embedding, node.embedding) * 100.0

            node_scores[node.page_id] = score

        ranked_ids = sorted(self.nodes.keys(), key=lambda key: node_scores[key], reverse=True)
        return [self.nodes[node_id] for node_id in ranked_ids[:top_k]]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class LangGraphAgent:
    def __init__(self, llm_service: OllamaService | None = None) -> None:
        self.graph = KnowledgeGraph()
        self.llm_service = llm_service or OllamaService()
        self.embedding_provider: Optional[EmbeddingProvider] = None
        if settings.ollama_host and settings.embedding_model:
            self.embedding_provider = EmbeddingProvider()

        self.tools = ToolRegistry(
            [
                Tool(
                    name="graph_search",
                    description="Find the most relevant knowledge graph nodes for a question.",
                    func=self._graph_search,
                ),
                Tool(
                    name="citation_formatter",
                    description="Generate source citations from selected graph nodes.",
                    func=self._format_citations,
                ),
            ]
        )

    def ingest_documents(self, documents: Iterable[dict[str, Any]]) -> None:
        for document in documents:
            self.graph.add_document(document)
            if self.embedding_provider and self.embedding_provider.client:
                try:
                    node = self.graph.nodes[document["page_id"]]
                    node.embedding = self.embedding_provider.embed_query(node.text or node.title)
                except Exception:
                    node.embedding = None

    def ask(self, question: str, top_k: int = 3) -> dict[str, Any]:
        graph_nodes = self.tools.get("graph_search").execute(question, top_k=top_k)
        citations = self.tools.get("citation_formatter").execute(graph_nodes)
        context = self._build_context(graph_nodes)
        prompt = self._build_prompt(question, context, citations)
        answer = self.llm_service.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "nodes": [node.to_dict() for node in graph_nodes],
            "citations": [citation.to_dict() for citation in citations],
        }

    def _graph_search(self, question: str, top_k: int = 3) -> list[KnowledgeGraphNode]:
        query_embedding = None
        if self.embedding_provider and self.embedding_provider.client:
            try:
                query_embedding = self.embedding_provider.embed_query(question)
            except Exception:
                query_embedding = None

        return self.graph.query(question, top_k=top_k, query_embedding=query_embedding)

    def _format_citations(self, nodes: list[KnowledgeGraphNode]) -> list[KnowledgeGraphCitation]:
        return [
            KnowledgeGraphCitation(
                page_id=node.page_id,
                title=node.title,
                source_url=node.source_url,
                snippet=self._build_snippet(node.text),
            )
            for node in nodes
        ]

    def _build_context(self, nodes: list[KnowledgeGraphNode]) -> str:
        return "\n\n".join(
            f"Title: {node.title}\nURL: {node.source_url}\nEdges: {', '.join(sorted(node.edges))}\nContent:\n{node.text}"
            for node in nodes
        )

    def _build_snippet(self, text: str, max_length: int = 200) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= max_length:
            return cleaned
        return f"{cleaned[:max_length].rstrip()}..."

    def _build_prompt(self, question: str, context: str, citations: list[KnowledgeGraphCitation]) -> str:
        tool_descriptions = self.tools.available_descriptions()
        citations_block = "\n".join(
            f"- {citation.title} ({citation.source_url}): {citation.snippet}"
            for citation in citations
        )

        return (
            "You are an official LangGraph reasoning agent. Use the graph search tool results and the citations below "
            "to answer the question accurately and transparently. If the answer is not contained in the provided context, say that you do not know.\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            f"Graph context:\n{context}\n\n"
            f"Citations:\n{citations_block}\n\n"
            f"Question: {question}\nAnswer:"
        )
