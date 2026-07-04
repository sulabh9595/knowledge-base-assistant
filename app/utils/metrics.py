# Creator: Sulabh Bansod
# Description: Prometheus custom metrics definition.
# Use: Exposes counters and histograms to track latency and ingestion performance.

from prometheus_client import Counter, Histogram

# Ingestion metrics
DOCUMENTS_INGESTED = Counter(
    "kb_documents_ingested_total",
    "Total number of successfully ingested documents",
    ["source"],  # e.g. "confluence", "file_upload"
)
INGESTION_FAILURES = Counter(
    "kb_ingestion_failures_total",
    "Total number of document ingestion failures",
    ["source"],
)

# Latency and performance metrics
RAG_QUERY_LATENCY = Histogram(
    "kb_rag_query_latency_seconds",
    "Latency of RAG pipeline question answering",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0),
)
LLM_GEN_LATENCY = Histogram(
    "kb_llm_generation_latency_seconds",
    "Latency of Ollama LLM text generation",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0),
)
VECTOR_SEARCH_LATENCY = Histogram(
    "kb_vector_search_latency_seconds",
    "Latency of similarity search inside Chroma DB",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
