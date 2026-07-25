# Creator: Sulabh Bansod
# Description: Prometheus custom metrics definition.
# Use: Exposes counters and histograms to track latency and ingestion performance.

from prometheus_client import Counter, Histogram, Gauge

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

# Evaluation metrics
EVAL_CASE_PASS_RATE = Gauge(
    "kb_eval_case_pass_rate",
    "Overall success rate for the latest evaluation run",
    ["dataset_name", "run_id"],
)
EVAL_RUN_DURATION = Gauge(
    "kb_eval_run_duration_seconds",
    "Total evaluation run time in seconds",
    ["dataset_name", "run_id"],
)
EVAL_METRIC_SCORE = Histogram(
    "kb_eval_metric_score",
    "Score per evaluation metric",
    ["metric_name", "dataset_name", "run_id"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
EVAL_METRIC_LATENCY = Histogram(
    "kb_eval_metric_latency_seconds",
    "Latency per evaluation metric",
    ["metric_name"],
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0),
)
EVAL_CASES_PROCESSED = Counter(
    "kb_eval_cases_processed_total",
    "Cumulative cases evaluated",
    ["dataset_name"],
)
EVAL_METRICS_EVALUATED = Counter(
    "kb_eval_metrics_evaluated_total",
    "Cumulative metric evaluations",
    ["metric_name", "dataset_name"],
)

