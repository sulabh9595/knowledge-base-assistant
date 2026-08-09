# DeepEval Implementation Plan for the Knowledge Base Application

> [!NOTE]
> **Status:** Fully Implemented & Validated
> **Date Completed:** 2026-07-17
>
> All phases of the plan have been successfully implemented, including extended safety metrics (Toxicity, Bias), quality check metrics (Refusal Quality, Citation Quality), synthetic data generator, multi-target runner (RAG, Agent, API RAG, API Agent), and the interactive Streamlit evaluation dashboard.

## Objective

Integrate DeepEval into this repository to evaluate the quality, groundedness, safety, and reliability of the RAG pipeline and LangGraph agent in a repeatable, production-style manner.

This plan is aligned to the architecture described in the knowledge-base architect guidance: local-first RAG, grounded responses, citation-aware answers, modular services, and observability-first design.

---

## Goals

1. Measure whether answers are factually grounded in retrieved context.
2. Measure whether retrieval is relevant and complete.
3. Measure whether the system answers the user question directly and usefully.
4. Catch hallucinations, unsafe content, and low-quality citations.
5. Add automated evaluation to CI and local development workflows.
6. Create a reusable evaluation harness for future model and prompt changes.

---

## Scope

### Primary evaluation targets

- Standard RAG flow in the retrieval and generation pipeline
- LangGraph agent reasoning flow
- Prompt quality and citation behavior
- Retrieval quality for chunked document context
- Safety and content quality for generated answers

### Systems to evaluate

- RAG pipeline: retrieval + answer generation
- LangGraph agent: graph-based retrieval + reasoning + citation formatting
- API endpoints that expose these flows
- Future prompt and model iterations

---

## Proposed implementation phases

### Phase 1 — Foundation and scaffolding

#### Deliverables

- Add DeepEval as a project dependency
- Create a dedicated evaluation module under the application services layer
- Create a reusable evaluation runner for local and CI usage
- Define configuration for evaluation model and dataset paths
- Define the LLM model strategy for evaluation, using the same local Ollama-based stack as the application

#### Model strategy

- Primary generation model for evaluation: the same Ollama-backed model used by the application, currently aligned to the project default such as `Qwen3:8b`.
- Evaluation judge model: use a configurable LLM backend so the plan can support either the same local model or a stronger external judge model depending on availability and cost.
- Keep the judge model configurable via environment settings so the evaluation flow can be adapted without code changes.
- Use the new `CONFIDENT_API_KEY` as an optional integration point for visualization, reporting, and external monitoring workflows alongside Langfuse.

#### Planned files

- `app/services/deepeval_service.py`
- `evals/` directory for datasets and scripts
- `tests/evals/` for evaluation-based tests
- `scripts/run_deepeval.py`

#### Implementation notes

- Keep evaluations separate from business logic so runtime services remain clean.
- Introduce a thin wrapper around DeepEval rather than embedding evaluation logic into the RAG service directly.
- Support both direct unit-style evaluation and end-to-end API-style evaluation.

---

### Phase 2 — Dataset design, goldens, and synthetic data generation

#### Deliverables

- Create an evaluation dataset with realistic knowledge-base questions
- Include expected context, expected answer behavior, and metadata for each case
- Cover success cases, failure cases, ambiguity, and unsupported-answer cases
- Add a synthetic data generation step to expand coverage efficiently

#### Dataset categories

1. Direct factual questions
2. Multi-hop or comparative questions
3. Questions that require citations
4. Questions with no direct answer in the indexed documents
5. Ambiguous or underspecified questions
6. Safety-focused edge cases

#### Recommended dataset structure

Each evaluation case should include:

- `input`: the user question
- `expected_answer`: ideal answer or expected behavior
- `expected_context_ids`: relevant document or page identifiers
- `expected_sources`: expected source titles or URLs
- `query_type`: factual, comparative, ambiguous, unsupported, etc.
- `metadata`: domain, document source, difficulty level

#### Suggested initial dataset size

- Start with 50–100 curated evaluation cases
- Expand to 200+ over time as confidence grows

#### Synthetic data generation approach

- Use the indexed knowledge base documents as the source corpus for generating additional evaluation cases.
- Generate synthetic questions from document chunks, metadata, and topic clusters.
- Produce multiple question types, including:
  - direct factual questions
  - comparative questions
  - multi-hop questions
  - unsupported-answer questions
  - ambiguous questions
- Generate expected answer hints and source references from the source documents.
- Use synthetic data to increase coverage, but keep a human-reviewed goldens subset for reliability.
- Maintain a clear split between:
  - human-curated goldens for quality benchmarking
  - synthetic data for expansion and regression testing

#### Synthetic generation workflow

1. Sample representative chunks from the indexed corpus.
2. Prompt an LLM to generate question-answer pairs grounded in those chunks.
3. Validate that the generated answers are supported by the source text.
4. Attach expected source IDs and metadata.
5. Review and filter low-quality or overly generic synthetic cases.

---

### Phase 3 — DeepEval metric integration

The project should evaluate both retrieval quality and generation quality. The plan should cover the maximum relevant DeepEval metrics for this use case.

#### Core metrics to implement

1. Faithfulness
   - Measures whether the generated answer is supported by the retrieved context.
   - Essential for groundedness and hallucination detection.

2. Answer Relevancy
   - Measures whether the answer addresses the user question directly.
   - Important for RAG and agent response quality.

3. Contextual Precision
   - Measures whether the retrieved context contains relevant information and avoids irrelevant context.
   - Helps identify poor retrieval behavior and noisy chunks.

4. Contextual Recall
   - Measures whether the retrieved context covers the full information required to answer the question.
   - Important for completeness and missing-context detection.

5. Contextual Relevancy
   - Measures how relevant the retrieved context is to the question overall.
   - Useful for understanding retrieval effectiveness.

6. Hallucination
   - Detects unsupported or invented content.
   - Critical for a knowledge base assistant that must remain grounded.

7. Toxicity
   - Flags harmful or inappropriate response content.
   - Important for safety and enterprise readiness.

8. Bias
   - Detects biased language and skewed answer patterns.
   - Useful for responsible AI checks.

9. G-Eval / custom LLM-based evaluation
   - Create custom evaluation criteria for:
     - citation quality
     - answer completeness
     - source groundedness
     - refusal quality for unsupported questions

10. Summarization metrics
   - If the project adds document summarization or agent summarization features, evaluate their quality as well.

#### Agent-specific metrics

For the LangGraph agent path, add custom evaluation criteria for:

- tool-use appropriateness
- citation usefulness
- reasoning trace quality
- answer transparency when confidence is low
- refusal quality for unsupported information

---

### Phase 4 — Evaluation harness design

#### Recommended design

Create one evaluation pipeline that can run against:

- the direct RAG pipeline
- the LangGraph agent flow
- API endpoints when a full end-to-end check is needed

#### Evaluation flow

1. Load evaluation cases from a JSON or YAML dataset
2. For each case, run the application’s retrieval + generation logic
3. Collect:
   - question
   - retrieved documents or graph nodes
   - generated answer
   - citations
4. Pass these into DeepEval metrics
5. Save results as JSON/CSV for review and trend analysis

#### Output artifacts

- `eval_results/latest.json`
- `eval_results/history.csv`
- `eval_results/summary.md`
- optional dashboard export for local or CI review

---

### Phase 5 — Integration with the existing architecture

#### RAG integration

Wrap or instrument the existing RAG flow so that evaluation can capture:

- the question
- retrieved documents
- similarity scores
- generated answer
- prompt used

This should be done without breaking the production API.

#### LangGraph integration

Wrap the agent flow so that evaluation can capture:

- graph nodes selected
- citations generated
- final answer
- whether the answer explicitly says it does not know when context is insufficient

#### API integration

Expose a lightweight internal endpoint or script for running evaluation jobs, for example:

- `POST /eval/run`
- `GET /eval/results`

For initial implementation, a script-based runner is simpler and safer than exposing full evaluation APIs to end users.

---

### Phase 6 — Automated quality gates

#### Local development

- Run evaluation suite before releasing major prompt or retrieval changes
- Support a fast smoke evaluation subset for quick feedback

#### CI/CD

- Run evaluation on pull requests that touch:
  - prompts
  - retrieval logic
  - embeddings
  - vector store behavior
  - agent reasoning logic
- Fail the build if key metrics regress beyond a defined threshold

#### Suggested thresholds

- Faithfulness: minimum threshold above 0.85
- Answer Relevancy: minimum threshold above 0.80
- Contextual Precision: minimum threshold above 0.80
- Contextual Recall: minimum threshold above 0.75
- Toxicity and Bias: should remain at zero or near-zero flagged cases

Thresholds should be adjusted after the initial dataset is evaluated.

---

### Phase 7 — Comprehensive observability, tracing, and reporting

#### Reporting strategy

- Store evaluation history over time
- Track metrics by:
  - prompt version
  - model version
  - chunk size / overlap settings
  - retrieval strategy
  - document source type

#### Langfuse and Confident tracing integration

Deep integration with the existing Langfuse service and the newly added Confident API key for metrics and report visualization:

1. **Per-evaluation trace spans**
   - Create a root span for each evaluation run
   - Log metadata: dataset name, case ID, metrics being evaluated, model version
   - Capture start/end timestamps and latency for each evaluation

2. **Per-metric execution spans**
   - Create a child span for each DeepEval metric (Faithfulness, Answer Relevancy, etc.)
   - Log inputs (question, context, response) at the start of each span
   - Log metric result (score, passing, reason) at span completion
   - Include execution time per metric

3. **Retrieval and generation tracing**
   - Capture RAG retrieval trace within evaluation context
   - Log retrieved documents, similarity scores, and chunk metadata
   - Log LLM generation prompts and responses for the judge model
   - Link evaluation trace to application traces for end-to-end visibility

4. **Custom attributes and tags**
   - Tag evaluations by:
     - dataset name
     - metric category (retrieval, generation, safety)
     - pass/fail status
     - evaluation run ID
     - reporting backend (Langfuse or Confident)
   - Attach user metadata for filtering and segmentation

#### Structured logging for evaluation runs

- Log all evaluation decisions and results in JSON format
- Include contextual information:
  - timestamp
  - evaluation run ID
  - case ID
  - metric name and configuration
  - score and pass/fail
  - error details if applicable
- Store logs locally and optionally export to centralized logging

#### Evaluation execution tracing

Track the full evaluation execution flow:

1. **Start of evaluation run**
   - Log dataset loaded (number of cases, split between goldens and synthetic)
   - Log metrics to be evaluated
   - Log model configuration (judge model, generation model, settings)

2. **Per-case execution**
   - Log case execution start with inputs
   - Log retrieval execution and results
   - Log generation execution and results
   - Log each metric evaluation with intermediate steps
   - Log case completion with all metric scores

3. **End of evaluation run**
   - Aggregate statistics (pass rate, score distribution, latency)
   - Comparison to previous runs (regressions, improvements)
   - Summary report generation

#### Integration opportunities

- Connect evaluation summaries to existing Langfuse tracing where possible
- Route evaluation results and report artifacts through the new Confident-backed reporting workflow when configured
- Export metrics to local JSON/CSV and optionally to Prometheus-compatible logs later
- Add structured logging for evaluation runs
- Create evaluation-specific Prometheus metrics:
  - `eval_metric_score_histogram` (per metric)
  - `eval_metric_latency_seconds` (per metric)
  - `eval_case_pass_rate` (overall success rate)
  - `eval_run_duration_seconds` (full run time)

#### Evaluation dashboard and visualization

Create a local evaluation dashboard using:

1. **Langfuse UI integration**
   - View evaluation runs as traces in Langfuse
   - Filter by dataset, metric, date range
   - Drill down into individual case traces
   - Compare metric scores across runs

2. **Confident reporting integration**
   - Use the configured `CONFIDENT_API_KEY` to publish evaluation summaries and scorecards
   - Visualize evaluation results, trends, and regressions in Confident-backed dashboards
   - Share report snapshots with stakeholders and maintain a history of runs

3. **Local dashboard (optional)**
   - Build a simple dashboard using Streamlit or FastAPI
   - Show evaluation history over time
   - Display metric trends and regressions
   - Compare multiple evaluation runs
   - Export results to CSV/PDF

4. **Metrics to visualize**
   - Score distributions for each metric
   - Pass rate trends over time
   - Latency per metric
   - Success/failure breakdown by query type
   - Model performance comparison
   - Dataset coverage and breakdown

---

### Phase 8 — Tracing infrastructure implementation

#### Langfuse and Confident tracing service extension

Extend the existing `LangfuseService` and add optional Confident-based reporting support for evaluation tracing and dashboard visibility:

1. **Evaluation context manager**
   - Create `@trace_evaluation()` decorator for wrapping evaluation functions
   - Automatically capture start/end times, inputs, outputs, and metadata
   - Support nested spans for metrics and sub-components

2. **Metric tracing wrapper**
   ```python
   with trace_evaluation_metric(
       name="faithfulness",
       inputs={"question": q, "response": r, "context": c},
       metadata={"case_id": "123", "dataset": "goldens"}
   ) as span:
       score = metric.measure(test_case)
       span.end({"score": score, "passed": score >= 0.7})
   ```

3. **Retrieval and generation logging**
   - Log RAG retrieval events within evaluation context
   - Log judge model generation prompts and responses
   - Include latency and token counts where available

#### Metrics export to Prometheus

Create evaluation-specific Prometheus metrics:

1. **Histogram metrics**
   - `eval_metric_score` (per metric, tagged by metric_name, dataset, run_id)
   - `eval_metric_latency_seconds` (per metric)

2. **Gauge metrics**
   - `eval_case_pass_rate` (overall success rate for latest run)
   - `eval_run_duration_seconds` (total evaluation run time)

3. **Counter metrics**
   - `eval_cases_processed_total` (cumulative cases evaluated)
   - `eval_metrics_evaluated_total` (cumulative metric evaluations)

#### Local storage and history

1. **Evaluation results storage**
   - Directory: `eval_results/`
   - Structure:
     ```
     eval_results/
       latest.json          # Latest evaluation run
       history/
         run_2026-07-17_10-30-00.json
         run_2026-07-17_14-15-00.json
       metrics.csv          # Aggregated metrics over time
     ```

2. **JSON result schema**
   ```json
   {
     "run_id": "uuid",
     "timestamp": "ISO8601",
     "dataset": "goldens",
     "model": "Qwen3:8b",
     "metrics_evaluated": ["faithfulness", "answer_relevancy"],
     "cases": [
       {
         "case_id": "123",
         "question": "...",
         "response": "...",
         "context": "...",
         "metric_results": {
           "faithfulness": {"score": 0.85, "passed": true, "latency_ms": 2500},
           "answer_relevancy": {"score": 0.92, "passed": true, "latency_ms": 1800}
         }
       }
     ],
     "summary": {
       "total_cases": 50,
       "passed_cases": 48,
       "pass_rate": 0.96,
       "avg_latency_ms": 2150,
       "total_duration_seconds": 107.5
     }
   }
   ```

#### Implementation files

Add to the project structure:

```text
app/services/
  deepeval_service.py          # Enhanced with tracing
  deepeval_tracing.py          # New: Langfuse tracing integration
  
evals/
  datasets/
    knowledge_base_cases.json
  scripts/
    run_deepeval.py
    export_eval_results.py     # New: Export and format results
  
eval_results/
  latest.json
  history/
  metrics.csv

frontend/
  eval_dashboard.py            # New: Streamlit dashboard for results
```

#### Tracing API design

New service methods to expose tracing:

```python
# In deepeval_service.py or deepeval_tracing.py

def evaluate_with_tracing(
    task: str,
    response: str,
    context: str,
    case_id: str,
    dataset: str = "goldens",
    use_langfuse: bool = True,
) -> Dict[str, Any]:
    """Run evaluation with full Langfuse tracing."""
    
def trace_evaluation_run(
    dataset_path: str,
    metrics: List[str],
    dataset_type: str = "goldens",
) -> Dict[str, Any]:
    """Run full evaluation suite with detailed tracing."""
    
def export_eval_results(
    run_id: str,
    format: str = "json",  # or "csv", "markdown"
) -> str:
    """Export evaluation results to specified format."""
```

---

## Recommended implementation order

1. Add DeepEval dependency and create the evaluation module (Phase 1)
2. Build a small curated evaluation dataset (Phase 2)
3. Implement core metrics: Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall (Phase 3)
4. Create evaluation harness and RAG/LangGraph integration (Phases 4-5)
5. **Add Langfuse tracing integration to evaluation service (Phase 8 — early priority)**
6. Add safety metrics: Toxicity and Bias (Phase 3 continued)
7. Evaluate the LangGraph agent path with custom G-Eval criteria (Phase 3 continued)
8. Add a local runner and CI integration (Phase 6)
9. Create evaluation results dashboard and visualization (Phase 8)
10. Review results and tune prompts, retrieval settings, or chunking strategy
11. Set up automated quality gates and CI/CD (Phase 6 continued)

---

## Suggested project structure additions

```text
app/
  services/
    deepeval_service.py

evals/
  datasets/
    knowledge_base_cases.json
  scripts/
    run_deepeval.py

tests/
  evals/
    test_rag_evals.py
    test_agent_evals.py
```

---

## Risks and mitigations

### Risk: DeepEval judges may be expensive or inconsistent

Mitigation:
- Start with a smaller evaluation set
- Use a stable judge model for CI
- Keep the initial metric set focused on the highest-value checks

### Risk: Local-only evaluation may be noisy

Mitigation:
- Curate the dataset carefully
- Compare results over time rather than treating one run as absolute truth
- Use deterministic prompts and fixed retrieval settings for evaluation

### Risk: Evaluation becomes a one-off task

Mitigation:
- Make it part of the development workflow
- Add a script and CI step
- Save evaluation history for regression tracking

---

## Success criteria

The implementation is successful when:

- the project can run automated DeepEval evaluations locally
- the RAG pipeline and LangGraph agent are both evaluated against multiple metrics
- results are stored and reviewable over time
- prompt or retrieval changes can be measured objectively
- the system can detect hallucinations and unsupported answers before release
- **all evaluation runs are traced and visible in Langfuse**
- **evaluation summaries can be published through Confident when the API key is configured**
- **per-metric execution times are tracked and reported**
- **evaluation results are exported to JSON/CSV with full history**
- **a dashboard or UI exists to visualize evaluation trends over time**
- **Prometheus metrics are exposed for evaluation monitoring**

---

## Recommended first milestone

Implement the first milestone as:

- one curated dataset of 25–50 questions
- evaluation for RAG only
- metrics: Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall, Hallucination
- **Langfuse tracing integration (per-run and per-metric spans)**
- **Confident reporting integration using the configured API key**
- **JSON export of evaluation results with history tracking**
- local script runner and initial reporting
- **Prometheus metrics exposure for key metrics**

After that milestone is stable, extend to:
- LangGraph agent evaluation
- safety metrics (Toxicity, Bias)
- evaluation dashboard/UI
- CI automation and quality gates
