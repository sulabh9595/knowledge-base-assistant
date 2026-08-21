# OPIK Multi-Agent Validation Plan

> [!NOTE]
> **Purpose:** Define an implementation plan for adding OPIK-style validation to the multi-agent flows in the Knowledge Base Application.
> **Scope:** Validate LangGraph agent, RAG pipeline, and multi-agent orchestration with repeatable, evidence-based judgments.

## Objective

Implement an OPIK-based validation layer for multi-agent evaluation in this repository. The goal is to measure answer quality, groundedness, tool usage, and agent collaboration performance across the projects existing AI services.

## Goals

1. Create a reusable validation harness for multi-agent responses.
2. Ensure the LangGraph agent and RAG endpoints are evaluated with the same validation semantics.
3. Surface agent-specific issues such as hallucinations, unsupported claims, bad citations, or tool misuse.
4. Enable local and CI validation of new prompt/model changes.
5. Keep validation modular so it can evolve independently from runtime request handling.

## What OPIK Should Validate

- Answer groundedness: is the response supported by the retrieved documents?
- Citation quality: are references meaningful, accurate, and correctly formatted?
- Tool correctness: did the LangGraph agent invoke the right retrieval tools or sub-flows?
- Multi-agent coordination: if multiple agents are involved, does their combined output remain coherent and factually correct?
- Refusal quality: does the system refuse gracefully when the answer cannot be grounded?
- Safety and alignment: are outputs free of toxic, biased, or unsafe language?

## Proposed Implementation Phases

### Phase 1 — Foundation

- Add a dedicated validation module under `app/services/` or `app/validation/`.
- Define an OPIK validation runner that can call:
  - `app/api/rag.py` path
  - `app/api/langgraph.py` path
  - any shared multi-agent orchestration flows
- Create a configurable evaluation entrypoint for local runs and CI.
- Keep validation harness separate from production request handlers.

#### Deliverables

- `app/services/opik_validation_service.py`
- `tests/test_opik_validation.py`
- `evals/opik/` dataset scaffolding
- `scripts/run_opik_validation.py`

### Phase 2 — Dataset and Cases

- Create a validation dataset covering:
  - agent reasoning with citations
  - conflicting source content
  - unsupported or missing knowledge cases
  - multi-step and multi-agent coordination cases
  - safety-sensitive queries
- Include expected validation signals such as:
  - supported vs unsupported
  - citation presence
  - refusal behavior
  - tool usage correctness
- Start with a small seed dataset and expand to cover regressions.

### Phase 3 — OPIK Metrics and Checks

Implement checks tailored for multi-agent validation:

- Grounding Score
- Citation Accuracy
- Tool Invocation Validity
- Response Relevance
- Answer Correctness
- Safety / Toxicity
- Refusal Quality

For agent flows, add:

- Reasoning Trace Audit
- Tool Selection Audit
- Multi-Agent Consistency

### Phase 4 — Integration and Automation

- Wire OPIK validation into local scripts and CI.
- Add targeted tests for known failure modes.
- Add a reporting artifact for validation results.
- Optionally integrate with existing observability / evaluation dashboards.

## Recommended Files and Locations

- `.skills/opik.md` — this plan document
- `app/services/opik_validation_service.py` — validation service implementation
- `evals/opik/` — dataset cases and expected outcomes
- `tests/test_opik_validation.py` — unit/integration validation tests
- `scripts/run_opik_validation.py` — local validation runner

## Success Criteria

- Validation harness can run against both RAG and LangGraph agent flows.
- The system flags unsupported answers and poor citation behavior.
- The framework is reusable for future model, prompt, or agent updates.
- Validation runs can be executed locally and in CI without impacting production runtime.

## Notes

- The plan should preserve separation between runtime traffic handling and validation/evaluation logic.
- Focus OPIK validation on evidence-based judgment rather than serving live user requests.
- Keep the design extensible so new agents or tools can join the validation flow.
