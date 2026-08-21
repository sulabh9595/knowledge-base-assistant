# OPIK Evaluation Dataset

This directory contains OPIK validation dataset scaffolding for multi-agent validation.

## Structure

- `opik_cases.json` — seed cases for RAG and LangGraph agent validation.

## Case format

Each case is an object with the following fields:

- `case_id`: unique identifier for the validation case
- `input`: user question or prompt
- `expected_output`: optional expected answer text or behavior
- `query_type`: type of query such as `factual`, `agent`, or `unsupported`
- `target`: one of `rag`, `agent`, `api-rag`, or `api-agent`
- `metadata`: optional additional metadata about the case

## Usage

Run the validation script:

```bash
python scripts/run_opik_validation.py --target rag --dataset opik
```
