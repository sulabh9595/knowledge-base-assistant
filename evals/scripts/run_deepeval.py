#!/usr/bin/env python3
"""Evaluation runner script for DeepEval.

Runs queries from a dataset (goldens or synthetic) against RAG, LangGraph Agent,
or API endpoints, evaluates results against all metrics (including safety and quality),
logs spans to Langfuse, exposes Prometheus metrics, and exports reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure python path includes root directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config.settings import settings
from app.rag.pipeline import RAGPipeline
from app.services.deepeval_service import DeepEvalService
from app.utils.metrics import EVAL_CASE_PASS_RATE, EVAL_RUN_DURATION


def run_query(target: str, question: str, top_k: int = 3, api_url: str = "http://127.0.0.1:8000") -> Tuple[str, List[str], List[Dict[str, Any]], float]:
    """Runs the query against the target and returns (answer, retrieval_context, retrieved_docs, latency)."""
    start_time = time.time()

    if target == "rag":
        pipeline = RAGPipeline()
        # Ingest documents if DB is empty
        try:
            doc_count = pipeline.vector_store.store._collection.count()
        except Exception:
            doc_count = 0
        if doc_count == 0:
            base_dir = Path(__file__).resolve().parents[2]
            doc_store = base_dir / "memory/documents.json"
            if doc_store.exists():
                with doc_store.open("r", encoding="utf-8") as f:
                    documents = json.load(f)
                pipeline.ingest_documents(documents)
        
        res = pipeline.answer_question(question, top_k=top_k)
        latency = time.time() - start_time
        retrieved_docs = res["retrieved_documents"]
        retrieval_context = [doc["text"] for doc in retrieved_docs]
        docs_summary = [
            {"page_id": doc["page_id"], "title": doc["title"], "similarity_score": doc["similarity_score"]}
            for doc in retrieved_docs
        ]
        return res["answer"], retrieval_context, docs_summary, latency

    elif target == "agent":
        from graph.langgraph_agent import LangGraphAgent
        agent = LangGraphAgent()
        if not agent.graph.nodes:
            base_dir = Path(__file__).resolve().parents[2]
            doc_store = base_dir / "memory/documents.json"
            if doc_store.exists():
                with doc_store.open("r", encoding="utf-8") as f:
                    documents = json.load(f)
                agent.ingest_documents(documents)

        res = agent.ask(question, top_k=top_k)
        latency = time.time() - start_time
        retrieval_context = []
        docs_summary = []
        for node_dict in res["nodes"]:
            page_id = node_dict["page_id"]
            if page_id in agent.graph.nodes:
                retrieval_context.append(agent.graph.nodes[page_id].text)
                docs_summary.append({
                    "page_id": page_id,
                    "title": agent.graph.nodes[page_id].title,
                    "similarity_score": 1.0
                })
        return res["answer"], retrieval_context, docs_summary, latency

    elif target == "api-rag":
        import httpx
        res = httpx.post(f"{api_url}/rag/query", json={"question": question, "top_k": top_k}, timeout=120)
        res.raise_for_status()
        data = res.json()
        latency = time.time() - start_time
        
        # Retrieve context from /documents/{page_id}
        retrieval_context = []
        docs_summary = []
        for doc in data.get("retrieved_documents", []):
            page_id = doc["page_id"]
            docs_summary.append({
                "page_id": page_id,
                "title": doc["title"],
                "similarity_score": doc.get("similarity_score", 1.0)
            })
            try:
                page_res = httpx.get(f"{api_url}/documents/{page_id}", timeout=10)
                if page_res.status_code == 200:
                    retrieval_context.append(page_res.json()["text"])
            except Exception:
                pass
        return data["answer"], retrieval_context, docs_summary, latency

    elif target == "api-agent":
        import httpx
        res = httpx.post(f"{api_url}/agent/langgraph/query", json={"question": question, "top_k": top_k}, timeout=120)
        res.raise_for_status()
        data = res.json()
        latency = time.time() - start_time
        
        # Retrieve context from /documents/{page_id}
        retrieval_context = []
        docs_summary = []
        for node in data.get("nodes", []):
            page_id = node["page_id"]
            docs_summary.append({
                "page_id": page_id,
                "title": node["title"],
                "similarity_score": 1.0
            })
            try:
                page_res = httpx.get(f"{api_url}/documents/{page_id}", timeout=10)
                if page_res.status_code == 200:
                    retrieval_context.append(page_res.json()["text"])
            except Exception:
                pass
        return data["answer"], retrieval_context, docs_summary, latency

    else:
        raise ValueError(f"Unknown target: {target}")


def run_evaluation() -> None:
    parser = argparse.ArgumentParser(description="Run RAG or Agent DeepEval suite.")
    parser.add_argument("--target", type=str, default="rag", choices=["rag", "agent", "api-rag", "api-agent"], help="Evaluation target.")
    parser.add_argument("--dataset", type=str, default="goldens", choices=["goldens", "synthetic"], help="Dataset to run.")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000", help="FastAPI backend base URL.")
    parser.add_argument("--top-k", type=int, default=3, help="Top K documents to retrieve.")
    parser.add_argument("--use-heuristics", action="store_true", help="Force fast heuristic fallback evaluation without LLM queries.")
    args = parser.parse_args()

    # 1. Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    dataset_file = "knowledge_base_cases.json" if args.dataset == "goldens" else "synthetic_cases.json"
    dataset_path = base_dir / "evals/datasets" / dataset_file
    results_dir = base_dir / "eval_results"
    history_dir = results_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("      DEEPEVAL INTEGRATED PIPELINE EVALUATION HARNESS      ")
    print("=" * 60)
    print(f"Target:            {args.target.upper()}")
    print(f"Dataset:           {args.dataset.upper()} ({dataset_path.name})")
    print(f"Top K:             {args.top_k}")
    if args.target.startswith("api-"):
        print(f"Backend URL:       {args.api_url}")
    print("=" * 60)

    # 2. Check dataset
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        if args.dataset == "synthetic":
            print("Please run `python evals/scripts/generate_synthetic_data.py` first.")
        sys.exit(1)

    with dataset_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} evaluation cases.")

    # 3. Setup environments
    if settings.confident_api_key:
        os.environ["CONFIDENT_API_KEY"] = settings.confident_api_key
        print("Confident AI API Key is configured. Runs will be tracked.")

    # 4. Initialize DeepEval service
    deepeval_service = DeepEvalService()
    if args.use_heuristics:
        deepeval_service.judge = None
        print("Forcing fast heuristic fallback evaluation (no LLM judge calls).")
    run_id = f"run_{args.target}_{args.dataset}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    timestamp = datetime.utcnow().isoformat()

    metrics_to_run = [
        "faithfulness",
        "answer_relevancy",
        "contextual_precision",
        "contextual_recall",
        "hallucination",
        "toxicity",
        "bias",
        "refusal_quality",
        "citation_quality"
    ]

    print(f"Starting run ID: {run_id}")
    print(f"Metrics: {', '.join(metrics_to_run)}")
    print("-" * 60)

    evaluated_cases = []
    total_passed_cases = 0
    start_run_time = time.time()

    # Track metrics aggregates
    metric_sums: Dict[str, float] = {m: 0.0 for m in metrics_to_run}
    metric_counts: Dict[str, int] = {m: 0 for m in metrics_to_run}

    # Start root Langfuse trace
    with deepeval_service.tracing_service.trace_run(
        dataset_name=args.dataset,
        run_id=run_id,
        model_name=deepeval_service.judge_model_name,
        metrics=metrics_to_run,
    ) as run_trace:

        for idx, case in enumerate(cases, 1):
            question = case["input"]
            expected_output = case.get("expected_output")
            case_id = f"case_{idx}"
            query_type = case.get("query_type", "factual")

            print(f"[{idx}/{len(cases)}] Evaluating: {question[:50]}...")

            try:
                # Run query against target
                actual_output, retrieval_context, docs_summary, query_duration = run_query(
                    target=args.target,
                    question=question,
                    top_k=args.top_k,
                    api_url=args.api_url
                )

                # Run DeepEval Metrics
                case_eval = deepeval_service.evaluate_case(
                    question=question,
                    answer=actual_output,
                    retrieval_context=retrieval_context,
                    expected_output=expected_output,
                    metrics_to_run=metrics_to_run,
                    case_id=case_id,
                    dataset_name=args.dataset,
                    run_id=run_id,
                    parent_span=run_trace,
                )

                # Check if all metrics passed
                all_passed = True
                for m in metrics_to_run:
                    if m in case_eval:
                        metric_sums[m] += case_eval[m]["score"]
                        metric_counts[m] += 1
                        if not case_eval[m]["passed"]:
                            all_passed = False

                if all_passed:
                    total_passed_cases += 1

                evaluated_cases.append({
                    "case_id": case_id,
                    "question": question,
                    "expected_output": expected_output,
                    "actual_output": actual_output,
                    "retrieved_documents": docs_summary,
                    "metric_results": case_eval,
                    "latency_seconds": query_duration,
                    "all_passed": all_passed,
                    "query_type": query_type
                })

                print(f"   Passed: {all_passed} | Latency: {query_duration:.2f}s")
                for m, res in case_eval.items():
                    print(f"   - {m}: score={res['score']:.3f} passed={res['passed']}")

            except Exception as e:
                print(f"   Error evaluating case {idx}: {e}")
                evaluated_cases.append({
                    "case_id": case_id,
                    "question": question,
                    "error": str(e),
                    "all_passed": False
                })

    total_duration = time.time() - start_run_time
    pass_rate = total_passed_cases / len(cases) if cases else 0.0

    # Record to Prometheus gauges
    try:
        EVAL_CASE_PASS_RATE.labels(dataset_name=args.dataset, run_id=run_id).set(pass_rate)
        EVAL_RUN_DURATION.labels(dataset_name=args.dataset, run_id=run_id).set(total_duration)
    except Exception:
        pass

    # Compute averages
    metric_avgs = {}
    for m in metrics_to_run:
        metric_avgs[m] = metric_sums[m] / metric_counts[m] if metric_counts[m] > 0 else 0.0

    # 6. Save results
    summary = {
        "total_cases": len(cases),
        "passed_cases": total_passed_cases,
        "pass_rate": round(pass_rate, 3),
        "total_duration_seconds": round(total_duration, 2),
        "avg_metric_scores": {m: round(score, 3) for m, score in metric_avgs.items()}
    }

    run_report = {
        "run_id": run_id,
        "timestamp": timestamp,
        "target": args.target,
        "dataset": args.dataset,
        "judge_model": deepeval_service.judge_model_name,
        "summary": summary,
        "cases": evaluated_cases
    }

    # Save latest.json
    latest_path = results_dir / "latest.json"
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(run_report, f, ensure_ascii=False, indent=2)

    # Save history json
    history_path = history_dir / f"{run_id}.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(run_report, f, ensure_ascii=False, indent=2)

    # Append to metrics.csv
    csv_path = results_dir / "metrics.csv"
    csv_headers = [
        "run_id", "timestamp", "target", "dataset", "total_cases", "passed_cases", "pass_rate",
        "avg_faithfulness", "avg_answer_relevancy", "avg_contextual_precision",
        "avg_contextual_recall", "avg_hallucination", "avg_toxicity", "avg_bias",
        "avg_refusal_quality", "avg_citation_quality", "duration_seconds"
    ]
    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(csv_headers)
        writer.writerow([
            run_id,
            timestamp,
            args.target,
            args.dataset,
            len(cases),
            total_passed_cases,
            round(pass_rate, 4),
            round(metric_avgs.get("faithfulness", 0.0), 4),
            round(metric_avgs.get("answer_relevancy", 0.0), 4),
            round(metric_avgs.get("contextual_precision", 0.0), 4),
            round(metric_avgs.get("contextual_recall", 0.0), 4),
            round(metric_avgs.get("hallucination", 0.0), 4),
            round(metric_avgs.get("toxicity", 0.0), 4),
            round(metric_avgs.get("bias", 0.0), 4),
            round(metric_avgs.get("refusal_quality", 0.0), 4),
            round(metric_avgs.get("citation_quality", 0.0), 4),
            round(total_duration, 2)
        ])

    print("=" * 60)
    print("                  EVALUATION COMPLETE                       ")
    print("=" * 60)
    print(f"Run ID:            {run_id}")
    print(f"Timestamp:         {timestamp}")
    print(f"Pass Rate:         {pass_rate * 100:.1f}% ({total_passed_cases}/{len(cases)})")
    print(f"Total Duration:    {total_duration:.2f} seconds")
    print("-" * 60)
    print("Average Scores:")
    for m, score in metric_avgs.items():
        print(f"  - {m:<20}: {score:.3f}")
    print("-" * 60)
    print(f"Latest report saved: {latest_path}")
    print(f"History report saved: {history_path}")
    print(f"Metrics CSV updated: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
