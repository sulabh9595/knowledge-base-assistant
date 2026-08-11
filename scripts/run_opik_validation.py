#!/usr/bin/env python3
"""Runner for OPIK multi-agent validation cases."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.opik_validation_service import OPIKValidationService
from app.config.settings import settings


def run_query(target: str, question: str, top_k: int = 3, api_url: str = "http://127.0.0.1:8000") -> Tuple[str, List[str], List[Dict[str, Any]], float]:
    from app.rag.pipeline import RAGPipeline

    start_time = time.time()

    if target == "rag":
        pipeline = RAGPipeline()
        res = pipeline.answer_question(question, top_k=top_k)
        latency = time.time() - start_time
        retrieved_docs = res["retrieved_documents"]
        retrieval_context = [doc.get("text", "") for doc in retrieved_docs]
        docs_summary = [
            {"page_id": doc.get("page_id"), "title": doc.get("title"), "similarity_score": doc.get("similarity_score", 1.0)}
            for doc in retrieved_docs
        ]
        return res["answer"], retrieval_context, docs_summary, latency

    if target == "agent":
        from graph.langgraph_agent import LangGraphAgent

        agent = LangGraphAgent()
        res = agent.ask(question, top_k=top_k)
        latency = time.time() - start_time
        retrieval_context = [node.get("text", "") for node in res.get("nodes", []) if isinstance(node, dict)]
        docs_summary = [
            {"page_id": node.get("page_id"), "title": node.get("title"), "similarity_score": 1.0}
            for node in res.get("nodes", []) if isinstance(node, dict)
        ]
        return res["answer"], retrieval_context, docs_summary, latency

    if target in {"api-rag", "api-agent"}:
        import httpx

        endpoint = f"{api_url}/rag/query" if target == "api-rag" else f"{api_url}/agent/langgraph/query"
        res = httpx.post(endpoint, json={"question": question, "top_k": top_k}, timeout=120)
        res.raise_for_status()
        data = res.json()
        latency = time.time() - start_time
        retrieval_context = []
        docs_summary = []
        for entry in data.get("retrieved_documents", []) if target == "api-rag" else data.get("nodes", []):
            page_id = entry.get("page_id")
            docs_summary.append({
                "page_id": page_id,
                "title": entry.get("title"),
                "similarity_score": entry.get("similarity_score", 1.0),
            })
            try:
                page_res = httpx.get(f"{api_url}/documents/{page_id}", timeout=10)
                if page_res.status_code == 200:
                    retrieval_context.append(page_res.json().get("text", ""))
            except Exception:
                pass
        return data.get("answer", ""), retrieval_context, docs_summary, latency

    raise ValueError(f"Unknown target: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OPIK validation against RAG and LangGraph targets.")
    parser.add_argument("--target", type=str, default="rag", choices=["rag", "agent", "api-rag", "api-agent"], help="Evaluation target.")
    parser.add_argument("--dataset", type=str, default="opik", help="OPIK dataset folder name under evals/.")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000", help="FastAPI backend base URL.")
    parser.add_argument("--top-k", type=int, default=3, help="Top K documents to retrieve.")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory to write validation reports.")
    parser.add_argument("--use-heuristics", action="store_true", help="Force heuristic fallback evaluation without LLM judge calls.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    dataset_path = base_dir / "evals" / args.dataset / f"{args.dataset}_cases.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"OPIK dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)

    if isinstance(cases, dict) and "cases" in cases:
        cases = cases["cases"]

    print("=" * 70)
    print("      OPIK MULTI-AGENT VALIDATION RUNNER")
    print("=" * 70)
    print(f"Target:      {args.target}")
    print(f"Dataset:     {dataset_path}")
    print(f"Top K:       {args.top_k}")
    print(f"API URL:     {args.api_url}")
    print("=" * 70)

    if settings.confident_api_key:
        os.environ["CONFIDENT_API_KEY"] = settings.confident_api_key
        print("Confident AI API Key is configured.")

    service = OPIKValidationService()
    if args.use_heuristics:
        service.deep_eval.judge = None
        print("Forcing heuristic-only evaluation.")

    run_id = f"opik_{args.target}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S') }"
    results: List[Dict[str, Any]] = []
    total_passed = 0
    start_time = time.time()

    for index, case in enumerate(cases, start=1):
        question = case.get("input", "")
        expected_output = case.get("expected_output")
        case_id = case.get("case_id", f"opik_case_{index}")

        print(f"[{index}/{len(cases)}] Evaluating case {case_id}")

        try:
            answer, retrieval_context, docs_summary, latency = run_query(
                target=args.target,
                question=question,
                top_k=args.top_k,
                api_url=args.api_url,
            )

            metric_results = service.evaluate_answer(
                question=question,
                answer=answer,
                retrieval_context=retrieval_context,
                expected_output=expected_output,
                case_id=case_id,
                dataset_name=args.dataset,
                run_id=run_id,
            )

            all_passed = all(value.get("passed", False) for value in metric_results.values())
            if all_passed:
                total_passed += 1

            results.append({
                "case_id": case_id,
                "question": question,
                "expected_output": expected_output,
                "answer": answer,
                "retrieval_context": retrieval_context,
                "retrieved_documents": docs_summary,
                "metric_results": metric_results,
                "latency_seconds": latency,
                "all_passed": all_passed,
            })

        except Exception as exc:
            print(f"  Error evaluating case {case_id}: {exc}")
            results.append({
                "case_id": case_id,
                "question": question,
                "error": str(exc),
                "all_passed": False,
            })

    total_duration = time.time() - start_time
    summary = {
        "run_id": run_id,
        "target": args.target,
        "dataset": args.dataset,
        "total_cases": len(cases),
        "passed_cases": total_passed,
        "pass_rate": float(total_passed) / len(cases) if cases else 0.0,
        "duration_seconds": total_duration,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"opik_validation_{run_id}.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": results}, handle, indent=2)

    print("=" * 70)
    print(f"Validation complete. Report written to: {report_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
