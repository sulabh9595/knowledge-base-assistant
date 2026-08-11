from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.deepeval_service import DeepEvalService
from app.services.deepeval_tracing import DeepEvalTracingService


class OPIKValidationService:
    def __init__(self, judge_model_name: Optional[str] = None) -> None:
        self.deep_eval = DeepEvalService(judge_model_name=judge_model_name)
        self.tracing_service = self.deep_eval.tracing_service if hasattr(self.deep_eval, "tracing_service") else DeepEvalTracingService()
        self.metrics_to_run = [
            "faithfulness",
            "answer_relevancy",
            "contextual_precision",
            "contextual_recall",
            "hallucination",
            "toxicity",
            "bias",
            "refusal_quality",
            "citation_quality",
        ]

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        retrieval_context: List[str],
        expected_output: Optional[str] = None,
        case_id: str = "unknown",
        dataset_name: str = "opik",
        run_id: str = "unknown",
        parent_span: Optional[Any] = None,
    ) -> Dict[str, Any]:
        return self.deep_eval.evaluate_case(
            question=question,
            answer=answer,
            retrieval_context=retrieval_context,
            expected_output=expected_output,
            metrics_to_run=self.metrics_to_run,
            case_id=case_id,
            dataset_name=dataset_name,
            run_id=run_id,
            parent_span=parent_span,
        )

    def validate_rag_response(
        self,
        question: str,
        rag_result: Dict[str, Any],
        expected_output: Optional[str] = None,
        case_id: str = "unknown",
        dataset_name: str = "opik",
        run_id: str = "unknown",
        parent_span: Optional[Any] = None,
    ) -> Dict[str, Any]:
        answer = rag_result.get("answer", "")
        retrieval_context = [doc.get("text", "") for doc in rag_result.get("retrieved_documents", [])]
        return self.evaluate_answer(
            question=question,
            answer=answer,
            retrieval_context=retrieval_context,
            expected_output=expected_output,
            case_id=case_id,
            dataset_name=dataset_name,
            run_id=run_id,
            parent_span=parent_span,
        )

    def validate_langgraph_response(
        self,
        question: str,
        agent_result: Dict[str, Any],
        expected_output: Optional[str] = None,
        case_id: str = "unknown",
        dataset_name: str = "opik",
        run_id: str = "unknown",
        parent_span: Optional[Any] = None,
    ) -> Dict[str, Any]:
        answer = agent_result.get("answer", "")
        retrieval_context = [node.get("text", "") for node in agent_result.get("nodes", []) if isinstance(node, dict)]
        return self.evaluate_answer(
            question=question,
            answer=answer,
            retrieval_context=retrieval_context,
            expected_output=expected_output,
            case_id=case_id,
            dataset_name=dataset_name,
            run_id=run_id,
            parent_span=parent_span,
        )

    def load_cases(self, dataset_path: Path) -> List[Dict[str, Any]]:
        if not dataset_path.exists():
            raise FileNotFoundError(f"OPIK dataset not found: {dataset_path}")

        with dataset_path.open("r", encoding="utf-8") as handle:
            cases = json.load(handle)

        if isinstance(cases, dict) and "cases" in cases:
            return cases["cases"]

        if isinstance(cases, list):
            return cases

        raise ValueError("OPIK dataset must be a list of cases or an object with a 'cases' key.")

    def summarize_results(self, eval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        totals = {metric: 0.0 for metric in self.metrics_to_run}
        counts = {metric: 0 for metric in self.metrics_to_run}
        passed_cases = 0

        for result in eval_results:
            if result.get("all_passed"):
                passed_cases += 1
            metric_results = result.get("metric_results", {})
            for metric, metric_data in metric_results.items():
                totals[metric] += metric_data.get("score", 0.0)
                counts[metric] += 1

        average_scores = {
            metric: (totals[metric] / counts[metric]) if counts[metric] else 0.0
            for metric in self.metrics_to_run
        }

        return {
            "total_cases": len(eval_results),
            "passed_cases": passed_cases,
            "pass_rate": float(passed_cases) / len(eval_results) if eval_results else 0.0,
            "average_scores": average_scores,
        }


def load_opik_dataset(dataset_name: str = "opik") -> List[Dict[str, Any]]:
    base_dir = Path(__file__).resolve().parents[2]
    dataset_path = base_dir / "evals" / dataset_name / f"{dataset_name}_cases.json"
    service = OPIKValidationService()
    return service.load_cases(dataset_path)
