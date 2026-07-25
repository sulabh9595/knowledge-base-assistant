from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, Iterator, List, Optional
from app.services.langfuse_service import LangfuseService


class DeepEvalTracingService:
    def __init__(self, langfuse_service: Optional[LangfuseService] = None) -> None:
        self.langfuse_service = langfuse_service or LangfuseService()

    @contextlib.contextmanager
    def trace_run(self, dataset_name: str, run_id: str, model_name: str, metrics: List[str]) -> Iterator[Optional[Any]]:
        client = self.langfuse_service._get_client()
        if not client:
            yield None
            return

        metadata = {
            "dataset_name": dataset_name,
            "run_id": run_id,
            "model_name": model_name,
            "metrics_evaluated": metrics,
        }

        # Create a trace for the evaluation run
        with client.start_as_current_span(
            name="deepeval_run",
            input={"dataset_name": dataset_name, "metrics": metrics},
            metadata=metadata
        ) as trace:
            try:
                yield trace
            finally:
                self.langfuse_service.flush()

    @contextlib.contextmanager
    def trace_case(self, parent_trace: Optional[Any], case_id: str, question: str, expected_output: Optional[str] = None) -> Iterator[Optional[Any]]:
        client = self.langfuse_service._get_client()
        if not client:
            yield None
            return

        # Create a span for this case
        with client.start_as_current_span(
            name="case_evaluation",
            input={"question": question, "expected_output": expected_output},
            metadata={"case_id": case_id}
        ) as span:
            try:
                yield span
            finally:
                pass

    @contextlib.contextmanager
    def trace_metric(self, parent_span: Optional[Any], metric_name: str, input_data: Dict[str, Any]) -> Iterator[Optional[Any]]:
        client = self.langfuse_service._get_client()
        if not client:
            yield None
            return

        # Create a child span for the specific metric
        with client.start_as_current_span(
            name=f"metric_{metric_name}",
            input=input_data,
            metadata={"metric_name": metric_name}
        ) as metric_span:
            try:
                yield metric_span
            except Exception as exc:
                if hasattr(metric_span, "update"):
                    try:
                        metric_span.update(output={"error": str(exc)})
                    except Exception:
                        pass
                raise
            finally:
                pass
