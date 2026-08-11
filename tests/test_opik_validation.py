from __future__ import annotations

import json
from pathlib import Path

from app.services.opik_validation_service import OPIKValidationService


def test_opik_validation_service_initializes() -> None:
    service = OPIKValidationService()
    assert service is not None
    assert hasattr(service, "evaluate_answer")
    assert "faithfulness" in service.metrics_to_run


def test_opik_validation_service_evaluates_case() -> None:
    service = OPIKValidationService()
    service.deep_eval.judge = None

    question = "What is the project mission?"
    answer = "The project mission is to provide a grounded knowledge base assistant."
    retrieval_context = [
        "The Knowledge Base Application is built to answer user questions using local documents and agents.",
        "Project mission statements should focus on accuracy, reliability, and enterprise readiness.",
    ]
    expected_output = "Provide grounded answers using internal documents."

    result = service.evaluate_answer(
        question=question,
        answer=answer,
        retrieval_context=retrieval_context,
        expected_output=expected_output,
        case_id="unit_test_01",
        dataset_name="pytest_opik",
        run_id="pytest_run",
    )

    assert isinstance(result, dict)
    assert "faithfulness" in result
    assert "answer_relevancy" in result
    assert result["faithfulness"]["score"] >= 0.0


def test_opik_dataset_loader_reads_list() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    dataset_path = base_dir / "evals" / "opik" / "opik_cases.json"
    if not dataset_path.exists():
        return

    service = OPIKValidationService()
    cases = service.load_cases(dataset_path)
    assert isinstance(cases, list)
    if cases:
        assert "input" in cases[0]
