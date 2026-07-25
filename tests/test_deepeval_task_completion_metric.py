from app.services.deepeval_service import evaluate_task_completion_demo


def test_evaluate_task_completion_demo_returns_structured_result() -> None:
    result = evaluate_task_completion_demo(
        task="Summarize the company policy on document submission.",
        response="The policy requires the user to submit the document within seven days.",
        context="The company policy says documents must be submitted within seven days.",
    )

    assert result["task_completed"] is True
    assert result["score"] >= 0.0
    assert result["score"] <= 1.0
    assert "reason" in result
    assert result["metric_name"] == "TaskCompletionMetric"


def test_evaluate_task_completion_demo_with_ollama_flag() -> None:
    """Test with Ollama flag set (will fall back gracefully if Ollama unavailable)."""
    result = evaluate_task_completion_demo(
        task="What is the document submission deadline?",
        response="Seven days from the request date.",
        context="Documents must be submitted within seven days.",
        use_ollama=True,
    )

    assert result["task_completed"] is not None
    assert result["score"] >= 0.0
    assert result["score"] <= 1.0
    assert "reason" in result
    assert result["metric_name"] == "TaskCompletionMetric"


def test_evaluate_task_completion_demo_low_score_case() -> None:
    """Test with misaligned response to detect incomplete task."""
    result = evaluate_task_completion_demo(
        task="What is the submission deadline?",
        response="The office is closed on weekends.",
        context="Documents must be submitted within seven days.",
    )

    assert result["task_completed"] is False or result["score"] < 0.7
    assert result["score"] >= 0.0
    assert result["score"] <= 1.0
    assert result["metric_name"] == "TaskCompletionMetric"

