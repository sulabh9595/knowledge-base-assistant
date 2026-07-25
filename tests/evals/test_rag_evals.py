import json
from pathlib import Path
import pytest

from app.rag.pipeline import RAGPipeline
from app.services.deepeval_service import DeepEvalService

# Load cases for parametrization
dataset_path = Path(__file__).resolve().parents[2] / "evals/datasets/knowledge_base_cases.json"
if dataset_path.exists():
    with dataset_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
else:
    cases = []


@pytest.fixture(scope="module")
def deepeval_service() -> DeepEvalService:
    return DeepEvalService()


@pytest.fixture(scope="module")
def rag_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline()
    # Ingest if empty
    try:
        count = pipeline.vector_store.store._collection.count()
    except Exception:
        count = 0
    if count == 0:
        doc_store = Path(__file__).resolve().parents[2] / "memory/documents.json"
        if doc_store.exists():
            with doc_store.open("r", encoding="utf-8") as f:
                documents = json.load(f)
            pipeline.ingest_documents(documents)
    return pipeline


@pytest.mark.parametrize("case_index, case_data", list(enumerate(cases)))
def test_rag_case_quality(
    rag_pipeline: RAGPipeline,
    deepeval_service: DeepEvalService,
    case_index: int,
    case_data: dict,
) -> None:
    question = case_data["input"]
    expected_output = case_data.get("expected_output")
    
    # Get RAG answer
    rag_result = rag_pipeline.answer_question(question)
    answer = rag_result["answer"]
    retrieval_context = [doc["text"] for doc in rag_result["retrieved_documents"]]
    
    # Evaluate case
    eval_results = deepeval_service.evaluate_case(
        question=question,
        answer=answer,
        retrieval_context=retrieval_context,
        expected_output=expected_output,
        case_id=f"test_case_{case_index}",
        dataset_name="pytest_goldens",
    )
    
    errors = []
    
    # Quality Gates Assertions
    if "faithfulness" in eval_results:
        f_score = eval_results["faithfulness"]["score"]
        # Threshold: Faithfulness >= 0.85 (heuristic has higher base, so this matches nicely)
        if f_score < 0.85:
            errors.append(
                f"Faithfulness score {f_score:.3f} is below threshold 0.85. "
                f"Reason: {eval_results['faithfulness']['reason']}"
            )
            
    if "answer_relevancy" in eval_results:
        ar_score = eval_results["answer_relevancy"]["score"]
        # Threshold: Answer Relevancy >= 0.80
        if ar_score < 0.80:
            errors.append(
                f"Answer Relevancy score {ar_score:.3f} is below threshold 0.80. "
                f"Reason: {eval_results['answer_relevancy']['reason']}"
            )
            
    if "contextual_precision" in eval_results:
        cp_score = eval_results["contextual_precision"]["score"]
        # Threshold: Contextual Precision >= 0.80
        if cp_score < 0.80:
            errors.append(
                f"Contextual Precision score {cp_score:.3f} is below threshold 0.80. "
                f"Reason: {eval_results['contextual_precision']['reason']}"
            )
            
    if "contextual_recall" in eval_results:
        cr_score = eval_results["contextual_recall"]["score"]
        # Threshold: Contextual Recall >= 0.75
        if cr_score < 0.75:
            errors.append(
                f"Contextual Recall score {cr_score:.3f} is below threshold 0.75. "
                f"Reason: {eval_results['contextual_recall']['reason']}"
            )
            
    if "hallucination" in eval_results:
        # In deepeval, Hallucination is hallucination rate (0.0 means no hallucination, which is good)
        # Threshold: Hallucination <= 0.30
        h_score = eval_results["hallucination"]["score"]
        if h_score > 0.30:
            errors.append(
                f"Hallucination score {h_score:.3f} is above threshold 0.30. "
                f"Reason: {eval_results['hallucination']['reason']}"
            )
            
    assert not errors, f"Quality gate failures for case index {case_index} (Question: '{question}'):\n" + "\n".join(errors)
