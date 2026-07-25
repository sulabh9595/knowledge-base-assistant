import json
from pathlib import Path
import pytest

from graph.langgraph_agent import LangGraphAgent
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
def agent() -> LangGraphAgent:
    agent_instance = LangGraphAgent()
    # Ingest if graph is empty
    if not agent_instance.graph.nodes:
        doc_store = Path(__file__).resolve().parents[2] / "memory/documents.json"
        if doc_store.exists():
            with doc_store.open("r", encoding="utf-8") as f:
                documents = json.load(f)
            agent_instance.ingest_documents(documents)
    return agent_instance


@pytest.mark.parametrize("case_index, case_data", list(enumerate(cases)))
def test_agent_case_quality(
    agent: LangGraphAgent,
    deepeval_service: DeepEvalService,
    case_index: int,
    case_data: dict,
) -> None:
    question = case_data["input"]
    expected_output = case_data.get("expected_output")
    
    # Get Agent answer
    agent_result = agent.ask(question)
    answer = agent_result["answer"]
    
    # Extract retrieval context from graph nodes
    retrieval_context = []
    for node_dict in agent_result["nodes"]:
        page_id = node_dict["page_id"]
        if page_id in agent.graph.nodes:
            retrieval_context.append(agent.graph.nodes[page_id].text)
            
    # Evaluate case
    eval_results = deepeval_service.evaluate_case(
        question=question,
        answer=answer,
        retrieval_context=retrieval_context,
        expected_output=expected_output,
        case_id=f"agent_test_case_{case_index}",
        dataset_name="pytest_agent_goldens",
    )
    
    errors = []
    
    # Quality Gates Assertions
    if "faithfulness" in eval_results:
        f_score = eval_results["faithfulness"]["score"]
        if f_score < 0.85:
            errors.append(
                f"Faithfulness score {f_score:.3f} is below threshold 0.85. "
                f"Reason: {eval_results['faithfulness']['reason']}"
            )
            
    if "answer_relevancy" in eval_results:
        ar_score = eval_results["answer_relevancy"]["score"]
        if ar_score < 0.80:
            errors.append(
                f"Answer Relevancy score {ar_score:.3f} is below threshold 0.80. "
                f"Reason: {eval_results['answer_relevancy']['reason']}"
            )
            
    if "contextual_precision" in eval_results:
        cp_score = eval_results["contextual_precision"]["score"]
        if cp_score < 0.80:
            errors.append(
                f"Contextual Precision score {cp_score:.3f} is below threshold 0.80. "
                f"Reason: {eval_results['contextual_precision']['reason']}"
            )
            
    if "contextual_recall" in eval_results:
        cr_score = eval_results["contextual_recall"]["score"]
        if cr_score < 0.75:
            errors.append(
                f"Contextual Recall score {cr_score:.3f} is below threshold 0.75. "
                f"Reason: {eval_results['contextual_recall']['reason']}"
            )
            
    if "hallucination" in eval_results:
        h_score = eval_results["hallucination"]["score"]
        if h_score > 0.30:
            errors.append(
                f"Hallucination score {h_score:.3f} is above threshold 0.30. "
                f"Reason: {eval_results['hallucination']['reason']}"
            )
            
    assert not errors, f"Quality gate failures for agent case index {case_index} (Question: '{question}'):\n" + "\n".join(errors)
