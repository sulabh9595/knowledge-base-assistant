import os
import streamlit as st
import httpx


st.set_page_config(page_title="Knowledge Base Assistant UI", layout="wide")

default_backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
BASE_URL = st.sidebar.text_input("Backend URL", default_backend_url)

st.sidebar.markdown("---")
st.sidebar.write("Use this UI to ingest Confluence spaces and query the RAG / LangGraph agents.")


def post_json(path: str, payload: dict) -> dict:
    url = f"{BASE_URL}{path}"
    with httpx.Client(timeout=120) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def get_json(path: str) -> dict:
    url = f"{BASE_URL}{path}"
    with httpx.Client(timeout=30) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


st.title("Knowledge Base Assistant")
st.markdown("A simple Streamlit UI for Confluence ingestion, RAG querying, and LangGraph agent reasoning.")

tabs = st.tabs(["Health", "Confluence Ingestion", "RAG Query", "LangGraph Agent"])

with tabs[0]:
    st.header("Service Health")
    if st.button("Check health"):
        try:
            health = get_json("/health")
            st.success("Backend is available")
            st.json(health)
        except httpx.HTTPError as exc:
            st.error(f"Health check failed: {exc}")

with tabs[1]:
    st.header("Confluence Ingestion")
    space_key = st.text_input("Space Key", key="confluence_space_key")
    if st.button("Ingest Confluence Space"):
        if not space_key:
            st.warning("Enter a Confluence space key before ingesting.")
        else:
            try:
                result = post_json("/ingest/confluence", {"space_key": space_key})
                page_count = result.get("page_count", 0)
                if page_count == 0:
                    st.warning(
                        "No pages were ingested from that space. "
                        "Please verify the space key, your Confluence permissions, and that the space contains pages."
                    )
                else:
                    st.success(f"Ingested {page_count} pages.")
                st.json(result)
            except httpx.HTTPError as exc:
                st.error(f"Confluence ingestion failed: {exc}")

with tabs[2]:
    st.header("RAG Query")
    question = st.text_area("Question", key="rag_question")
    top_k = st.slider("Top K documents", min_value=1, max_value=10, value=3)
    if st.button("Query RAG"):
        if not question.strip():
            st.warning("Enter a question to query the RAG endpoint.")
        else:
            try:
                result = post_json("/rag/query", {"question": question, "top_k": top_k})
                st.success("RAG answer generated")
                st.subheader("Answer")
                st.markdown(result.get("answer", ""))
                st.subheader("Retrieved Documents")
                st.json(result.get("retrieved_documents", []))
            except httpx.HTTPError as exc:
                st.error(f"RAG query failed: {exc}")

with tabs[3]:
    st.header("LangGraph Agent Query")
    question = st.text_area("Question", key="langgraph_question")
    top_k = st.slider("Top K nodes", min_value=1, max_value=10, value=3, key="langgraph_top_k")
    if st.button("Query LangGraph Agent"):
        if not question.strip():
            st.warning("Enter a question to query the LangGraph agent.")
        else:
            try:
                result = post_json(
                    "/agent/langgraph/query",
                    {"question": question, "top_k": top_k},
                )
                st.success("LangGraph answer generated")
                st.subheader("Answer")
                st.write(result.get("answer", ""))
                st.subheader("Graph Nodes")
                st.json(result.get("nodes", []))
            except httpx.HTTPError as exc:
                st.error(f"LangGraph query failed: {exc}")
