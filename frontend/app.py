# Creator: Sulabh Bansod
# Description: Streamlit web application frontend.
# Use: Provides a user interface to run health checks, ingest pages, and query models.

import os
import streamlit as st
import httpx


st.set_page_config(page_title="Knowledge Base Assistant UI", layout="wide")

default_backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
BASE_URL = st.sidebar.text_input("Backend URL", default_backend_url)

st.sidebar.markdown("---")
enable_tts = st.sidebar.checkbox("🔊 Enable Text-to-Speech Output", value=True)
st.sidebar.write("Use this UI to ingest Confluence spaces, query RAG / LangGraph agents, and synthesize speech.")


def post_json(path: str, payload: dict) -> dict:
    url = f"{BASE_URL}{path}"
    with httpx.Client(timeout=120) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def post_file(path: str, files: dict) -> dict:
    url = f"{BASE_URL}{path}"
    with httpx.Client(timeout=120) as client:
        response = client.post(url, files=files)
        response.raise_for_status()
        return response.json()


def get_json(path: str) -> dict:
    url = f"{BASE_URL}{path}"
    with httpx.Client(timeout=30) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def render_audio_player(result: dict):
    """Renders HTML5 st.audio player if audio_base64 is present in API response."""
    audio_b64 = result.get("audio_base64")
    if audio_b64:
        try:
            import base64
            audio_bytes = base64.b64decode(audio_b64)
            st.subheader("🔊 Audio Output")
            st.audio(audio_bytes, format="audio/mp3")
        except Exception as exc:
            st.warning(f"Could not render audio playback: {exc}")


st.title("Knowledge Base Assistant")
st.markdown("A simple Streamlit UI for document ingestion, RAG querying, LangGraph agent reasoning, and Text-to-Speech synthesis.")

tabs = st.tabs(["Health", "Document & Audio Ingestion", "RAG Query", "LangGraph Agent", "Text-to-Audio (TTS)"])

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
    st.header("Document & Audio Ingestion")

    ingest_type = st.radio("Select Ingestion Method", ["Confluence Space", "Local File Upload", "Audio Recording / Meeting Ingestion"])

    if ingest_type == "Confluence Space":
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

    elif ingest_type == "Local File Upload":
        uploaded_file = st.file_uploader(
            "Choose a PDF, Word, Text, or Audio file",
            type=["pdf", "docx", "txt", "md", "mp3", "wav", "m4a", "ogg", "flac"]
        )
        if st.button("Upload and Ingest File"):
            if not uploaded_file:
                st.warning("Please select a file first.")
            else:
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    result = post_file("/ingest/file", files)
                    st.success(f"Ingested file '{result.get('title')}' successfully!")
                    st.json(result)
                except httpx.HTTPError as exc:
                    st.error(f"File ingestion failed: {exc}")

    elif ingest_type == "Audio Recording / Meeting Ingestion":
        st.subheader("🎙️ Local Audio Recording Ingestion")
        audio_file = st.file_uploader(
            "Upload Meeting Audio / Voice Note (.mp3, .wav, .m4a, .ogg, .flac)",
            type=["mp3", "wav", "m4a", "ogg", "flac", "aac"],
            key="ingest_audio_file"
        )
        gen_summary = st.checkbox("Generate Local Executive Summary & Action Items (Qwen3:8b)", value=True)

        if audio_file:
            st.audio(audio_file)
            if st.button("Transcribe & Ingest Audio"):
                try:
                    files = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type or "audio/wav")}
                    path = f"/ingest/audio?generate_summary={str(gen_summary).lower()}"
                    result = post_file(path, files)
                    st.success(f"Successfully transcribed and ingested '{result.get('title')}'!")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Duration", f"{result.get('duration_seconds')}s")
                    col2.metric("Language", result.get('language', 'en').upper())
                    col3.metric("Word Count", result.get('word_count'))

                    if result.get("summary"):
                        st.subheader("Executive Summary")
                        st.markdown(result.get("summary"))
                    st.subheader("Ingestion Details")
                    st.json(result)
                except httpx.HTTPError as exc:
                    st.error(f"Audio ingestion failed: {exc}")


with tabs[2]:
    st.header("RAG Query")
    query_mode = st.radio("Select Query Mode", ["Text Question", "Voice / Audio Question"], horizontal=True, key="rag_query_mode")

    if query_mode == "Text Question":
        question = st.text_area("Question", key="rag_question")
        top_k = st.slider("Top K documents", min_value=1, max_value=10, value=3)
        if st.button("Query RAG"):
            if not question.strip():
                st.warning("Enter a question to query the RAG endpoint.")
            else:
                try:
                    result = post_json("/rag/query", {"question": question, "top_k": top_k, "include_audio": enable_tts})
                    st.success("RAG answer generated")
                    st.subheader("Answer")
                    st.markdown(result.get("answer", ""))
                    render_audio_player(result)
                    st.subheader("Retrieved Documents")
                    st.json(result.get("retrieved_documents", []))
                except httpx.HTTPError as exc:
                    st.error(f"RAG query failed: {exc}")

    else:
        st.subheader("🎤 Voice Question")
        audio_prompt = None
        if hasattr(st, "audio_input"):
            audio_prompt = st.audio_input("Record your voice question", key="rag_audio_input")
        
        audio_file = st.file_uploader("Or upload an audio query file (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a", "ogg", "flac"], key="rag_audio_file")
        top_k = st.slider("Top K documents", min_value=1, max_value=10, value=3, key="rag_audio_top_k")

        selected_audio = audio_prompt or audio_file

        if selected_audio:
            st.audio(selected_audio)
            if st.button("Submit Voice Query"):
                try:
                    files = {"file": (getattr(selected_audio, "name", "voice_query.wav"), selected_audio.getvalue(), "audio/wav")}
                    result = post_file(f"/rag/query/audio?top_k={top_k}", files)
                    st.info(f"Transcribed Question: **{result.get('transcribed_question')}**")
                    st.success("RAG Answer Generated")
                    st.subheader("Answer")
                    st.markdown(result.get("answer", ""))
                    render_audio_player(result)
                    st.subheader("Retrieved Documents")
                    st.json(result.get("retrieved_documents", []))
                except httpx.HTTPError as exc:
                    st.error(f"Voice RAG query failed: {exc}")

with tabs[3]:
    st.header("LangGraph Agent Query")
    langgraph_query_mode = st.radio("Select Query Mode", ["Text Question", "Voice / Audio Question"], horizontal=True, key="langgraph_query_mode")

    if langgraph_query_mode == "Text Question":
        question = st.text_area("Question", key="langgraph_question")
        top_k = st.slider("Top K nodes", min_value=1, max_value=10, value=3, key="langgraph_top_k")
        if st.button("Query LangGraph Agent"):
            if not question.strip():
                st.warning("Enter a question to query the LangGraph agent.")
            else:
                try:
                    result = post_json(
                        "/agent/langgraph/query",
                        {"question": question, "top_k": top_k, "include_audio": enable_tts},
                    )
                    st.success("LangGraph answer generated")
                    st.subheader("Answer")
                    st.write(result.get("answer", ""))
                    render_audio_player(result)
                    st.subheader("Graph Nodes")
                    st.json(result.get("nodes", []))
                except httpx.HTTPError as exc:
                    st.error(f"LangGraph query failed: {exc}")

    else:
        st.subheader("🎤 Voice Question (LangGraph Agent)")
        lg_audio_prompt = None
        if hasattr(st, "audio_input"):
            lg_audio_prompt = st.audio_input("Record your voice question", key="langgraph_audio_input")
        
        lg_audio_file = st.file_uploader("Or upload an audio query file (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a", "ogg", "flac"], key="langgraph_audio_file")
        top_k = st.slider("Top K nodes", min_value=1, max_value=10, value=3, key="langgraph_audio_top_k")

        selected_lg_audio = lg_audio_prompt or lg_audio_file

        if selected_lg_audio:
            st.audio(selected_lg_audio)
            if st.button("Submit Voice Query to Agent"):
                try:
                    files = {"file": (getattr(selected_lg_audio, "name", "voice_query.wav"), selected_lg_audio.getvalue(), "audio/wav")}
                    result = post_file(f"/agent/langgraph/query/audio?top_k={top_k}", files)
                    st.info(f"Transcribed Question: **{result.get('transcribed_question')}**")
                    st.success("LangGraph Answer Generated")
                    st.subheader("Answer")
                    st.markdown(result.get("answer", ""))
                    render_audio_player(result)
                    st.subheader("Retrieved Documents & Nodes")
                    st.json(result.get("retrieved_documents", []))
                except httpx.HTTPError as exc:
                    st.error(f"Voice LangGraph query failed: {exc}")

with tabs[4]:
    st.header("🔊 Text-to-Audio (TTS) Synthesizer")
    st.markdown("Convert arbitrary text into spoken audio using Local Kokoro Neural TTS, Microsoft Edge TTS, Azure, or system fallbacks.")

    provider_selection = st.selectbox(
        "Select TTS Engine Provider",
        ["kokoro", "edge-tts", "azure", "gTTS", "say", "pyttsx3"],
        key="tts_provider_select"
    )

    if provider_selection == "kokoro":
        voice_options = ["af_heart", "af_bella", "am_adam", "am_michael", "bf_emma"]
    elif provider_selection == "azure":
        voice_options = ["en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"]
    else:
        voice_options = ["en-US-AvaNeural", "en-US-AndrewNeural", "en-GB-SoniaNeural"]

    text_to_speak = st.text_area("Text to Synthesize", "Welcome to the Enterprise Agentic Knowledge Platform. How can I assist you today?", key="tts_input_text")
    voice_selection = st.selectbox("Select Voice", voice_options, key="tts_voice_select")
    speed_selection = st.slider("Speaking Speed Multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.1, key="tts_speed_select")

    if st.button("Synthesize Speech"):
        if not text_to_speak.strip():
            st.warning("Please enter text to synthesize.")
        else:
            try:
                payload = {
                    "text": text_to_speak,
                    "voice": voice_selection,
                    "provider": provider_selection,
                    "speed": speed_selection
                }
                result = post_json("/tts/synthesize", payload)
                st.success("Speech synthesized successfully!")
                render_audio_player(result)
            except httpx.HTTPError as exc:
                st.error(f"Text-to-Speech synthesis failed: {exc}")

