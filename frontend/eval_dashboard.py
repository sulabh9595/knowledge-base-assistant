# Creator: Sulabh Bansod
# Description: Streamlit dashboard for evaluating DeepEval results, trends, and history.
# Use: Provides an interactive interface to view run metrics, history, compare runs, and generate synthetic data.

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st

# Streamlit prepends this script's directory (frontend/) to sys.path. That
# makes `frontend/app.py` discoverable as a top-level `app` module, which
# shadows the real project package `app/` (with app.config, app.services, ...)
# and breaks any `from app.config...` import (e.g. inside the synthetic data
# generator we import below). Fix it before any app imports run.
root_dir = str(Path(__file__).resolve().parents[1])
frontend_dir = str(Path(__file__).resolve().parent)

# 1. Remove only the frontend directory from sys.path so `app` does not
#    resolve to frontend/app.py. Leave `""` and `"."` alone — stripping
#    them corrupts importlib's path finder cache (KeyError: 'evals').
sys.path[:] = [p for p in sys.path if p != frontend_dir]

# 2. Prepend the project root so the real `app/` package wins. If it is
#    already in sys.path, move it to the front.
if root_dir in sys.path:
    sys.path.remove(root_dir)
sys.path.insert(0, root_dir)

# 3. Drop any cached `app` module from sys.modules so a re-import rebinds
#    to the real package.
sys.modules.pop("app", None)

from evals.scripts.generate_synthetic_data import generate_synthetic_data

st.set_page_config(page_title="DeepEval Evaluation Dashboard", layout="wide")

st.title("📊 DeepEval Evaluation Dashboard")
st.markdown("Monitor, analyze, and track the quality, safety, and performance of the RAG pipeline and LangGraph Agent.")

# 1. Setup paths
base_dir = Path(__file__).resolve().parents[1]
results_dir = base_dir / "eval_results"
history_dir = results_dir / "history"
metrics_csv_path = results_dir / "metrics.csv"

# Sidebar controls
st.sidebar.header("Navigation & Configuration")

# Load history
run_files = sorted(history_dir.glob("*.json"), key=os.path.getmtime, reverse=True) if history_dir.exists() else []
run_names = [f.stem for f in run_files]

st.sidebar.subheader("Select Run Details")
selected_run = st.sidebar.selectbox("Select Evaluation Run", ["Latest Run"] + run_names)

# Ingestion stats
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Generation")
num_docs = st.sidebar.slider("Number of docs to sample", min_value=1, max_value=20, value=5)
if st.sidebar.button("Generate Synthetic Data"):
    with st.spinner("Generating synthetic cases..."):
        try:
            generate_synthetic_data(num_docs=num_docs, output_file="evals/datasets/synthetic_cases.json")
            st.sidebar.success("Synthetic dataset generated successfully!")
        except Exception as exc:
            st.sidebar.error(f"Generation failed: {exc}")

# Helper to load a specific run json
from typing import Optional, Dict, Any

def load_run_data(run_name: str) -> Optional[Dict[str, Any]]:
    if run_name == "Latest Run":
        latest_file = results_dir / "latest.json"
        if latest_file.exists():
            with latest_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        return None
    
    run_file = history_dir / f"{run_name}.json"
    if run_file.exists():
        with run_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None

run_data = load_run_data(selected_run)

# Setup tabs
tabs = st.tabs(["🎯 Run Overview", "📈 Trends & History", "📋 Detailed Case Analysis", "⚙️ Run Configuration"])

with tabs[0]:
    if run_data is None:
        st.info("No evaluation runs found. Run an evaluation using `python evals/scripts/run_deepeval.py` first.")
    else:
        st.header(f"Run Overview: {run_data.get('run_id')}")
        st.markdown(f"**Timestamp:** {run_data.get('timestamp')} | **Target:** `{run_data.get('target', 'rag')}` | **Dataset:** `{run_data.get('dataset', 'goldens')}`")

        summary = run_data.get("summary", {})
        
        # Top level metrics cards
        col1, col2, col3, col4 = st.columns(4)
        pass_rate = summary.get("pass_rate", 0.0) * 100
        col1.metric("Pass Rate", f"{pass_rate:.1f}%")
        col2.metric("Total Cases", summary.get("total_cases", 0))
        col3.metric("Passed Cases", summary.get("passed_cases", 0))
        col4.metric("Duration", f"{summary.get('total_duration_seconds', 0.0)}s")

        st.markdown("---")

        # Metrics bar chart
        st.subheader("Average Metrics Scores")
        avg_scores = summary.get("avg_metric_scores", {})
        if avg_scores:
            df_scores = pd.DataFrame({
                "Metric": list(avg_scores.keys()),
                "Average Score": list(avg_scores.values())
            })
            st.bar_chart(df_scores.set_index("Metric"))
        else:
            st.write("No average metrics scores recorded for this run.")

with tabs[1]:
    st.header("Evaluation Trends over Time")
    
    if metrics_csv_path.exists():
        df_metrics = pd.read_csv(metrics_csv_path)
        
        if not df_metrics.empty:
            df_metrics["timestamp"] = pd.to_datetime(df_metrics["timestamp"])
            df_metrics = df_metrics.sort_values("timestamp")

            # Metrics history selection
            st.subheader("Pass Rate Trend")
            st.line_chart(df_metrics.set_index("timestamp")["pass_rate"])

            # Metric scores trends
            st.subheader("Key Quality Metrics Trend")
            cols_to_plot = [col for col in df_metrics.columns if col.startswith("avg_")]
            if cols_to_plot:
                st.line_chart(df_metrics.set_index("timestamp")[cols_to_plot])

            # Performance / Duration Trend
            st.subheader("Run Latency Trend")
            st.line_chart(df_metrics.set_index("timestamp")["duration_seconds"])

            st.subheader("All Historical Runs")
            st.dataframe(df_metrics, use_container_width=True)
        else:
            st.info("Metrics history is empty.")
    else:
        st.info("No historical metrics CSV found. Runs will build history over time.")

with tabs[2]:
    if run_data is None:
        st.info("No run data loaded.")
    else:
        st.header("Detailed Case Analysis")
        cases = run_data.get("cases", [])
        
        if not cases:
            st.write("No cases found in this run.")
        else:
            for idx, c in enumerate(cases, 1):
                all_passed = c.get("all_passed", False)
                status_emoji = "🟢 PASS" if all_passed else "🔴 FAIL"
                
                with st.expander(f"Case {idx}: {c.get('question')[:80]}... ({status_emoji})"):
                    st.markdown(f"**Question:** {c.get('question')}")
                    st.markdown(f"**Expected Output:** {c.get('expected_output')}")
                    st.markdown(f"**Actual Output:** {c.get('actual_output')}")
                    st.markdown(f"**Query Type:** `{c.get('query_type', 'factual')}` | **Latency:** `{c.get('latency_seconds', 0.0):.2f}s`")
                    
                    st.markdown("#### Metric Results")
                    results = c.get("metric_results", {})
                    if results:
                        df_res = pd.DataFrame([
                            {
                                "Metric": m,
                                "Score": res.get("score"),
                                "Passed": "Passed" if res.get("passed") else "Failed",
                                "Latency (s)": f"{res.get('latency_seconds', 0.0):.2f}",
                                "Reason": res.get("reason", "")
                            }
                            for m, res in results.items()
                        ])
                        st.table(df_res)
                    elif "error" in c:
                        st.error(f"Error during execution: {c['error']}")
                    else:
                        st.write("No metrics computed.")

with tabs[3]:
    if run_data is None:
        st.info("No run data loaded.")
    else:
        st.header("Run Context & Parameters")
        st.json({
            "run_id": run_data.get("run_id"),
            "timestamp": run_data.get("timestamp"),
            "target": run_data.get("target"),
            "dataset": run_data.get("dataset"),
            "judge_model": run_data.get("judge_model"),
        })
