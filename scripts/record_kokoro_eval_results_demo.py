"""
Automated Video Demo: Full Application Overview, Kokoro Neural TTS, and Comprehensive Evaluated Metrics Results.
Saves a new video with dedicated focus on live benchmark results and sample-level metric breakdowns.
"""

import os
import time
from playwright.sync_api import sync_playwright

def record_full_eval_demo():
    video_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    print(f"[1/7] Initializing browser & video recorder in {video_dir}...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            record_video_dir=video_dir,
            record_video_size={"width": 1280, "height": 800},
            viewport={"width": 1280, "height": 800}
        )
        
        page = context.new_page()
        
        # --- 1. APPLICATION OVERVIEW ---
        print("[2/7] Opening Streamlit Application Overview at http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle")
        time.sleep(2.5)
        
        # Document & Audio Ingestion Tab
        print("Showcasing Ingestion Tab...")
        page.locator('button[data-baseweb="tab"]:has-text("Document & Audio Ingestion")').first.click()
        time.sleep(2)
        
        # RAG Query Tab
        print("Showcasing RAG Query Tab...")
        page.locator('button[data-baseweb="tab"]:has-text("RAG Query")').first.click()
        time.sleep(2)
        
        # LangGraph Agent Tab
        print("Showcasing LangGraph Agent Tab...")
        page.locator('button[data-baseweb="tab"]:has-text("LangGraph Agent")').first.click()
        time.sleep(2)
        
        # --- 2. KOKORO TTS IMPLEMENTATION ---
        print("[3/7] Navigating to Text-to-Audio (TTS) Tab...")
        tts_tab = page.locator('button[data-baseweb="tab"]:has-text("Text-to-Audio")').first
        tts_tab.click()
        time.sleep(2)
        
        # Focus on TTS controls
        page.mouse.wheel(0, 100)
        time.sleep(1)
        
        # Type realistic demo prompt
        print("[4/7] Entering Kokoro neural synthesis demo prompt...")
        prompt_textarea = page.locator('textarea:visible').first
        prompt_textarea.scroll_into_view_if_needed()
        time.sleep(0.8)
        
        prompt_textarea.fill(
            "Welcome to the Enterprise Agentic Knowledge Platform. "
            "This neural voice is synthesized locally using the Kokoro-82M TTS engine, "
            "delivering 100% offline privacy, zero cloud egress, and high acoustic naturalness."
        )
        time.sleep(1.5)
        page.keyboard.press("Tab")
        time.sleep(1)
        
        # Click Synthesize Speech
        print("[5/7] Clicking 'Synthesize Speech'...")
        synth_btn = page.locator('button:has-text("Synthesize Speech"):visible').first
        synth_btn.scroll_into_view_if_needed()
        time.sleep(0.8)
        synth_btn.click()
        
        # Wait for success notification and audio playback rendering
        try:
            page.locator('text="Speech synthesized successfully!"').wait_for(timeout=25000)
            print("Speech synthesis completed successfully!")
        except Exception as e:
            print(f"Wait note: {e}")
            
        time.sleep(2.5)
        
        # Scroll down smoothly to show the rendered audio player & controls
        page.mouse.wheel(0, 250)
        time.sleep(4)
        
        # --- 3. QUALITY & EVALUATION RESULTS DASHBOARD ---
        print("[6/7] Navigating to DeepEval & Quality Dashboard (TTS Evaluation Metrics Results)...")
        dash_tab = page.locator('button[data-baseweb="tab"]:has-text("DeepEval & Quality")').first
        dash_tab.click()
        time.sleep(2.5)
        
        # Switch to inner tab: "🔊 TTS Benchmarks & Quality"
        inner_tts_tab = page.locator('button[data-baseweb="tab"]:has-text("TTS Benchmarks")').first
        if inner_tts_tab.count() > 0:
            inner_tts_tab.click()
            time.sleep(2.5)
            
        # Top KPI Summary Cards (Pass Rate 100%, Avg Latency, RTF, WER 0.00%)
        print("Highlighting Top KPI Metric Cards...")
        time.sleep(4)
        
        # Scroll to Provider Benchmark Comparison Table
        print("Displaying Provider Benchmark Comparison Table...")
        page.mouse.wheel(0, 220)
        time.sleep(3.5)
        
        # Scroll to Sample-Level Evaluated Metrics Details Table (Live Tested Results)
        print("Displaying Sample-Level Tested Metrics Details (WER, CER, SNR, RTF, F0 Pitch, WPM)...")
        page.mouse.wheel(0, 320)
        time.sleep(5.5)  # Ample time to read all columns and rows
        
        # Scroll down to the Core Metrics Taxonomy Expander
        print("Displaying Core TTS Evaluation Metrics Taxonomy (.skills/tts-evaluation.md)...")
        page.mouse.wheel(0, 350)
        time.sleep(4.5)
        
        # Open Raw JSON Payload expander to show complete evaluated metrics data
        raw_json_expander = page.locator('text="View Raw Evaluated Benchmark JSON Payload"')
        if raw_json_expander.count() > 0:
            raw_json_expander.first.click()
            time.sleep(3)
            page.mouse.wheel(0, 250)
            time.sleep(2.5)
        
        # Scroll back up smoothly
        page.mouse.wheel(0, -900)
        time.sleep(1.5)
        
        # Pytest Results Inspector Tab
        print("[7/7] Navigating to Pytest Test Suite Results...")
        pytest_tab = page.locator('button[data-baseweb="tab"]:has-text("Pytest")').first
        pytest_tab.click()
        time.sleep(2.5)
        
        # Scroll down to show passing Kokoro and TTS test cases (26/26 passed, 100% pass rate)
        page.mouse.wheel(0, 300)
        time.sleep(3.5)
        page.mouse.wheel(0, -300)
        time.sleep(1.5)
        
        # Return cleanly to TTS tab to close the demo
        tts_tab.click()
        time.sleep(2.5)
        
        # Finalize and save video
        print("Finalizing video...")
        page_video = page.video
        video_temp_path = page_video.path() if page_video else None
        
        page.close()
        context.close()
        browser.close()
        
        final_video = os.path.join(video_dir, "kokoro_tts_eval_results_demo.webm")
        if video_temp_path and os.path.exists(video_temp_path):
            if os.path.exists(final_video):
                os.remove(final_video)
            os.rename(video_temp_path, final_video)
            print(f"SUCCESS: New video recorded and saved at: {final_video}")
            return final_video
            
        return video_dir

if __name__ == "__main__":
    record_full_eval_demo()
