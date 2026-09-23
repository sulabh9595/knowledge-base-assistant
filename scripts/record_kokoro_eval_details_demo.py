"""
Automated Video Demo: Full Walkthrough with Prominent Display of 'Evaluated Metrics Details Table (Live Results):'
"""

import os
import time
from playwright.sync_api import sync_playwright

def record_eval_details_demo():
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
        
        # Ingestion Tab
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
        time.sleep(3.5)
        
        # --- 3. QUALITY & EVALUATION RESULTS DASHBOARD ---
        print("[6/7] Navigating to DeepEval & Quality Dashboard...")
        dash_tab = page.locator('button[data-baseweb="tab"]:has-text("DeepEval & Quality")').first
        dash_tab.click()
        time.sleep(2.5)
        
        # Switch to inner tab: "🔊 TTS Benchmarks & Quality"
        inner_tts_tab = page.locator('button[data-baseweb="tab"]:has-text("TTS Benchmarks")').first
        if inner_tts_tab.count() > 0:
            inner_tts_tab.click()
            time.sleep(2.5)
            
        # Top KPI Summary Cards
        print("Highlighting Top KPI Metric Cards (Pass Rate 100%, WER 0.00%, RTF 0.207)...")
        time.sleep(3)
        
        # PROMINENT FOCUS: Evaluated Metrics Details Table (Live Results):
        print("Centering and displaying 'Evaluated Metrics Details Table (Live Results):'...")
        results_header = page.locator('text="Evaluated Metrics Details Table (Live Results):"')
        if results_header.count() > 0:
            results_header.first.scroll_into_view_if_needed()
        else:
            page.mouse.wheel(0, 200)
        time.sleep(6)  # Generous duration for the viewer to read all the metrics table columns
        
        # Scroll down to Provider Benchmark Comparison Table
        print("Displaying Provider Benchmark Comparison Table...")
        comp_header = page.locator('text="Provider Benchmark Comparison"')
        if comp_header.count() > 0:
            comp_header.first.scroll_into_view_if_needed()
        else:
            page.mouse.wheel(0, 250)
        time.sleep(3.5)
        
        # Scroll down to Core Metrics Taxonomy Expander
        print("Displaying Core TTS Evaluation Metrics Taxonomy (.skills/tts-evaluation.md)...")
        tax_header = page.locator('text="Core TTS Evaluation Metrics Taxonomy"')
        if tax_header.count() > 0:
            tax_header.first.scroll_into_view_if_needed()
        else:
            page.mouse.wheel(0, 300)
        time.sleep(4)
        
        # Open Raw JSON Payload expander
        raw_json = page.locator('text="View Raw Evaluated Benchmark JSON Payload"')
        if raw_json.count() > 0:
            raw_json.first.click()
            time.sleep(2)
            page.mouse.wheel(0, 200)
            time.sleep(2)
        
        # Scroll back up smoothly
        page.mouse.wheel(0, -900)
        time.sleep(1.5)
        
        # Pytest Results Inspector Tab
        print("[7/7] Navigating to Pytest Test Suite Results...")
        pytest_tab = page.locator('button[data-baseweb="tab"]:has-text("Pytest")').first
        pytest_tab.click()
        time.sleep(2.5)
        
        # Show passing Kokoro and TTS test cases (26/26 passed, 100% pass rate)
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
        
        final_video = os.path.join(video_dir, "kokoro_tts_eval_details_demo.webm")
        if video_temp_path and os.path.exists(video_temp_path):
            if os.path.exists(final_video):
                os.remove(final_video)
            os.rename(video_temp_path, final_video)
            print(f"SUCCESS: Video recorded and saved at: {final_video}")
            return final_video
            
        return video_dir

if __name__ == "__main__":
    record_eval_details_demo()
