"""
Automated Video Demo: Full Application Overview, Kokoro Neural TTS,
Dedicated TTS Benchmarks & Quality Dashboard, and Pytest Test Suite Results.
"""

import os
import time
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright
import imageio_ffmpeg


def click_tab(page, tab_text, timeout=10000):
    """Scrolls to top and clicks a tab by text within Streamlit."""
    page.evaluate("window.scrollTo(0, 0)")
    time.sleep(0.4)
    tab = page.locator('[data-baseweb="tab"]').filter(has_text=tab_text).first
    tab.scroll_into_view_if_needed()
    tab.wait_for(state="visible", timeout=timeout)
    tab.click()
    time.sleep(1.5)


def smooth_scroll(page, delta_y, steps=10, interval=0.08):
    """Performs smooth scrolling."""
    step_y = delta_y / steps
    for _ in range(steps):
        page.mouse.wheel(0, step_y)
        time.sleep(interval)


def record_video():
    root_dir = Path(__file__).resolve().parents[1]
    video_dir = root_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Clean old temp webms
    for old_webm in video_dir.glob("*.webm"):
        try:
            old_webm.unlink()
        except Exception:
            pass

    print(f"[1/9] Launching browser and video recorder in {video_dir}...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            record_video_dir=str(video_dir),
            record_video_size={"width": 1280, "height": 800},
            viewport={"width": 1280, "height": 800}
        )

        page = context.new_page()

        # ==========================================
        # 1. APPLICATION OVERVIEW & SERVICE HEALTH
        # ==========================================
        print("[2/9] Opening Application at http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle")
        time.sleep(2.5)

        # Health Tab
        print("Showcasing Service Health Tab...")
        click_tab(page, "Health")
        time.sleep(1.0)
        health_btn = page.locator('button:has-text("Check health")').first
        if health_btn.count() > 0:
            health_btn.click()
            time.sleep(1.8)

        # ==========================================
        # 2. INGESTION TAB
        # ==========================================
        print("Showcasing Document & Audio Ingestion Tab...")
        click_tab(page, "Document & Audio Ingestion")
        time.sleep(1.5)

        smooth_scroll(page, 150)
        time.sleep(1.0)

        # Switch to Local File Upload
        file_radio = page.locator('label:has-text("Local File Upload")').first
        if file_radio.count() > 0:
            file_radio.click()
            time.sleep(1.5)

        # Switch to Audio Recording / Meeting Ingestion
        audio_radio = page.locator('label:has-text("Audio Recording")').first
        if audio_radio.count() > 0:
            audio_radio.click()
            time.sleep(1.8)

        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(0.8)

        # ==========================================
        # 3. RAG QUERY TAB
        # ==========================================
        print("Showcasing RAG Query Tab...")
        click_tab(page, "RAG Query")
        time.sleep(1.8)

        # ==========================================
        # 4. LANGGRAPH AGENT TAB
        # ==========================================
        print("Showcasing LangGraph Agent Tab...")
        click_tab(page, "LangGraph Agent")
        time.sleep(1.8)

        # ==========================================
        # 5. KOKORO TTS IMPLEMENTATION (TEXT-TO-AUDIO TAB)
        # ==========================================
        print("[3/9] Navigating to Text-to-Audio (TTS) Tab...")
        click_tab(page, "Text-to-Audio (TTS)")
        time.sleep(1.5)

        print("[4/9] Entering Kokoro Neural TTS prompt...")
        prompt_textarea = page.locator('textarea:visible').first
        prompt_textarea.scroll_into_view_if_needed()
        time.sleep(0.5)
        prompt_textarea.fill(
            "Welcome to the Enterprise Agentic Knowledge Platform. "
            "This neural voice is synthesized locally using the Kokoro-82M TTS engine, "
            "delivering 100% offline privacy, zero cloud egress, and high acoustic naturalness."
        )
        time.sleep(1.2)

        # Click Synthesize Speech
        print("[5/9] Clicking 'Synthesize Speech'...")
        synth_btn = page.locator('button:has-text("Synthesize Speech"):visible').first
        synth_btn.scroll_into_view_if_needed()
        time.sleep(0.5)
        synth_btn.click()

        # Wait for audio synthesis success
        try:
            page.locator('text="Speech synthesized successfully!"').wait_for(timeout=25000)
            print("Speech synthesis completed successfully!")
        except Exception as exc:
            print(f"Synthesis status note: {exc}")

        time.sleep(2.0)

        # Show rendered audio player
        smooth_scroll(page, 220)
        time.sleep(3.0)

        # Scroll to embedded TTS Quality & Benchmark Section on Tab 4
        print("Displaying embedded TTS Quality & Benchmark Metrics on TTS tab...")
        smooth_scroll(page, 280)
        time.sleep(3.0)  # KPI metric cards

        # Live Results Table
        smooth_scroll(page, 320)
        time.sleep(4.5)  # Evaluated live metrics table

        # Provider Comparison
        smooth_scroll(page, 280)
        time.sleep(3.0)

        # Taxonomy
        smooth_scroll(page, 280)
        time.sleep(3.5)

        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1.0)

        # ==========================================
        # 6. DEEPEVAL & QUALITY DASHBOARD -> TTS BENCHMARKS & QUALITY TAB
        # ==========================================
        print("[6/9] Navigating to DeepEval & Quality Dashboard...")
        click_tab(page, "DeepEval & Quality Dashboard")
        time.sleep(2.0)

        # Click inner tab: '🔊 TTS Benchmarks & Quality'
        print("[7/9] Selecting inner 'TTS Benchmarks & Quality' tab...")
        inner_tab = page.locator('[data-baseweb="tab"]').filter(has_text="TTS Benchmarks & Quality").first
        inner_tab.scroll_into_view_if_needed()
        inner_tab.click()
        time.sleep(2.0)

        # Showcase Top KPI Cards
        print("Showcasing KPI Metrics Cards...")
        smooth_scroll(page, 150)
        time.sleep(3.0)

        # Showcase Evaluated Metrics Details Table (Live Results)
        print("Showcasing Evaluated Metrics Details Table (Live Results)...")
        smooth_scroll(page, 250)
        time.sleep(4.5)

        # Showcase Provider Benchmark Comparison
        print("Showcasing Provider Benchmark Comparison...")
        smooth_scroll(page, 250)
        time.sleep(3.0)

        # Showcase Taxonomy
        print("Showcasing Core TTS Evaluation Metrics Taxonomy...")
        smooth_scroll(page, 250)
        time.sleep(4.0)

        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1.0)

        # ==========================================
        # 7. PYTEST TEST SUITE RESULTS TAB (PROMINENT COVERAGE)
        # ==========================================
        print("[8/9] Navigating to Pytest & Test Suite Results Tab...")
        click_tab(page, "Pytest & Test Suite Results")
        time.sleep(2.5)

        # Show Summary Metrics (Total Tests 26, Passed 26, Pass Rate 100%)
        print("Showcasing Pytest execution summary cards...")
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(3.0)

        # Show filters & scroll down to test cases
        print("Showcasing passing Kokoro and TTS test cases...")
        smooth_scroll(page, 220)
        time.sleep(2.5)

        # Expand a test case to show stack/execution detail
        first_expander = page.locator('[data-testid="stExpander"]').first
        if first_expander.count() > 0:
            first_expander.click()
            time.sleep(2.5)

        # Scroll further down to show more passed test cases
        smooth_scroll(page, 300)
        time.sleep(3.5)

        smooth_scroll(page, -400)
        time.sleep(2.0)

        # ==========================================
        # 8. FINALIZE RECORDING
        # ==========================================
        print("[9/9] Finalizing video...")
        page.close()
        context.close()
        browser.close()

        final_webm = video_dir / "kokoro_tts_complete_walkthrough.webm"
        final_mp4 = video_dir / "kokoro_tts_complete_walkthrough.mp4"

        # Locate generated Playwright webm file
        generated_webms = list(video_dir.glob("*.webm"))
        if generated_webms:
            src_webm = generated_webms[0]
            if src_webm != final_webm:
                if final_webm.exists():
                    final_webm.unlink()
                src_webm.rename(final_webm)
            print(f"SUCCESS: Saved WebM video to {final_webm}")

            # Convert to MP4 using imageio-ffmpeg
            try:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                cmd = [
                    ffmpeg_exe, "-y",
                    "-i", str(final_webm),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "fast",
                    str(final_mp4)
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"SUCCESS: Converted MP4 video to {final_mp4}")
            except Exception as conv_err:
                print(f"MP4 conversion notice: {conv_err}")

        print("Recording complete!")


if __name__ == "__main__":
    record_video()
