"""
Automated Playwright Video Recorder for Kokoro TTS Implementation.
Records a high-definition video walkthrough of the Streamlit application.
"""

import os
import time
from playwright.sync_api import sync_playwright

def record_demo():
    video_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    print(f"Starting video recording into: {video_dir}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            record_video_dir=video_dir,
            record_video_size={"width": 1280, "height": 800},
            viewport={"width": 1280, "height": 800}
        )
        
        page = context.new_page()
        
        print("Navigating to Streamlit UI at http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle")
        time.sleep(3)
        
        # Click Tab 5: Text-to-Audio (TTS)
        print("Switching to Text-to-Audio (TTS) Tab...")
        tab_btn = page.locator('button[data-baseweb="tab"]:has-text("Text-to-Audio")')
        if tab_btn.count() > 0:
            tab_btn.first.click()
        else:
            page.locator('text="Text-to-Audio (TTS)"').first.click()
        
        time.sleep(2.5)
        
        # Show the TTS configuration options
        print("Interacting with TTS options...")
        page.mouse.wheel(0, 150)
        time.sleep(1.5)
        
        # Fill the visible textarea
        print("Entering custom speech prompt...")
        prompt_textarea = page.locator('textarea:visible').first
        prompt_textarea.scroll_into_view_if_needed()
        time.sleep(1)
        prompt_textarea.fill(
            "Welcome to the Enterprise Agentic Knowledge Platform. "
            "This neural speech is powered by the Kokoro Text-to-Speech engine, "
            "offering privacy-first local synthesis with high acoustic naturalness and zero cloud egress."
        )
        time.sleep(1.5)
        page.keyboard.press("Tab")
        time.sleep(1.5)
        
        # Click Synthesize Speech button
        print("Clicking 'Synthesize Speech' button...")
        synth_btn = page.locator('button:has-text("Synthesize Speech"):visible').first
        synth_btn.scroll_into_view_if_needed()
        time.sleep(1)
        synth_btn.click()
        
        print("Waiting for speech synthesis and audio player...")
        try:
            page.locator('text="Speech synthesized successfully!"').wait_for(timeout=25000)
            print("Synthesis success message detected.")
        except Exception as e:
            print(f"Note: Wait timeout on toast ({e}), checking audio player...")
        
        time.sleep(3)
        
        # Scroll down to showcase the rendered HTML5 audio player and output
        page.mouse.wheel(0, 250)
        time.sleep(4)
        
        # Navigate to Dashboard Tab to demonstrate TTS validation & benchmark metrics
        print("Navigating to DeepEval & Quality Dashboard Tab...")
        eval_tab = page.locator('button[data-baseweb="tab"]:has-text("Dashboard")')
        if eval_tab.count() > 0:
            eval_tab.first.click()
            time.sleep(3)
            page.mouse.wheel(0, 400)
            time.sleep(3)
            page.mouse.wheel(0, -400)
            time.sleep(2)
        
        # Navigate to Pytest Test Suite Results Tab to show verified test suite
        print("Navigating to Pytest Test Suite Results Tab...")
        test_tab = page.locator('button[data-baseweb="tab"]:has-text("Pytest")')
        if test_tab.count() > 0:
            test_tab.first.click()
            time.sleep(3)
            page.mouse.wheel(0, 300)
            time.sleep(3)
            page.mouse.wheel(0, -300)
            time.sleep(1.5)
            
        # Return to TTS Synthesizer Tab
        if tab_btn.count() > 0:
            tab_btn.first.click()
            time.sleep(3)
        
        # Close page & context to finalize video
        print("Finalizing and saving video...")
        page_video = page.video
        video_temp_path = page_video.path() if page_video else None
        
        page.close()
        context.close()
        browser.close()
        
        final_video = os.path.join(video_dir, "kokoro_tts_demo.webm")
        if video_temp_path and os.path.exists(video_temp_path):
            if os.path.exists(final_video):
                os.remove(final_video)
            os.rename(video_temp_path, final_video)
            print(f"SUCCESS: Video recorded at: {final_video}")
            return final_video
        
        return video_dir

if __name__ == "__main__":
    record_demo()
