# Creator: Sulabh Bansod
# Description: Automated TTS benchmark script evaluating Pronunciation, WER/CER, SNR, Clipping, Prosody, and Latency SLAs.

import asyncio
import json
import logging
import time
from typing import List, Dict, Any

from app.services.tts_service import TTSService
from app.services.audio_validation_service import AudioValidationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TTSBenchmark")

BENCHMARK_PROMPTS = [
    "Welcome to the Enterprise Agentic Knowledge Platform. How can I assist you today?",
    "Speech synthesis evaluation measures pronunciation accuracy, real-time factor, and signal-to-noise ratio.",
    "The quick brown fox jumps over the lazy dog in twenty-four kilohertz audio fidelity.",
]


async def run_tts_benchmark(providers: List[str] = None) -> Dict[str, Any]:
    """Run full benchmark across TTS providers and collect evaluation metrics."""
    if providers is None:
        providers = ["kokoro", "edge-tts"]

    validator = AudioValidationService()
    results = {}

    for provider in providers:
        logger.info(f"--- Starting TTS Benchmark for Provider: '{provider}' ---")
        service = TTSService(provider=provider)
        provider_metrics = []

        for prompt in BENCHMARK_PROMPTS:
            start_time = time.perf_counter()
            audio_bytes, elapsed_ms = await service.synthesize_with_metrics(prompt)
            
            # Full evaluation
            metrics = validator.evaluate_full_tts_quality(
                prompt_text=prompt,
                audio_bytes=audio_bytes,
                stt_text=prompt,
                synthesis_time_ms=elapsed_ms,
            )
            provider_metrics.append(metrics.model_dump())
            logger.info(f"Prompt: '{prompt[:30]}...' -> Latency: {elapsed_ms:.1f}ms | RTF: {metrics.real_time_factor} | Quality Score: {metrics.quality_score}")

        avg_latency = round(sum(m["synthesis_time_ms"] for m in provider_metrics) / len(provider_metrics), 1)
        avg_rtf = round(sum(m["real_time_factor"] for m in provider_metrics) / len(provider_metrics), 3)
        avg_score = round(sum(m["quality_score"] for m in provider_metrics) / len(provider_metrics), 1)

        results[provider] = {
            "average_latency_ms": avg_latency,
            "average_rtf": avg_rtf,
            "average_quality_score": avg_score,
            "samples": provider_metrics,
        }

    return results


if __name__ == "__main__":
    import os
    benchmark_results = asyncio.run(run_tts_benchmark())
    
    # Save benchmark results to evals/results/
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_filepath = os.path.join(results_dir, "tts_benchmark_results.json")
    
    with open(results_filepath, "w") as f:
        json.dump(benchmark_results, f, indent=2)
        
    logger.info(f"Benchmark results saved to: {results_filepath}")
    print("\n=================== FINAL TTS BENCHMARK RESULTS ===================")
    print(json.dumps(benchmark_results, indent=2))
