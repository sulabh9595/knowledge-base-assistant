#!/usr/bin/env python3
"""Bulk local directory audio ingestion script.

Scans a local directory for audio recordings (.mp3, .wav, .m4a, .ogg, .flac, .aac),
transcribes them locally, and ingests them into the Knowledge Base Platform.
"""

import argparse
import sys
from pathlib import Path
import httpx

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}


def ingest_directory(dir_path: str, api_url: str = "http://127.0.0.1:8000", generate_summary: bool = True):
    path = Path(dir_path).resolve()
    if not path.is_dir():
        print(f"Error: {dir_path} is not a valid directory.")
        sys.exit(1)

    print("=" * 60)
    print("      LOCAL AUDIO DIRECTORY BULK INGESTION TOOL      ")
    print("=" * 60)
    print(f"Scanning directory: {path}")

    audio_files = [f for f in path.glob("**/*") if f.suffix.lower() in AUDIO_EXTENSIONS]

    if not audio_files:
        print("No audio files found matching supported formats (.mp3, .wav, .m4a, .ogg, .flac, .aac).")
        return

    print(f"Found {len(audio_files)} audio file(s) to process.")
    print("-" * 60)

    success_count = 0

    for idx, file_path in enumerate(audio_files, 1):
        print(f"[{idx}/{len(audio_files)}] Processing: {file_path.name}...")
        try:
            with file_path.open("rb") as f:
                files = {"file": (file_path.name, f.read(), "audio/wav")}
                res = httpx.post(
                    f"{api_url}/ingest/audio?generate_summary={str(generate_summary).lower()}",
                    files=files,
                    timeout=600
                )
                if res.status_code == 200:
                    data = res.json()
                    success_count += 1
                    print(f"   ✓ Ingested successfully!")
                    print(f"     - Page ID:    {data.get('page_id')}")
                    print(f"     - Language:   {data.get('language')}")
                    print(f"     - Duration:   {data.get('duration_seconds')}s")
                    print(f"     - Word Count: {data.get('word_count')}")
                else:
                    print(f"   ✗ Ingestion failed with status {res.status_code}: {res.text}")
        except Exception as e:
            print(f"   ✗ Error ingesting {file_path.name}: {e}")

    print("=" * 60)
    print(f"Ingestion complete: {success_count}/{len(audio_files)} audio files indexed.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk ingest local directory of audio recordings.")
    parser.add_argument("--dir", type=str, required=True, help="Path to local folder containing audio files.")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000", help="FastAPI backend URL.")
    parser.add_argument("--no-summary", action="store_true", help="Disable automatic local LLM summary generation.")
    args = parser.parse_args()

    ingest_directory(args.dir, args.api_url, generate_summary=not args.no_summary)
