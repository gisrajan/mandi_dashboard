#!/usr/bin/env python3
"""
Extract vegetable prices from Hindi/Hinglish transcripts using a
locally-running open-source LLM via Ollama.

Behavior:
- Reads data/market_transcripts_master.csv
- Finds the latest date available in the CSV with transcript text
- Parses transcripts only from that latest available date
- Writes output to data/prices.json

This ensures that if there is NO new video today, the dashboard still shows
the most recently available mandi prices from the latest uploaded video(s).
"""

import os
import json
import csv
import re
import time
import urllib.request
from datetime import datetime

CSV_FILE = "data/market_transcripts_master.csv"
OUTPUT_FILE = "data/prices.json"
OLLAMA_URL = "http://localhost:11434"
MODEL = os.environ.get("OLLAMA_MODEL", "gemma2:2b")

SYSTEM_PROMPT = """You are a data extraction assistant for Indian vegetable markets.
You receive a Hindi or Hinglish YouTube transcript about mandi (wholesale market) prices.
Extract ALL vegetable prices mentioned. Return ONLY a valid JSON array — no explanation, no markdown.

Each object in the array must have exactly these fields:
{
  "name_hi": "<Hindi name, e.g. टमाटर>",
  "name_en": "<English name, e.g. Tomato>",
  "price_min": <number or null>,
  "price_max": <number or null>,
  "price_avg": <number or null>,
  "unit": "kg"
}

Rules:
- price in Indian Rupees (INR) per unit
- If a range like "20 se 30 rupye" → price_min=20, price_max=30, price_avg=25
- If a single price → set price_avg only; price_min and price_max = null
- unit is always "kg" unless clearly stated otherwise (dozen, piece, quintal)
- If a vegetable appears multiple times, average the prices
- Return [] if no prices are found
- Output ONLY the JSON array, nothing else
"""


def parse_date_flexible(date_str: str):
    """
    Parse multiple possible date formats.
    Supported:
        2026-04-03
        03-04-26
        03-04-2026
        04-03-26
        04-03-2026
    Returns a datetime object or None.
    """
    if not date_str:
        return None

    date_str = date_str.strip()

    formats = [
        "%Y-%m-%d",  # 2026-04-03
        "%d-%m-%y",  # 03-04-26
        "%d-%m-%Y",  # 03-04-2026
        "%m-%d-%y",  # fallback
        "%m-%d-%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def ollama_request(payload: dict, timeout: int = 120) -> dict:
    """Send a request to the local Ollama REST API."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")

    # Ollama streams NDJSON; concatenate content parts
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    full_text = ""

    for line in lines:
        try:
            obj = json.loads(line)
            full_text += obj.get("message", {}).get("content", "")
        except json.JSONDecodeError:
            pass

    return {"content": full_text}


def wait_for_ollama(retries: int = 20, delay: float = 3.0):
    """Poll until Ollama is ready."""
    print("Waiting for Ollama to be ready...", end="", flush=True)

    for _ in range(retries):
        try:
            urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
            print(" ready!")
            return True
        except Exception:
            print(".", end="", flush=True)
            time.sleep(delay)

    print(" timed out!")
    return False


def pull_model(model: str):
    """Pull the model if not already cached."""
    print(f"Pulling model '{model}' (skipped if cached)...")

    payload = {"name": model}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/pull",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            while resp.read(4096):
                pass
        print(f"Model '{model}' ready.")
    except Exception as e:
        print(f"Pull warning (may be fine if cached): {e}")


def extract_prices(transcript: str) -> list:
    """Ask Ollama to extract structured price data from transcript text."""
    transcript_trimmed = transcript[:3000]

    payload = {
        "model": MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript_trimmed},
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
        },
    }

    try:
        result = ollama_request(payload, timeout=180)
        raw = result["content"].strip()

        # Strip markdown fences if model adds them
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\n?```$", "", raw)

        # Try to find the JSON array even if the model added extra text
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())

        return json.loads(raw)

    except Exception as e:
        print(f"  -> Extraction error: {e}")
        return []


def load_latest_available_transcripts(csv_file: str) -> tuple[list, str | None]:
    """
    Load transcripts only from the most recent date available in the CSV.
    Returns:
        (rows, latest_date_str)
    """
    rows = []

    if not os.path.isfile(csv_file):
        print(f"CSV not found: {csv_file}")
        return rows, None

    parsed_rows = []

    with open(csv_file, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)

        for row in reader:
            date_str = (row.get("Date") or "").strip()
            transcript = (row.get("Transcript") or "").strip()

            parsed_date = parse_date_flexible(date_str)
            if parsed_date is None or not transcript:
                continue

            parsed_rows.append({
                "parsed_date": parsed_date,
                "date": parsed_date.strftime("%Y-%m-%d"),
                "title": (row.get("Title") or "").strip(),
                "video_id": (row.get("Video_ID") or "").strip(),
                "transcript": transcript,
            })

    if not parsed_rows:
        return [], None

    latest_date = max(r["parsed_date"] for r in parsed_rows)
