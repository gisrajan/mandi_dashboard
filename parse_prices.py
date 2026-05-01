#!/usr/bin/env python3
"""
Extracts vegetable prices from Hindi/Hinglish transcripts using a
locally-running open-source LLM via Ollama (no paid API needed).

Model: mistral:7b (or gemma2:2b for faster/lighter runs)
Ollama is started as a service in the GitHub Actions workflow.
"""

import os
import json
import csv
import re
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta

CSV_FILE    = "data/market_transcripts_master.csv"
OUTPUT_FILE = "data/prices.json"
LOOKBACK_DAYS = 7
OLLAMA_URL  = "http://localhost:11434"
MODEL       = os.environ.get("OLLAMA_MODEL", "mistral")   # or "gemma2:2b"

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
- Output ONLY the JSON array, nothing else"""


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

    # Ollama streams NDJSON — each line is a JSON object; last has done=true
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    full_text = ""
    for line in lines:
        try:
            obj = json.loads(line)
            full_text += obj.get("message", {}).get("content", "")
        except json.JSONDecodeError:
            pass
    return {"content": full_text}


def wait_for_ollama(retries: int = 20, delay: float = 3.0):
    """Poll until Ollama is ready (it takes ~10s to start in CI)."""
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
    """Pull the model if not already cached (no-op if cached in CI)."""
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
        # Streaming response — just drain it
        with urllib.request.urlopen(req, timeout=300) as resp:
            while resp.read(4096):
                pass
        print(f"Model '{model}' ready.")
    except Exception as e:
        print(f"Pull warning (may be fine if cached): {e}")


def extract_prices(transcript: str) -> list:
    """Ask Ollama to extract price data from transcript text."""
    # Trim to ~3000 chars to stay within context window
    transcript_trimmed = transcript[:3000]

    payload = {
        "model": MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": transcript_trimmed},
        ],
        "options": {
            "temperature": 0.1,    # Low temp = more deterministic JSON
            "num_predict": 1024,
        },
    }

    try:
        result = ollama_request(payload, timeout=180)
        raw = result["content"].strip()

        # Strip markdown fences if model added them
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\n?```$", "", raw)

        # Find the JSON array (robust even if model adds preamble)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(raw)

    except Exception as e:
        print(f"  -> Extraction error: {e}")
        return []


def load_recent_transcripts(csv_file: str, lookback_days: int) -> list:
    cutoff = datetime.now() - timedelta(days=lookback_days)
    rows = []
    if not os.path.isfile(csv_file):
        print(f"CSV not found: {csv_file}")
        return rows
    with open(csv_file, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row.get("Date", "")
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            if date >= cutoff and row.get("Transcript", "").strip():
                rows.append({
                    "date":      date_str,
                    "title":     row.get("Title", ""),
                    "video_id":  row.get("Video_ID", ""),
                    "transcript":row.get("Transcript", ""),
                })
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows


def merge_prices(all_entries: list) -> list:
    by_name: dict[str, dict] = {}
    for entry in all_entries:
        date = entry["date"]
        for item in entry["prices"]:
            key = item.get("name_en", "").lower().strip()
            if not key:
                continue
            if key not in by_name:
                by_name[key] = {
                    "name_hi": item.get("name_hi", ""),
                    "name_en": item.get("name_en", ""),
                    "unit":    item.get("unit", "kg"),
                    "currency":"INR",
                    "history": [],
                }
            avg = item.get("price_avg")
            if avg is None:
                mn = item.get("price_min")
                mx = item.get("price_max")
                if mn is not None and mx is not None:
                    avg = (mn + mx) / 2
            if avg is not None:
                by_name[key]["history"].append({"date": date, "price": round(avg, 2)})

    results = []
    for _, veg in by_name.items():
        history = sorted(veg["history"], key=lambda h: h["date"])
        if not history:
            continue
        latest = history[-1]["price"]
        prev   = history[-2]["price"] if len(history) >= 2 else None
        trend  = "stable"
        change_pct = 0
        if prev is not None:
            trend = "up" if latest > prev else ("down" if latest < prev else "stable")
            change_pct = round(((latest - prev) / prev) * 100, 1) if prev else 0

        results.append({
            "name_hi":     veg["name_hi"],
            "name_en":     veg["name_en"].title(),
            "unit":        veg["unit"],
            "currency":    "INR",
            "latest_price":latest,
            "trend":       trend,
            "change_pct":  change_pct,
            "latest_date": history[-1]["date"],
            "history":     history[-14:],
        })

    results.sort(key=lambda v: v["name_en"])
    return results


def main():
    os.makedirs("data", exist_ok=True)

    # Wait for Ollama sidecar to start
    if not wait_for_ollama():
        raise RuntimeError("Ollama did not start in time.")

    # Pull model (uses cache if already downloaded in a previous run)
    pull_model(MODEL)

    transcripts = load_recent_transcripts(CSV_FILE, LOOKBACK_DAYS)
    print(f"\nFound {len(transcripts)} recent transcript(s) to parse")

    all_entries = []
    for t in transcripts:
        print(f"  [{t['date']}] {t['title'][:60]}")
        prices = extract_prices(t["transcript"])
        print(f"    -> {len(prices)} vegetable(s) found")
        if prices:
            all_entries.append({"date": t["date"], "prices": prices})

    if not all_entries:
        print("No new price data extracted.")
        if os.path.isfile(OUTPUT_FILE):
            print("Keeping existing prices.json.")
        return

    merged = merge_prices(all_entries)
    output = {
        "last_updated":   datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "source_channel": "UCxEW_BSHnu43J8-ANnSJ80w",
        "model_used":     MODEL,
        "vegetables":     merged,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(merged)} vegetable price(s) → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
