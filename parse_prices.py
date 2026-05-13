#!/usr/bin/env python3
"""
Extract vegetable prices from Hindi/Hinglish transcripts using Ollama.

Improved behavior:
- Reads data/market_transcripts_master.csv
- Scans transcript dates from newest to oldest
- Uses the newest date that ACTUALLY contains mandi prices
- If latest video has no mandi prices, falls back to the most recent previous date with prices
- Preserves history from existing data/prices.json
- Builds time-series price history for dashboard charts
- All prices must be converted to ₹ per kg
- Polythene bag prices: if "15 केजी पन्नी ₹X" → price per kg = X/15
- Crate (कैरेट) of tomato ≈ 20-25 kg → divide accordingly
- "किसान ₹X मांग रहे" → farmer_price = X
- "खरीदार ₹X दे रहे" → buyer_price = X
- If only one price mentioned → use as sold_price
- Skip if price is completely ambiguous

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

SYSTEM_PROMPT = """You are a highly accurate agricultural market data extraction assistant.

Your job is to extract vegetable mandi prices from a Hindi or Hinglish YouTube transcript chunk.

Return ONLY a valid JSON array.
Do NOT return markdown.
Do NOT return explanations.
Do NOT return comments.
Do NOT return text before or after the JSON.

Each array item must have exactly these fields:

{
  "name_hi": "<Hindi vegetable name or empty string>",
  "name_en": "<standardized English vegetable name>",
  "farmer_price": <number or null>,
  "buyer_price": <number or null>,
  "sold_price": <number or null>,
  "price_min": <number or null>,
  "price_max": <number or null>,
  "price_avg": <number or null>,
  "unit": "kg",
  "raw_text": "<short original phrase from transcript>"
}

Important extraction rules:

1. Extract ONLY vegetables or leafy vegetable/herb items whose prices are clearly mentioned.
2. All final prices must be converted to INR per kg.
3. If the transcript gives a direct price per kg, use it directly.
4. If the transcript gives a range like:
   "20 से 30 रुपये"
   then:
   - price_min = 20
   - price_max = 30
   - price_avg = 25
5. If only one final sale price is mentioned, set:
   - sold_price = that value
   - price_avg = that value
   - price_min = null
   - price_max = null
6. If the transcript says:
   - "किसान ₹X मांग रहे" → farmer_price = X
   - "खरीदार ₹X दे रहे" → buyer_price = X
   - if only one clear market transaction price exists, treat it as sold_price
7. If multiple prices for the same vegetable appear in the same transcript chunk, include them as separate items only if they refer to clearly different contexts.
   Otherwise, prefer the main mandi sale price.
8. Skip completely ambiguous prices.
9. Do not guess missing vegetables.
10. Do not invent units.
11. Do not extract prices for non-vegetable products unless clearly relevant to the mandi vegetable context.

Unit conversion rules:

A. Polythene / bag conversion:
   If price is given for a bag with total weight, convert to per kg.
   Example:
   "15 केजी पन्नी 300"
   means 300 / 15 = 20 per kg

B. Tomato crate conversion:
   If tomato price is given per crate / कैरेट / क्रेट / पेटी and weight is implied as roughly 20–25 kg:
   - If an exact weight is spoken, use that exact weight.
   - If no exact weight is spoken and the transcript only says crate/कैरेट/क्रेट for tomato,
     assume an approximate range of 20 to 25 kg and estimate price_avg using 22.5 kg.
   - Example:
     "टमाटर कैरेट 450"
     price_avg = 450 / 22.5 = 20
   - If you use this approximation, still return only the numeric result.
   - Keep unit = "kg"

C. If quantity is per dozen, piece, bunch, bundle, or any unit that cannot be reliably converted to per kg,
   skip that item unless the transcript clearly provides enough information to convert.

Standardization rules:

Use standardized English names where possible:
- टमाटर → Tomato
- प्याज़ → Onion
- आलू → Potato
- बैंगन → Brinjal
- भिंडी → Lady Finger
- लौकी → Bottle Gourd
- करेला → Bitter Gourd
- गोभी / फूलगोभी → Cauliflower
- पत्ता गोभी → Cabbage
- हरी मिर्च / मिर्च → Green Chilli
- धनिया / हरा धनिया → Coriander
- लहसुन → Garlic
- अदरक → Ginger
- शिमला मिर्च → Capsicum
- पालक → Spinach
- मटर → Peas
- सेम → Beans

Normalization rules:
- Corriander, coriander leaves, green coriander, dhaniya, धनिया, हरा धनिया → Coriander
- Okra, ladyfinger, bhindi → Lady Finger
- Eggplant, brinjal, बैंगन → Brinjal

Price interpretation priority:
1. sold_price
2. farmer_price / buyer_price
3. price range
4. single clear market price

How to fill fields:
- If only one price is available and it is the effective mandi price:
  farmer_price = null
  buyer_price = null
  sold_price = X
  price_min = null
  price_max = null
  price_avg = X

- If a farmer vs buyer negotiation is given:
  farmer_price = X
  buyer_price = Y
  sold_price = whichever is clearly stated as final deal, otherwise null
  price_avg = sold_price if available, otherwise null

- If a range is given:
  price_min = lower
  price_max = upper
  price_avg = midpoint

- If both a converted price and a quoted raw total are mentioned, return only the per-kg numeric values.

Very important:
- Skip any price that is too vague, incomplete, or cannot be confidently tied to a vegetable.
- Skip any vegetable mention without a price.
- Output ONLY valid JSON array.

Examples:

Input text:
"धनिया 15 केजी पन्नी 300 रुपये बिक रही है"
Output:
[
  {
    "name_hi": "धनिया",
    "name_en": "Coriander",
    "farmer_price": null,
    "buyer_price": null,
    "sold_price": 20,
    "price_min": null,
    "price_max": null,
    "price_avg": 20,
    "unit": "kg",
    "raw_text": "धनिया 15 केजी पन्नी 300 रुपये"
  }
]

Input text:
"टमाटर कैरेट 450 से 500 चल रहा है"
Output:
[
  {
    "name_hi": "टमाटर",
    "name_en": "Tomato",
    "farmer_price": null,
    "buyer_price": null,
    "sold_price": null,
    "price_min": 20,
    "price_max": 22.22,
    "price_avg": 21.11,
    "unit": "kg",
    "raw_text": "टमाटर कैरेट 450 से 500"
  }
]

Input text:
"किसान 18 मांग रहे हैं, खरीदार 16 दे रहे हैं, आलू"
Output:
[
  {
    "name_hi": "आलू",
    "name_en": "Potato",
    "farmer_price": 18,
    "buyer_price": 16,
    "sold_price": null,
    "price_min": null,
    "price_max": null,
    "price_avg": null,
    "unit": "kg",
    "raw_text": "किसान 18 मांग रहे हैं, खरीदार 16 दे रहे हैं, आलू"
  }
]

Input text:
"आज मौसम अच्छा है, बाजार में रौनक है"
Output:
[]"""


def parse_date_flexible(date_str: str):
    """Support old and new date formats."""
    if not date_str:
        return None

    date_str = date_str.strip()
    formats = [
        "%Y-%m-%d",
        "%d-%m-%y",
        "%d-%m-%Y",
        "%m-%d-%y",
        "%m-%d-%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def ollama_request(payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")

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
    """Extract structured prices from transcript text."""
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

        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\n?```$", "", raw)

        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())

        return json.loads(raw)

    except Exception as e:
        print(f"  -> Extraction error: {e}")
        return []


def load_all_transcripts(csv_file: str) -> list:
    """Load all transcript rows with valid dates."""
    rows = []

    if not os.path.isfile(csv_file):
        print(f"CSV not found: {csv_file}")
        return rows

    with open(csv_file, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)

        for row in reader:
            date_str = (row.get("Date") or "").strip()
            transcript = (row.get("Transcript") or "").strip()

            parsed_date = parse_date_flexible(date_str)
            if parsed_date is None or not transcript:
                continue

            rows.append({
                "parsed_date": parsed_date,
                "date": parsed_date.strftime("%Y-%m-%d"),
                "title": (row.get("Title") or "").strip(),
                "video_id": (row.get("Video_ID") or "").strip(),
                "transcript": transcript,
            })

    rows.sort(key=lambda r: r["parsed_date"], reverse=True)
    return rows


def group_transcripts_by_date(rows: list) -> dict:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["date"], []).append(row)
    return grouped


def normalize_price_entry(item: dict) -> dict | None:
    """Normalize one extracted vegetable price entry."""
    name_en = (item.get("name_en") or "").strip()
    if not name_en:
        return None

    avg = item.get("price_avg")
    if avg is None:
        mn = item.get("price_min")
        mx = item.get("price_max")
        if mn is not None and mx is not None:
            try:
                avg = (float(mn) + float(mx)) / 2
            except Exception:
                avg = None

    if avg is None:
        return None

    try:
        avg = round(float(avg), 2)
    except Exception:
        return None

    return {
        "name_hi": (item.get("name_hi") or "").strip(),
        "name_en": name_en.strip().title(),
        "unit": (item.get("unit") or "kg").strip(),
        "currency": "INR",
        "price": avg,
    }


def pick_latest_price_bearing_date(transcripts: list) -> tuple[str | None, list]:
    """
    Starting from newest date, find the first date that yields any mandi prices.
    Returns:
        (source_date, normalized_prices_for_that_date)
    """
    if not transcripts:
        return None, []

    grouped = group_transcripts_by_date(transcripts)

    # Sort dates descending
    dates = sorted(grouped.keys(), reverse=True)

    for date_str in dates:
        print(f"\nTrying transcript date: {date_str}")
        day_rows = grouped[date_str]
        extracted = []

        for row in day_rows:
            print(f"  Parsing: {row['title'][:70]}")
            prices = extract_prices(row["transcript"])
            print(f"    -> {len(prices)} extracted item(s)")
            extracted.extend(prices)

        normalized = []
        for item in extracted:
            norm = normalize_price_entry(item)
            if norm:
                normalized.append(norm)

        if normalized:
            print(f"Using latest price-bearing date: {date_str}")
            return date_str, normalized

        print(f"No mandi prices found on {date_str}, checking older date...")

    return None, []


def load_existing_output() -> dict:
    """Load existing prices.json if available."""
    if not os.path.isfile(OUTPUT_FILE):
        return {}

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_history_map(existing_output: dict) -> dict:
    """
    Convert existing vegetables history into a map:
    key -> vegetable name_en lower
    """
    history_map = {}
    vegetables = existing_output.get("vegetables", [])

    for veg in vegetables:
        key = (veg.get("name_en") or "").strip().lower()
        if not key:
            continue

        history_map[key] = {
            "name_hi": veg.get("name_hi", ""),
            "name_en": veg.get("name_en", ""),
            "unit": veg.get("unit", "kg"),
            "currency": veg.get("currency", "INR"),
            "history": veg.get("history", [])[:] if isinstance(veg.get("history", []), list) else [],
        }

    return history_map


def upsert_history_point(history: list, date_str: str, price: float):
    """
    Insert or replace a history point for a given date.
    Keeps unique dates.
    """
    replaced = False

    for item in history:
        if item.get("date") == date_str:
            item["price"] = round(float(price), 2)
            replaced = True
            break

    if not replaced:
        history.append({
            "date": date_str,
            "price": round(float(price), 2),
        })

    history.sort(key=lambda x: x["date"])


def merge_current_prices_with_existing(existing_output: dict, source_date: str, current_prices: list) -> list:
    """
    Merge newly extracted current prices into existing history.
    This preserves old history and updates only with the newest valid price-bearing date.
    """
    history_map = build_history_map(existing_output)

    for item in current_prices:
        key = item["name_en"].lower()

        if key not in history_map:
            history_map[key] = {
                "name_hi": item["name_hi"],
                "name_en": item["name_en"],
                "unit": item["unit"],
                "currency": item["currency"],
                "history": [],
            }

        # update names/unit if blank in old data
        if item["name_hi"]:
            history_map[key]["name_hi"] = item["name_hi"]
        history_map[key]["name_en"] = item["name_en"]
        history_map[key]["unit"] = item["unit"]
        history_map[key]["currency"] = item["currency"]

        upsert_history_point(history_map[key]["history"], source_date, item["price"])

    # Build final vegetable output
    result = []

    for _, veg in history_map.items():
        history = sorted(veg["history"], key=lambda h: h["date"])
        if not history:
            continue

        latest = history[-1]["price"]
        prev = history[-2]["price"] if len(history) >= 2 else None

        trend = "stable"
        change_pct = 0

        if prev is not None:
            if latest > prev:
                trend = "up"
            elif latest < prev:
                trend = "down"
            else:
                trend = "stable"

            if prev != 0:
                change_pct = round(((latest - prev) / prev) * 100, 1)

        peak_point = max(history, key=lambda x: x["price"])
        low_point = min(history, key=lambda x: x["price"])

        result.append({
            "name_hi": veg["name_hi"],
            "name_en": veg["name_en"],
            "unit": veg["unit"],
            "currency": veg["currency"],
            "latest_price": latest,
            "trend": trend,
            "change_pct": change_pct,
            "latest_date": history[-1]["date"],
            "max_price": peak_point["price"],
            "max_price_date": peak_point["date"],
            "min_price": low_point["price"],
            "min_price_date": low_point["date"],
            "history": history[-60:],  # keep more history for charting
        })

    result.sort(key=lambda v: v["name_en"])
    return result


def write_output(existing_output: dict, source_date: str | None, vegetables: list):
    output = {
        "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "latest_source_date": source_date,
        "source_channel": "UCxEW_BSHnu43J8-ANnSJ80w",
        "model_used": MODEL,
        "vegetables": vegetables,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(vegetables)} vegetable(s) -> {OUTPUT_FILE}")


def main():
    os.makedirs("data", exist_ok=True)

    if not wait_for_ollama():
        raise RuntimeError("Ollama did not start in time.")

    pull_model(MODEL)

    transcripts = load_all_transcripts(CSV_FILE)
    print(f"\nLoaded {len(transcripts)} transcript row(s) from CSV")

    existing_output = load_existing_output()

    if not transcripts:
        print("No transcripts available.")
        if existing_output:
            print("Keeping existing prices.json")
            return
        write_output({}, None, [])
        return

    source_date, current_prices = pick_latest_price_bearing_date(transcripts)

    if not current_prices:
        print("No mandi-price transcripts found in CSV.")
        if existing_output:
            print("Keeping existing prices.json")
            return
        write_output({}, None, [])
        return

    existing_source_date = existing_output.get("latest_source_date")

    # If same price-bearing date already active, keep history but refresh metadata
    merged_vegetables = merge_current_prices_with_existing(existing_output, source_date, current_prices)

    # Optional optimization: if nothing meaningful changed, still refresh timestamp
    if existing_source_date == source_date:
        print(f"Latest valid mandi-price date unchanged ({source_date}). Refreshing output only.")
    else:
        print(f"Switching dashboard source date from {existing_source_date} -> {source_date}")

    write_output(existing_output, source_date, merged_vegetables)


if __name__ == "__main__":
    main()
