#!/usr/bin/env python3
"""
Extract vegetable mandi prices from Hindi/Hinglish transcripts using Ollama.

Behavior:
- Reads data/market_transcripts_master.csv
- Scans transcript dates from newest to oldest
- Uses the newest date that ACTUALLY contains mandi prices
- If latest video has no mandi prices, falls back to the most recent previous date with prices
- Preserves history from existing data/prices.json
- Builds time-series price history for dashboard charts
- If price_avg is missing, buyer_price is used as display price
"""

import os
import json
import csv
import re
import time
import urllib.request
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

CSV_FILE = "data/market_transcripts_master.csv"
OUTPUT_FILE = "data/prices.json"
OLLAMA_URL = "http://localhost:11434"
MODEL = os.environ.get("OLLAMA_MODEL", "gemma2:2b")

SYSTEM_PROMPT = """You are a highly accurate agricultural market data extraction assistant.

Your job is to extract vegetable mandi prices from a Hindi or Hinglish YouTube transcript chunk.

Return ONLY a valid JSON array of objects.
Do NOT return markdown.
Do NOT return explanations.

Extraction rules:
1. Extract ALL vegetables mentioned with a clear price.
2. Convert all prices to INR per kg.
3. If a range is given ("20 से 30"): price_min=20, price_max=30, price_avg=25.
4. If one final price is given: sold_price=X, price_avg=X.

Example Output Format:
[
  {
    "name_hi": "टमाटर",
    "name_en": "Tomato",
    "farmer_price": null,
    "buyer_price": null,
    "sold_price": 40,
    "price_min": null,
    "price_max": null,
    "price_avg": 40,
    "unit": "kg",
    "raw_text": "टमाटर 40 रुपये किलो"
  },
  {
    "name_hi": "आलू",
    "name_en": "Potato",
    "farmer_price": null,
    "buyer_price": null,
    "sold_price": null,
    "price_min": 15,
    "price_max": 20,
    "price_avg": 17.5,
    "unit": "kg",
    "raw_text": "आलू 15 से 20"
  }
]

Output ONLY a valid JSON array.
"""


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def parse_date_flexible(date_str: str) -> Optional[datetime]:
    """Support old and new date formats."""
    if not date_str:
        return None

    date_str = date_str.strip()
    formats = [
        "%Y-%m-%d",  # 2026-05-03
        "%d-%m-%y",  # 03-05-26
        "%d-%m-%Y",  # 03-05-2026
        "%m-%d-%y",
        "%m-%d-%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """
    Split long transcript into overlapping chunks so later price mentions are not missed.
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])

        if end == n:
            break

        start = max(end - overlap, 0)

    return chunks


def to_float(value: Any) -> Optional[float]:
    """Safe numeric conversion."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------
# Ollama communication
# ---------------------------------------------------------------------

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


def wait_for_ollama(retries: int = 20, delay: float = 3.0) -> bool:
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


def pull_model(model: str) -> None:
    """Pull model if not already cached."""
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


# ---------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------

def normalize_vegetable_names(item: dict) -> dict:
    """
    Normalize spelling variations and synonyms into a stable vegetable name.
    """
    name_hi = (item.get("name_hi") or "").strip()
    name_en = (item.get("name_en") or "").strip()

    key = name_en.lower().strip()

    aliases = {
        "corriander": "Coriander",
        "coriander leaves": "Coriander",
        "green coriander": "Coriander",
        "dhaniya": "Coriander",
        "coriander": "Coriander",
        "ladyfinger": "Lady Finger",
        "lady finger": "Lady Finger",
        "okra": "Lady Finger",
        "bhindi": "Lady Finger",
        "brinjal": "Brinjal",
        "eggplant": "Brinjal",
        "green chilli": "Green Chilli",
        "green chili": "Green Chilli",
        "chilli": "Green Chilli",
        "capsicum": "Capsicum",
        "shimla mirch": "Capsicum",
    }

    hi_aliases = {
        "धनिया": ("धनिया", "Coriander"),
        "हरा धनिया": ("धनिया", "Coriander"),
        "भिंडी": ("भिंडी", "Lady Finger"),
        "बैंगन": ("बैंगन", "Brinjal"),
        "टमाटर": ("टमाटर", "Tomato"),
        "प्याज़": ("प्याज़", "Onion"),
        "प्याज": ("प्याज़", "Onion"),
        "आलू": ("आलू", "Potato"),
        "लहसुन": ("लहसुन", "Garlic"),
        "अदरक": ("अदरक", "Ginger"),
        "शिमला मिर्च": ("शिमला मिर्च", "Capsicum"),
        "हरी मिर्च": ("हरी मिर्च", "Green Chilli"),
        "मिर्च": ("हरी मिर्च", "Green Chilli"),
        "करेला": ("करेला", "Bitter Gourd"),
        "लौकी": ("लौकी", "Bottle Gourd"),
        "गोभी": ("गोभी", "Cauliflower"),
        "फूलगोभी": ("गोभी", "Cauliflower"),
        "पत्ता गोभी": ("पत्ता गोभी", "Cabbage"),
        "पालक": ("पालक", "Spinach"),
        "मटर": ("मटर", "Peas"),
        "सेम": ("सेम", "Beans"),
    }

    if name_hi in hi_aliases:
        fixed_hi, fixed_en = hi_aliases[name_hi]
        item["name_hi"] = fixed_hi
        item["name_en"] = fixed_en
        return item

    if key in aliases:
        item["name_en"] = aliases[key]
        if item["name_en"] == "Coriander" and not item.get("name_hi"):
            item["name_hi"] = "धनिया"

    return item


def choose_display_price(item: dict) -> Tuple[Optional[float], Optional[str]]:
    """
    Choose which price should be used as the effective display price.

    Priority:
    1. price_avg
    2. buyer_price
    3. sold_price
    4. farmer_price
    5. midpoint of price_min/price_max
    """
    price_avg = to_float(item.get("price_avg"))
    if price_avg is not None:
        return price_avg, "price_avg"

    buyer_price = to_float(item.get("buyer_price"))
    if buyer_price is not None:
        return buyer_price, "buyer_price"

    sold_price = to_float(item.get("sold_price"))
    if sold_price is not None:
        return sold_price, "sold_price"

    farmer_price = to_float(item.get("farmer_price"))
    if farmer_price is not None:
        return farmer_price, "farmer_price"

    price_min = to_float(item.get("price_min"))
    price_max = to_float(item.get("price_max"))
    if price_min is not None and price_max is not None:
        return (price_min + price_max) / 2, "range_midpoint"

    return None, None


def normalize_price_entry(item: dict) -> Optional[dict]:
    """
    Normalize one extracted vegetable price entry.
    """
    if not isinstance(item, dict):
        return None

    item = normalize_vegetable_names(item)

    name_en = (item.get("name_en") or "").strip()
    if not name_en:
        return None

    chosen_price, chosen_type = choose_display_price(item)
    if chosen_price is None:
        return None

    return {
        "name_hi": (item.get("name_hi") or "").strip(),
        "name_en": name_en.title(),
        "unit": (item.get("unit") or "kg").strip(),
        "currency": "INR",
        "price": round(float(chosen_price), 2),
        "price_type": chosen_type,
        "buyer_price": to_float(item.get("buyer_price")),
        "sold_price": to_float(item.get("sold_price")),
        "farmer_price": to_float(item.get("farmer_price")),
        "price_avg": to_float(item.get("price_avg")),
        "raw_text": (item.get("raw_text") or "").strip(),
    }


def extract_prices(transcript: str) -> List[dict]:
    """
    Extract structured prices from the FULL transcript using chunking.
    """
    chunks = chunk_text(transcript, chunk_size=2500, overlap=300)
    if not chunks:
        return []

    raw_items = []

    for idx, chunk in enumerate(chunks, start=1):
        print(f"    -> Parsing transcript chunk {idx}/{len(chunks)}")

        payload = {
            "model": MODEL,
            "stream": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Extract all clearly stated vegetable mandi prices from the following Hindi/Hinglish transcript chunk.\n\nTranscript:\n{chunk}"
                },
            ],
            "options": {
                "temperature": 0.1,
                "num_predict": 1024,
            },
        }

        try:
            result = ollama_request(payload, timeout=180)
            raw = result["content"].strip()

            # ===== CRITICAL GITHUB DEBUG PRINTS =====
            print(f"\n================ [DEBUG] RAW LLM CHUNK OUTPUT ================")
            print(raw)
            print(f"==============================================================\n")
            # =======================================

          
            # Remove markdown fences if present
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\n?```$", "", raw)

            match = re.search(r"\[.*\]", raw, re.DOTALL)
            parsed = json.loads(match.group()) if match else json.loads(raw)

            if isinstance(parsed, list):
                raw_items.extend(parsed)

        except Exception as e:
            print(f"      Extraction error in chunk {idx}: {e}")

    # Normalize + merge duplicate vegetables across chunks
    merged: Dict[str, dict] = {}

    for item in raw_items:
        norm = normalize_price_entry(item)
        if not norm:
            continue

        key = norm["name_en"].lower()

        if key not in merged:
            merged[key] = {
                "name_hi": norm["name_hi"],
                "name_en": norm["name_en"],
                "unit": norm["unit"],
                "currency": norm["currency"],
                "prices": [],
                "buyer_prices": [],
                "sold_prices": [],
                "farmer_prices": [],
                "price_avgs": [],
                "raw_texts": [],
                "price_type_counts": {},
            }

        merged[key]["prices"].append(norm["price"])

        if norm["buyer_price"] is not None:
            merged[key]["buyer_prices"].append(norm["buyer_price"])
        if norm["sold_price"] is not None:
            merged[key]["sold_prices"].append(norm["sold_price"])
        if norm["farmer_price"] is not None:
            merged[key]["farmer_prices"].append(norm["farmer_price"])
        if norm["price_avg"] is not None:
            merged[key]["price_avgs"].append(norm["price_avg"])
        if norm["raw_text"]:
            merged[key]["raw_texts"].append(norm["raw_text"])

        merged[key]["price_type_counts"][norm["price_type"]] = (
            merged[key]["price_type_counts"].get(norm["price_type"], 0) + 1
        )

        # Prefer a non-empty Hindi name if found later
        if norm["name_hi"]:
            merged[key]["name_hi"] = norm["name_hi"]

    final = []

    for veg in merged.values():
        effective_price = round(sum(veg["prices"]) / len(veg["prices"]), 2)

        buyer_price = round(sum(veg["buyer_prices"]) / len(veg["buyer_prices"]), 2) if veg["buyer_prices"] else None
        sold_price = round(sum(veg["sold_prices"]) / len(veg["sold_prices"]), 2) if veg["sold_prices"] else None
        farmer_price = round(sum(veg["farmer_prices"]) / len(veg["farmer_prices"]), 2) if veg["farmer_prices"] else None
        price_avg = round(sum(veg["price_avgs"]) / len(veg["price_avgs"]), 2) if veg["price_avgs"] else None

        # pick the most frequent source type
        price_type = max(veg["price_type_counts"], key=veg["price_type_counts"].get)

        final.append({
            "name_hi": veg["name_hi"],
            "name_en": veg["name_en"],
            "unit": veg["unit"],
            "currency": veg["currency"],
            "display_price": effective_price,
            "price_type": price_type,
            "buyer_price": buyer_price,
            "sold_price": sold_price,
            "farmer_price": farmer_price,
            "price_avg": price_avg,
            "raw_text": " | ".join(veg["raw_texts"][:3]),  # keep it compact
        })

    return final


# ---------------------------------------------------------------------
# CSV loading and date selection
# ---------------------------------------------------------------------

def load_all_transcripts(csv_file: str) -> List[dict]:
    """Load all transcript rows with valid dates and non-empty transcript."""
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


def group_transcripts_by_date(rows: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(row["date"], []).append(row)
    return grouped


def pick_latest_price_bearing_date(transcripts: List[dict]) -> Tuple[Optional[str], List[dict]]:
    """
    Starting from newest date, find the first date that yields any mandi prices.
    Returns:
        (source_date, normalized_prices_for_that_date)
    """
    if not transcripts:
        return None, []

    grouped = group_transcripts_by_date(transcripts)
    dates = sorted(grouped.keys(), reverse=True)

    for date_str in dates:
        print(f"\nTrying transcript date: {date_str}")
        day_rows = grouped[date_str]
        extracted_items = []

        for row in day_rows:
            print(f"  Parsing: {row['title'][:80]}")
            prices = extract_prices(row["transcript"])
            print(f"    -> {len(prices)} normalized price item(s)")
            extracted_items.extend(prices)

        if extracted_items:
            print(f"Using latest price-bearing date: {date_str}")
            return date_str, extracted_items

        print(f"No mandi prices found on {date_str}, checking older date...")

    return None, []


# ---------------------------------------------------------------------
# Existing output loading / merging
# ---------------------------------------------------------------------

def load_existing_output() -> dict:
    """Load existing prices.json if available."""
    if not os.path.isfile(OUTPUT_FILE):
        return {}

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_history_map(existing_output: dict) -> Dict[str, dict]:
    """
    Convert existing vegetables history into a map keyed by name_en lowercase.
    """
    history_map: Dict[str, dict] = {}
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


def upsert_history_point(
    history: List[dict],
    date_str: str,
    price: float,
    price_type: str,
    buyer_price: Optional[float],
    sold_price: Optional[float],
    farmer_price: Optional[float],
    price_avg: Optional[float],
) -> None:
    """
    Insert or replace a history point for a given date.
    Keeps unique dates.
    """
    replaced = False

    for item in history:
        if item.get("date") == date_str:
            item["price"] = round(float(price), 2)
            item["price_type"] = price_type
            item["buyer_price"] = buyer_price
            item["sold_price"] = sold_price
            item["farmer_price"] = farmer_price
            item["price_avg"] = price_avg
            replaced = True
            break

    if not replaced:
        history.append({
            "date": date_str,
            "price": round(float(price), 2),
            "price_type": price_type,
            "buyer_price": buyer_price,
            "sold_price": sold_price,
            "farmer_price": farmer_price,
            "price_avg": price_avg,
        })

    history.sort(key=lambda x: x["date"])


def merge_current_prices_with_existing(existing_output: dict, source_date: str, current_prices: List[dict]) -> List[dict]:
    """
    Merge newly extracted prices into existing history.
    Preserves old history and updates only with the newest valid price-bearing date.
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

        # Update canonical metadata
        if item["name_hi"]:
            history_map[key]["name_hi"] = item["name_hi"]
        history_map[key]["name_en"] = item["name_en"]
        history_map[key]["unit"] = item["unit"]
        history_map[key]["currency"] = item["currency"]

        upsert_history_point(
            history=history_map[key]["history"],
            date_str=source_date,
            price=item["display_price"],
            price_type=item["price_type"],
            buyer_price=item["buyer_price"],
            sold_price=item["sold_price"],
            farmer_price=item["farmer_price"],
            price_avg=item["price_avg"],
        )

    # Build final vegetable output
    result = []

    for _, veg in history_map.items():
        history = sorted(veg["history"], key=lambda h: h["date"])
        if not history:
            continue

        latest_point = history[-1]
        latest = latest_point["price"]
        prev = history[-2]["price"] if len(history) >= 2 else None

        trend = "stable"
        change_pct = 0.0

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
            "latest_price": latest_point["price"],
            "latest_price_type": latest_point.get("price_type", "price_avg"),
            "buyer_price": latest_point.get("buyer_price"),
            "sold_price": latest_point.get("sold_price"),
            "farmer_price": latest_point.get("farmer_price"),
            "price_avg": latest_point.get("price_avg"),
            "trend": trend,
            "change_pct": change_pct,
            "latest_date": latest_point["date"],
            "max_price": peak_point["price"],
            "max_price_date": peak_point["date"],
            "min_price": low_point["price"],
            "min_price_date": low_point["date"],
            "history": history[-60:],  # keep enough history for charts
        })

    result.sort(key=lambda v: v["name_en"])
    return result


# ---------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------

def write_output(source_date: Optional[str], vegetables: List[dict]) -> None:
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
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
        write_output(None, [])
        return

    source_date, current_prices = pick_latest_price_bearing_date(transcripts)

    if not current_prices:
        print("No mandi-price transcripts found in CSV.")
        if existing_output:
            print("Keeping existing prices.json")
            return
        write_output(None, [])
        return

    existing_source_date = existing_output.get("latest_source_date")
    merged_vegetables = merge_current_prices_with_existing(existing_output, source_date, current_prices)

    if existing_source_date == source_date:
        print(f"Latest valid mandi-price date unchanged ({source_date}). Refreshing output.")
    else:
        print(f"Switching dashboard source date from {existing_source_date} -> {source_date}")

    write_output(source_date, merged_vegetables)


if __name__ == "__main__":
    main()
