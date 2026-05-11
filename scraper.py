#!/usr/bin/env python3
"""
Scrape recent YouTube transcripts for a channel and append only NEW videos
to data/market_transcripts_master.csv.

Logic:
- Read existing CSV
- Find latest date already stored
- Collect existing Video_IDs
- Fetch recent channel videos
- Process only videos newer than the latest stored date, or same date but new Video_ID
- Append transcript rows to CSV

Requirements:
    pip install youtube-transcript-api
"""

import os
import csv
import time
import html
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

from youtube_transcript_api import YouTubeTranscriptApi

CSV_FILE = "data/market_transcripts_master.csv"
CHANNEL_ID = os.environ.get("CHANNEL_ID", "UCxEW_BSHnu43J8-ANnSJ80w")
MAX_NEW_VIDEOS = int(os.environ.get("MAX_NEW_VIDEOS", "5"))

CSV_HEADERS = ["Date", "Title", "Video_ID", "Transcript"]


def ensure_csv():
    os.makedirs("data", exist_ok=True)

    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        print(f"Created new CSV: {CSV_FILE}")


def parse_date_flexible(date_str: str):
    """
    Parse multiple possible date formats found in old CSVs.
    Returns datetime.date or None.
    """
    if not date_str:
        return None

    date_str = date_str.strip()

    formats = [
        "%Y-%m-%d",  # 2026-04-03
        "%d-%m-%y",  # 03-04-26
        "%d-%m-%Y",  # 03-04-2026
        "%m-%d-%y",  # if old data accidentally used this
        "%m-%d-%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue

    return None


def load_existing_data():
    """
    Read CSV and return:
      - latest_date found in the CSV
      - set of existing video IDs
    """
    latest_date = None
    existing_ids = set()

    if not os.path.isfile(CSV_FILE):
        return latest_date, existing_ids

    with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = (row.get("Video_ID") or "").strip()
            if vid:
                existing_ids.add(vid)

            row_date = parse_date_flexible(row.get("Date", ""))
            if row_date and (latest_date is None or row_date > latest_date):
                latest_date = row_date

    return latest_date, existing_ids


def fetch_channel_videos(channel_id: str):
    """
    Fetch recent videos from YouTube RSS feed.
    Returns list of dicts with date/title/video_id.
    """
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    print(f"Fetching channel feed: {url}")

    with urllib.request.urlopen(url, timeout=30) as resp:
        xml_data = resp.read()

    root = ET.fromstring(xml_data)

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "yt": "http://www.youtube.com/xml/schemas/2015",
    }

    videos = []
    for entry in root.findall("atom:entry", ns):
        video_id = entry.findtext("yt:videoId", default="", namespaces=ns).strip()
        title = entry.findtext("atom:title", default="", namespaces=ns).strip()
        published = entry.findtext("atom:published", default="", namespaces=ns).strip()

        if not video_id:
            continue

        try:
            published_dt = datetime.strptime(published, "%Y-%m-%dT%H:%M:%S%z")
            published_date = published_dt.date()
        except Exception:
            continue

        videos.append({
            "video_id": video_id,
            "title": html.unescape(title),
            "published_date": published_date,
        })

    videos.sort(key=lambda x: x["published_date"])
    print(f"Found {len(videos)} recent video(s) in feed")
    return videos


def get_transcript_text(video_id: str):
    """
    Try Hindi first, then English.
    Returns transcript text or None.
    """
    language_preferences = [
        ["hi", "hi-IN"],
        ["en"],
    ]

    for langs in language_preferences:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
            text = " ".join(
                part.get("text", "").replace("\n", " ").strip()
                for part in transcript
                if part.get("text", "").strip()
            ).strip()

            if text:
                return text
        except Exception:
            continue

    return None


def append_rows(rows):
    if not rows:
        return

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow([
                row["Date"],
                row["Title"],
                row["Video_ID"],
                row["Transcript"],
            ])


def main():
    ensure_csv()

    latest_date, existing_ids = load_existing_data()
    print(f"Latest existing CSV date: {latest_date}")
    print(f"Existing video IDs in CSV: {len(existing_ids)}")

    videos = fetch_channel_videos(CHANNEL_ID)

    new_rows = []
    processed_count = 0

    for video in videos:
        if processed_count >= MAX_NEW_VIDEOS:
            break

        video_id = video["video_id"]
        title = video["title"]
        published_date = video["published_date"]

        # Skip if already present
        if video_id in existing_ids:
            continue

        # Only process videos after latest_date, or if no latest_date exists
        if latest_date is not None and published_date < latest_date:
            continue

        print(f"Processing: {published_date} | {video_id} | {title}")

        transcript_text = get_transcript_text(video_id)
        time.sleep(1)

        if not transcript_text:
            print(f"  No transcript found for {video_id}")
            continue

        new_rows.append({
            "Date": published_date.strftime("%Y-%m-%d"),  # normalize format
            "Title": title,
            "Video_ID": video_id,
            "Transcript": transcript_text,
        })

        processed_count += 1
        print(f"  Added transcript ({len(transcript_text)} chars)")

    if new_rows:
        append_rows(new_rows)
        print(f"\nAppended {len(new_rows)} new row(s) to {CSV_FILE}")
    else:
        print("\nNo new videos to append.")

    print(f"CSV exists: {os.path.isfile(CSV_FILE)}")
    if os.path.isfile(CSV_FILE):
        print(f"CSV size: {os.path.getsize(CSV_FILE)} bytes")


if __name__ == "__main__":
    main()
