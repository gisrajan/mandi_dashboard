# -*- coding: utf-8 -*-
"""
YouTube transcript scraper for mandi dashboard.

What it does:
- Reads existing data/market_transcripts_master.csv
- Finds latest date already stored
- Collects existing Video_ID values
- Scans the channel uploads playlist
- Processes only videos newer than the latest CSV date
  OR videos from the same/newer date that are not already in CSV
- Downloads subtitles, cleans transcript text
- Appends new rows to data/market_transcripts_master.csv
"""

import os
import time
import re
import csv
from datetime import datetime
import yt_dlp

# --- Configuration ---
CHANNEL_ID = os.environ.get("CHANNEL_ID", "UCxEW_BSHnu43J8-ANnSJ80w")
MAX_NEW_VIDEOS = int(os.environ.get("MAX_NEW_VIDEOS", "5"))
CSV_FILE = "data/market_transcripts_master.csv"
COOKIE_FILE = "data/youtube_cookies.txt"


def ensure_data_dir_and_csv():
    """Create data directory and CSV file if they do not exist."""
    os.makedirs("data", exist_ok=True)

    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Title", "Video_ID", "Transcript"])
        print(f"Created new CSV: {CSV_FILE}")

    # Set up cookies if passed from environment secrets
    env_cookies = os.environ.get("YOUTUBE_COOKIES")
    if env_cookies:
        with open(COOKIE_FILE, "w", encoding="utf-8") as f:
            f.write(env_cookies.strip())
        print("Generated temporary YouTube cookie file for authentication.")


def parse_date_flexible(date_str):
    """
    Parse multiple possible date formats from historical CSV data.
    Returns a datetime.date or None.
    """
    if not date_str:
        return None

    date_str = date_str.strip()
    formats = [
        "%Y-%m-%d",  # 2026-04-03
        "%d-%m-%y",  # 03-04-26
        "%d-%m-%Y",  # 03-04-2026
        "%m-%d-%y",  # fallback if older data used MM-DD-YY
        "%m-%d-%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue

    return None


def get_existing_video_ids_and_latest_date(csv_file):
    """
    Reads the CSV and returns:
    - a set of existing Video IDs
    - the latest date found in the CSV
    """
    existing_ids = set()
    latest_date = None

    if os.path.isfile(csv_file):
        with open(csv_file, mode="r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)

            for row in reader:
                video_id = (row.get("Video_ID") or "").strip()
                if video_id:
                    existing_ids.add(video_id)

                row_date = parse_date_flexible(row.get("Date", ""))
                if row_date and (latest_date is None or row_date > latest_date):
                    latest_date = row_date

    return existing_ids, latest_date


def get_channel_videos(channel_id):
    """
    Scrape the channel's master uploads playlist using yt-dlp.
    This includes videos, shorts, and livestream uploads.
    """
    print("Scanning channel for videos... This might take a minute.")

    # Convert Channel ID (UC...) to Uploads Playlist ID (UU...)
    if channel_id.startswith("UC"):
        uploads_playlist_id = "UU" + channel_id[2:]
    else:
        uploads_playlist_id = channel_id

    playlist_url = f"https://www.youtube.com/playlist?list={uploads_playlist_id}"

    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "ignoreerrors": True,
    }

    if os.path.exists(COOKIE_FILE):
        ydl_opts["cookiefile"] = COOKIE_FILE

    videos = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(playlist_url, download=False)
            if info and "entries" in info:
                for entry in info["entries"]:
                    if not entry:
                        continue

                    video_id = entry.get("id")
                    url = entry.get("url") or entry.get("webpage_url")
                    title = entry.get("title", "")

                    if video_id and url:
                        if not str(url).startswith("http"):
                            url = f"https://www.youtube.com/watch?v={video_id}"

                        videos.append({
                            "id": video_id,
                            "url": url,
                            "title": title,
                        })

        except Exception as e:
            print(f"Error fetching channel info: {e}")

    print(f"Found {len(videos)} total videos on the channel.")
    return videos


def clean_vtt_text(filepath):
    """Reads a VTT subtitle file and strips timestamps and HTML tags."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        cleaned_lines = []
        for line in lines:
            line = line.strip()

            if (
                not line
                or "WEBVTT" in line
                or "-->" in line
                or line.startswith("Kind:")
                or line.startswith("Language:")
            ):
                continue

            # Remove HTML tags
            clean_line = re.sub(r"<[^>]+>", "", line)

            # Avoid consecutive duplicate lines
            if clean_line and (not cleaned_lines or cleaned_lines[-1] != clean_line):
                cleaned_lines.append(clean_line)

        return " ".join(cleaned_lines).strip()

    except Exception as e:
        print(f"Error reading VTT file: {e}")
        return ""


def find_subtitle_file(base_filename):
    """
    Find downloaded subtitle file.
    Tries Hindi and English subtitle naming patterns.
    """
    candidates = [
        f"{base_filename}.hi.vtt",
        f"{base_filename}.hi-IN.vtt",
        f"{base_filename}.en.vtt",
        f"{base_filename}.en-US.vtt",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def process_video(video_url, video_id):
    """
    Downloads subtitles, extracts metadata, cleans text,
    and returns a row dict ready to append to CSV.
    """
    ydl_opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["hi", "hi-IN", "en", "en-US"],
        "subtitlesformat": "vtt",
        "skip_download": True,

        # Throttling to bypass anti-scraping
        "sleep_interval_requests": 3,
        "sleep_interval_subtitles": 3,

        # Mitigate cloud runner bot detection
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web_embedded"],
                "skip": ["dash", "hls"]
            }
        },
        "outtmpl": "%(title)s.%(id)s.%(ext)s",
        "quiet": True,
        "ignoreerrors": True,
    }

    # Apply cookie file if present
    if os.path.exists(COOKIE_FILE):
        ydl_opts["cookiefile"] = COOKIE_FILE

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(video_url, download=True)
            if not info_dict:
                print(f" -> No metadata returned for video {video_id} (Check authentication/bot walls)")
                return None

            title = info_dict.get("title", "Unknown Title")
            raw_date = info_dict.get("upload_date", "")

            # Convert YYYYMMDD -> YYYY-MM-DD
            if len(raw_date) == 8 and raw_date.isdigit():
                formatted_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            else:
                formatted_date = raw_date

            base_filename = os.path.splitext(ydl.prepare_filename(info_dict))[0]
            vtt_path = find_subtitle_file(base_filename)

            transcript_text = ""
            if vtt_path:
                print(" -> Cleaning text from subtitles...")
                transcript_text = clean_vtt_text(vtt_path)

                try:
                    os.remove(vtt_path)
                except Exception:
                    pass
            else:
                print(" -> No subtitles found or couldn't be extracted for this video.")

            return {
                "Date": formatted_date,
                "Title": title,
                "Video_ID": video_id,
                "Transcript": transcript_text,
            }

        except Exception as e:
            print(f" -> Error while processing {video_id}: {e}")
            return None


def append_rows_to_csv(csv_file, rows):
    """Append rows to the CSV."""
    if not rows:
        return

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Date", "Title", "Video_ID", "Transcript"])

        for row in rows:
            writer.writerow([
                row["Date"],
                row["Title"],
                row["Video_ID"],
                row["Transcript"],
            ])


def main():
    ensure_data_dir_and_csv()

    # 1. Read existing CSV state
    processed_ids, latest_date = get_existing_video_ids_and_latest_date(CSV_FILE)
    print(f"Found {len(processed_ids)} videos already in the database.")
    print(f"Latest date in CSV: {latest_date}")

    # 2. Get all channel videos
    all_videos = get_channel_videos(CHANNEL_ID)

    # 3. Filter videos:
    #    - skip already processed IDs
    candidate_videos = []
    for vid in all_videos:
        video_id = vid["id"]

        if video_id in processed_ids:
            continue

        candidate_videos.append(vid)

    print(f"Unprocessed videos found: {len(candidate_videos)}")

    new_rows = []
    processed_count = 0

    # 4. Process candidate videos one by one
    for index, video in enumerate(candidate_videos, start=1):
        if processed_count >= MAX_NEW_VIDEOS:
            break

        print(f"\n--- Processing Video {index} ---")
        print(f"ID: {video['id']}")
        print(f"URL: {video['url']}")

        row = process_video(video["url"], video["id"])
        time.sleep(1)

        if not row:
            continue

        row_date = parse_date_flexible(row["Date"])

        # If CSV already has data, skip older videos
        if latest_date is not None and row_date is not None and row_date <= latest_date:
            print(f" -> Skipping older video ({row['Date']})")
            continue

        new_rows.append(row)
        processed_count += 1
        print(" -> Added to append queue.")

    # 5. Append new rows
    if new_rows:
        append_rows_to_csv(CSV_FILE, new_rows)
        print(f"\nAppended {len(new_rows)} new row(s) to {CSV_FILE}")
    else:
        print("\nNo new videos to append.")

    # Clean up cookie file at execution end
    if os.path.exists(COOKIE_FILE):
        try:
            os.remove(COOKIE_FILE)
        except Exception:
            pass

    print("\nChannel extraction complete!")


if __name__ == "__main__":
    main()
