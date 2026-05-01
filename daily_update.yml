# -*- coding: utf-8 -*-
"""
Scrapes Hindi YouTube channel transcripts and appends to CSV.
Designed to run incrementally (skips already-processed videos).
"""

import os
import re
import csv
import yt_dlp

CHANNEL_ID = os.environ.get("CHANNEL_ID", "UCxEW_BSHnu43J8-ANnSJ80w")
CSV_FILENAME = "data/market_transcripts_master.csv"


def get_existing_video_ids(csv_file):
    existing_ids = set()
    if os.path.isfile(csv_file):
        with open(csv_file, mode="r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and "Video_ID" in header:
                id_index = header.index("Video_ID")
                for row in reader:
                    if len(row) > id_index:
                        existing_ids.add(row[id_index])
    return existing_ids


def get_channel_videos(channel_id):
    print("Scanning channel for videos...")
    if channel_id.startswith("UC"):
        uploads_playlist_id = "UU" + channel_id[2:]
    else:
        uploads_playlist_id = channel_id

    playlist_url = f"https://www.youtube.com/playlist?list={uploads_playlist_id}"
    ydl_opts = {"extract_flat": True, "quiet": True}
    videos = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(playlist_url, download=False)
            if "entries" in info:
                for entry in info["entries"]:
                    if entry.get("url") and entry.get("id"):
                        videos.append({"id": entry["id"], "url": entry["url"]})
        except Exception as e:
            print(f"Error fetching channel info: {e}")
    return videos


def clean_vtt_text(filepath):
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
            clean_line = re.sub(r"<[^>]+>", "", line)
            if not cleaned_lines or cleaned_lines[-1] != clean_line:
                cleaned_lines.append(clean_line)
        return " ".join(cleaned_lines)
    except Exception as e:
        print(f"Error reading VTT file: {e}")
        return ""


def process_video(video_url, video_id, csv_file):
    ydl_opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["hi"],
        "subtitlesformat": "vtt",
        "skip_download": True,
        "sleep_interval_requests": 2,
        "sleep_interval_subtitles": 3,
        "extractor_args": {
            "youtube": ["player_client=ios", "player_client=android"]
        },
        "format": "bestvideo+bestaudio/best",
        "outtmpl": "/tmp/%(id)s.%(ext)s",
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(video_url, download=True)
            title = info_dict.get("title", "Unknown Title")
            raw_date = info_dict.get("upload_date", "Unknown Date")
            if len(raw_date) == 8 and raw_date.isdigit():
                formatted_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            else:
                formatted_date = raw_date

            vtt_path = None
            for lang in ["hi", "en"]:
                potential_path = f"/tmp/{video_id}.{lang}.vtt"
                if os.path.exists(potential_path):
                    vtt_path = potential_path
                    break

            transcript_text = ""
            if vtt_path:
                print(f"  -> Cleaning subtitles for: {title[:60]}")
                transcript_text = clean_vtt_text(vtt_path)
                os.remove(vtt_path)
            else:
                print(f"  -> No subtitles found for: {title[:60]}")

            file_exists = os.path.isfile(csv_file)
            with open(csv_file, mode="a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Date", "Title", "Video_ID", "Transcript"])
                writer.writerow([formatted_date, title, video_id, transcript_text])
            print(f"  -> Saved.")
        except Exception as e:
            print(f"  -> Error: {e}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    processed_ids = get_existing_video_ids(CSV_FILENAME)
    print(f"Already in database: {len(processed_ids)} videos")

    all_videos = get_channel_videos(CHANNEL_ID)
    print(f"Total on channel: {len(all_videos)} videos")

    videos_to_process = [v for v in all_videos if v["id"] not in processed_ids]
    # In CI, only process the latest 5 to stay within time limits
    max_new = int(os.environ.get("MAX_NEW_VIDEOS", 5))
    videos_to_process = videos_to_process[:max_new]
    print(f"Processing: {len(videos_to_process)} new videos\n")

    for i, video in enumerate(videos_to_process, 1):
        print(f"[{i}/{len(videos_to_process)}] ID: {video['id']}")
        process_video(video["url"], video["id"], CSV_FILENAME)

    print("\nDone.")
