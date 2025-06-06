from pytubefix import YouTube, Search
import pandas as pd
import os
from loguru import logger
from time import sleep
import random

# TODO: add error handling
# TODO: add logging

def search_track(artist, track):
    results = Search(f"{artist} {track}")
    if not results.videos:
        logger.error(f"No results for {artist} - {track}")
        return {
            "title": None,
            "url": None,
            "duration": None,
            "views": None
        }
    first_video = results.videos[0]
    # TODO: check if the artist name is in the channel name
    try:
        return {
            "title": first_video.title,
            "url": first_video.watch_url,
            "duration": first_video.length,
            "views": first_video.views
        }
    except Exception as e:
        logger.error(f"Error searching {artist} - {track}: {e}")
        return {
            "title": None,
            "url": None,
            "duration": None,
            "views": None
        }

def download_track(url, save_dir):
    try:
        yt = YouTube(url)
        filename = f"{yt.title}.mp3"
        # yt.streams.filter(only_audio=True).first().download(output_path=save_dir, filename=filename)
        audio = sorted(yt.streams.filter(only_audio=True), key=lambda x: int(x.abr[:-4]))[-1]  # get audio with highest bitrate
        save_path = audio.download(output_path=save_dir, filename=filename, timeout=15)
        filename = os.path.basename(save_path)
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return
    return filename

def fetch_yt_data(input_csv, output_csv, audio_dir):
    os.makedirs(audio_dir, exist_ok=True)

    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
    else:
        df = pd.read_csv(input_csv)

    if "yt_url" not in df.columns:
        df["yt_url"] = None

    for i, row in df.iterrows():
        if pd.isnull(row["yt_url"]):
            sleep(random.uniform(0, 0.2))

            track_info = search_track(row["artist"], row["title"])
            df.loc[i, "yt_title"] = track_info["title"]
            df.loc[i, "yt_url"] = track_info["url"]
            df.loc[i, "yt_duration"] = track_info["duration"]
            df.loc[i, "yt_views"] = track_info["views"]

            if not pd.isnull(track_info["duration"]) and track_info["duration"] < 400:
                if not pd.isnull(track_info["url"]):
                    filename = download_track(track_info["url"], audio_dir)
                    df.loc[i, "audio_path"] = os.path.join(audio_dir, filename) if filename else None
            else:
                logger.warning(f"Skipping {row['artist']} - {row['title']} due to duration > 400s")

        if i % 10 == 0:
            logger.info(f"{i+1}/{len(df)}")
            df.to_csv(output_csv, index=False)

    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    songs_df_path = os.path.join("data", "yt_songs_filtered_v1.csv")
    fetch_yt_data(songs_df_path, songs_df_path, "audio")