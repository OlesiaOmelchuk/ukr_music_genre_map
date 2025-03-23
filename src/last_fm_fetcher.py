import pandas as pd
from time import sleep
import random
from loguru import logger
import os

from src.utils import get_soup, soup_select_wrapper, construct_lastfm_url


def get_ukr_artists_list(pages=1):
    artists = []
    for page in range(1, pages + 1):
        url = f"https://www.last.fm/tag/ukrainian/artists?page={page}"
        # TODO: add error handling
        soup = get_soup(url)
        artist_names = [a.text for a in soup_select_wrapper(soup, ".big-artist-list-title a") or []]
        artists.extend(artist_names)
    return artists

def get_artist_info(artist):
    # Artist popularity
    url = f"https://www.last.fm/music/{artist}"
    soup = get_soup(url)
    select_res = soup_select_wrapper(soup, ".header-metadata-tnew-display abbr")

    listener_count = select_res[0].get("title") if select_res else None
    scrobble_count = select_res[1].get("title") if select_res else None

    # Artist tags
    url = f"https://www.last.fm/music/{artist}/+tags"
    soup = get_soup(url)
    tags = [a.text for a in soup_select_wrapper(soup, ".big-tags-item-name a") or []]

    return {
        "listeners": listener_count, 
        "scrobbles": scrobble_count,
        "tags": tags
        }

def get_artist_tracks(artist, pages=1):
    # TODO: handle accessing unavailable pages
    tracks = []
    for page in range(1, pages + 1):
        url = f"https://www.last.fm/music/{artist}/+tracks?page={page}"
        soup = get_soup(url)
        track_names = [a.text for a in soup_select_wrapper(soup, ".chartlist-name a") or []]
        tracks.extend(track_names)
    return tracks

def get_track_info(track, artist):
    # Track popularity
    # url = f"https://www.last.fm/music/{artist}/_/{track}"
    url = construct_lastfm_url(artist, track)
    soup = get_soup(url)
    select_res = soup_select_wrapper(soup, ".header-metadata-tnew-display abbr")

    listener_count = select_res[0].get("title") if select_res else None
    scrobble_count = select_res[1].get("title") if select_res else None

    # Track tags
    tags = [a.text for a in soup_select_wrapper(soup, ".tags-list .tag a") or []]

    # Track duration
    duration = soup_select_wrapper(soup, ".catalogue-metadata-description")
    duration = duration[0].text.strip() if duration else None  # TODO: parse string to seconds

    return {
        "listeners": listener_count, 
        "scrobbles": scrobble_count,
        "tags": tags,
        "duration": duration
        }

def top_artists_to_csv(dir_path, artists_pages=1, tracks_pages=1, artists_df_path=None, songs_df_path=None):
    """
    Fetches the list of Ukrainian artists from Last.fm and saves it to a CSV file.
    Columns: artist, title, artist_listeners, artist_scrobbles, artist_tags,
             title_listeners, title_scrobbles, title_tags, title_duration
    """
    artists_save_path = os.path.join(dir_path, f"artists_{artists_pages}_pages.csv")
    songs_save_path = os.path.join(dir_path, f"songs_{artists_pages}_{tracks_pages}_pages.csv")

    # Create or load the dataframes
    if os.path.exists(artists_save_path) and os.path.exists(songs_save_path):
        logger.info(f"Files already exist: {artists_save_path}, {songs_save_path}")
        artists_df = pd.read_csv(artists_save_path)
        songs_df = pd.read_csv(songs_save_path)
    elif artists_df_path and songs_df_path:
        logger.info(f"Loading dataframes from files: {artists_df_path}, {songs_df_path}")
        artists_df = pd.read_csv(artists_df_path)
        songs_df = pd.read_csv(songs_df_path)
    else:
        artists_df = pd.DataFrame(columns=["artist", "artist_listeners", "artist_scrobbles", "artist_tags"])
        songs_df = pd.DataFrame(columns=["title", "artist", "title_listeners", "title_scrobbles", 
                                     "title_tags", "title_duration"])
    
    # Get list of Ukrainian artists
    artists = get_ukr_artists_list(artists_pages)

    logger.info(f"Found {len(artists)} artists")

    # Fetch info for each artist and their tracks
    for i, artist in enumerate(artists):
        if i < 100:
            continue

        # Add artist info to the dataframe if not already there
        if artist not in artists_df["artist"].values:
            artist_info = get_artist_info(artist)
            df_entry = {
                "artist": artist,
                "artist_listeners": artist_info["listeners"],
                "artist_scrobbles": artist_info["scrobbles"],
                "artist_tags": artist_info["tags"]
            }
            artists_df = pd.concat([artists_df, pd.DataFrame([df_entry])], ignore_index=True)

        # Add track info to the dataframe
        tracks = get_artist_tracks(artist, tracks_pages)

        for track in tracks:
            # Check if the track-artist pair is already in the dataset
            if ((songs_df["title"] == track) & (songs_df["artist"] == artist)).any():
                # print(f"Skipping {track} by {artist}")
                continue
            
            track_info = get_track_info(track, artist)
            df_entry = {
                "title": track,
                "artist": artist,
                "title_listeners": track_info["listeners"],
                "title_scrobbles": track_info["scrobbles"],
                "title_tags": track_info["tags"],
                "title_duration": track_info["duration"]
            }
            songs_df = pd.concat([songs_df, pd.DataFrame([df_entry])], ignore_index=True)

        # Intermediate save
        if i % 1 == 0:
            logger.info(f"Processed {i+1}/{len(artists)}")
            artists_df.to_csv(artists_save_path, index=False)
            songs_df.to_csv(songs_save_path, index=False)

    # Save results
    artists_df.to_csv(artists_save_path, index=False)
    songs_df.to_csv(songs_save_path, index=False)
    logger.info("Done!")


if __name__ == "__main__":
    top_artists_to_csv(
        "metadata", 
        artists_pages=10, 
        tracks_pages=2, 
        artists_df_path="metadata/artists_5_pages.csv", 
        songs_df_path="metadata/songs_5_2_pages.csv"
        )