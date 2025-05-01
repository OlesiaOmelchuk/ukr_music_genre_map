# Dataset of metadata for 3k Ukrainian songs
One of the deliverables of this thesis is a dataset of metadata for Ukrainian artists, their songs, and the albums they belong to. The dataset is available in CSV format and can be used for various purposes, including music recommendation systems, music analysis, and machine learning applications.

The final version of the dataset `merged_ukr_songs_v1_3k.csv` contains:
- 2,724 songs;
- 365 albums;
- 60 artists.

## Data sources
The data was collected from two main sources:
- [LastFm](https://www.last.fm/tag/ukrainian)
- [Spotify](https://www.spotify.com/)

## Data collection pipeline
The collection process consisted of the following steps with the collected data being described in next section:
1. **Ukrainian artists list**: to get the initial list of artists, we used one of the most popular music platforms, LastFM, which allows a country-level artists filtering. To fetch the list of artists, we prepared a simple Python script that parses the corresponding HTML pages with the BeautifulSoup library.
2. **Songs list and metadata**: the list of songs from the fetched artists, along with artists' and songs' metadata, was collected in the same manner. 
3. **Albums info**: One of the goals of this thesis was to test the hypothesis whether the album may be an analog of the genre label. Consequently, we required high-quality data about albums. In addition to the LastFM data, we decided to fetch more album data from the Spotify API, as it seems more reliable and requires much less preprocessing and filtering. 
    
Note: The presence of genre tags in the LastFM music platform was one of the main reasons why it was chosen as the primary source of information, even though the raw fetched data wasn't of the best quality and required an intensive review and filtering.

## Data CSV files overview
Apart from the main file with songs `merged_ukr_songs_v1_3k.csv`, we present additional datasets that provide more information about artists and albums. Below is a short description of all CSV files provided, along with the description of their data columns:

1. `merged_ukr_songs_v1_3k.csv` - merged dataset with main artists, songs and album information (LastFm + Spotify).
    - **sample_id**: Unique identifier for each song in the dataset.
    - **title**: The name of the song.
    - **artist**: The name of the artist who performed the song.
    - **title_listeners**: The number of unique listeners for the song on LastFm. A listener is a single person who has listened to the song.
    - **title_scrobbles**: The total number of times the song has been played (scrobbled) on LastFm. A scrobble represents the amount of times someone has listened to that song.
    - **title_tags**: A list of tags for the song fetched from LastFm; a list of unique tags provided in `unique_title_tags.txt`.
    - **target_genre_tags**: A list of primary genre tags assigned to the song, used for classification purposes; filtered and preprocessed version of the previous column; a list of unique tags provided in `unique_target_genre_tags.txt`.
    - **title_duration**: The duration of the song in seconds.
    - **merged_album**: The name of the album the song belongs to.

2. `spotify_artists_v1.csv` - provides additional information about the artists, including their Spotify IDs, popularity, and follower counts (Spotify).
    - **artist_id**: Unique Spotify identifier for each artist.
    - **artist_name**: The name of the artist.
    - **genres**: A list of genres associated with the artist (NOTE: mostly empty).
    - **followers**: The number of followers the artist has on Spotify.
    - **popularity**: A popularity score for the artist on Spotify (0-100).

3. `spotify_songs_albums_v1.csv` - provides detailed information about songs and the albums they belong to (Spotify).
    - **title**: The name of the song.
    - **artist**: The name of the artist who performed the song.
    - **spotify_artist**: The name of the artist as listed on Spotify.
    - **track_id**: Unique Spotify identifier for the track.
    - **artist_id**: Unique Spotify identifier for the artist.
    - **duration_ms**: The duration of the song in milliseconds.
    - **track_popularity**: A popularity score for the track on Spotify (0-100).
    - **album_id**: Unique Spotify identifier for the album.
    - **album_name**: The name of the album the song belongs to.
    - **album_release_date**: The release date of the album.
    - **total_tracks**: The total number of tracks in the album.

4. `lastfm_artists_v1.csv` - provides additional information about artists (LastFm).
    - **artist**: The name of the artist.
    - **artist_listeners**: The number of unique listeners for the artist on LastFm.
    - **artist_scrobbles**: The total number of times the artist's songs have been played (scrobbled) on LastFm.
    - **artist_tags**: A list of tags associated with the artist fetched from LastFm.

5. `lastfm_albums_v1.csv` - provides additional information about albums (LastFm).
    - **artist**: The name of the artist who released the album.
    - **album**: The name of the album.
    - **num_tracks**: The total number of tracks in the album.
    - **album_length**: The total duration of the album.
    - **release_date**: The release date of the album.
    - **listeners**: The number of unique listeners for the album on LastFm.
    - **scrobbles**: The total number of times the album's tracks have been played (scrobbled) on LastFm.
    - **tags**: A list of tags associated with the album fetched from LastFm.
    - **tracks**: A list of track names included in the album.