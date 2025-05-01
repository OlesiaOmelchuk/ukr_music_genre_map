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
    
> Note: The presence of genre tags in the LastFM music platform was one of the main reasons why it was chosen as the primary source of information, even though the raw fetched data wasn't of the best quality and required an intensive review and filtering.

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


## ["Datasheets for Datasets"](https://arxiv.org/pdf/1803.09010) (prepared questions)
### Motivation
-   **For what purpose was the dataset created?** Was there a specific
    task in mind? Was there a specific gap that needed to be filled?
    Please provide a description.

    The main reason for creation of this dataset was the absence of any other datasets of Ukrainian songs available for public access. To analyse the Ukrainian artists and their music we required a dataset with such data, along with the corresponding information about genre and albums distribution.

-   **Who created the dataset (e.g., which team, research group) and on
    behalf of which entity (e.g., company, institution, organization)?**

    The dataset was collected by me, Olesia Omelchuk.

### Composition
-   **How many instances are there in total (of each type, if
    appropriate)?**

    The final version of the dataset `merged_ukr_songs_v1_3k.csv` contains:
    - 2,724 songs;
    - 365 albums;
    - 60 artists.

-   **Does the dataset contain all possible instances or is it a sample
    (not necessarily random) of instances from a larger set?** If the
    dataset is a sample, then what is the larger set? Is the sample
    representative of the larger set (e.g., geographic coverage)? If so,
    please describe how this representativeness was validated/verified.
    If it is not representative of the larger set, please describe why
    not (e.g., to cover a more diverse range of instances, because
    instances were withheld or unavailable)

    This dataset is a small sample of the Ukrainian songs. The artists list was collected from 10 first pages of Ukrainian artists on LastFm webpage. The corresponding songs were further fetched for each artist (first 2 pages) and don't necessarily cover all songs from the mentioned artist.

-   **What data does each instance consist of?** "Raw" data (e.g.,
    unprocessed text or images) or features? In either case, please
    provide a description.

    The raw fetched data from LastFm and Spotify is present in the `data/raw_v1` folder. Note that some of the collected songs turned out to be russian and were further filtered from the dataset.

-   **Is there a label or target associated with each instance?** If so,
    please provide a description.

    Each song instance is asossiated with a list of genre tags and corresponding album used as targets during training.

-   **Is any information missing from individual instances?** If so,
    please provide a description, explaining why this information is
    missing (e.g., because it was unavailable). This does not include
    intentionally removed information, but might include, e.g., redacted
    text.

    Many artists are missing genre tags in the Spotify dataset. It shows the limitation of the provided API usage.

-   **Are there recommended data splits (e.g., training,
    development/validation, testing)?** If so, please provide a
    description of these splits, explaining the rationale behind them.

    We present the train/val/test datasplit used during models' training. It is specified in the `data/train_val_test_split_v1` folder.

-   **Are there any errors, sources of noise, or redundancies in the
    dataset?** If so, please provide a description.

    Some tags fetched from LastFm are set by the listeners themselves, thus may be not 100% accurate. Same applies to album information.

-   **Is the dataset self-contained, or does it link to or otherwise
    rely on external resources (e.g., websites, tweets, other
    datasets)?** If it links to or relies on external resources, a) are
    there guarantees that they will exist, and remain constant, over
    time; b) are there official archival versions of the complete
    dataset (i.e., including the external resources as they existed at
    the time the dataset was created); c) are there any restrictions
    (e.g., licenses, fees) associated with any of the external resources
    that might apply to a future user? Please provide descriptions of
    all external resources and any restrictions associated with them, as
    well as links or other access points, as appropriate.

    Due to copyright issues we don't provide audio files for the fetched list of songs. These have to be manually collected.

### Collection process

-   **How was the data associated with each instance acquired?** Was the
    data directly observable (e.g., raw text, movie ratings), reported
    by subjects (e.g., survey responses), or indirectly inferred/derived
    from other data (e.g., part-of-speech tags, model-based guesses for
    age or language)? If data was reported by subjects or indirectly
    inferred/derived from other data, was the data validated/verified?
    If so, please describe how.

    This data was directly observed in the two music platforms: LastFm and Spotify.

-   **What mechanisms or procedures were used to collect the data (e.g.,
    hardware apparatus or sensor, manual human curation, software
    program, software API)?** How were these mechanisms or procedures
    validated?

    The dataset was collected using custom python scripts for LastFm HTML pages fetching and parsing + python usage of Spotify developers API.

-   **Who was involved in the data collection process (e.g., students,
    crowdworkers, contractors) and how were they compensated (e.g., how
    much were crowdworkers paid)?**

    Only the author of thesis, Olesia Omelchuk.

### Preprocessing / cleaning / labeling

-   **Was any preprocessing/cleaning/labeling of the data done (e.g.,
    discretization or bucketing, tokenization, part-of-speech tagging,
    SIFT feature extraction, removal of instances, processing of missing
    values)?** If so, please provide a description. If not, you may skip
    the remainder of the questions in this section.

    The whole preprocessing and filtering pipeline is described in the corresponding `.ipynb` notebooks in the `dataset_notebooks` folder.

-   **Was the "raw" data saved in addition to the
    preprocessed/cleaned/labeled data (e.g., to support unanticipated
    future uses)?** If so, please provide a link or other access point
    to the "raw" data.

    Yes, the raw data is provided in the `data/raw_v1` folder.