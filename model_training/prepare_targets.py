import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from loguru import logger

import musicnn_training.src.config_file as config


def generate_index_files(df_path: str, output_dir: str, config_dir: str = "configs", audio_dir: str = "audio"):
    logger.info(f"Generating index files from {df_path}")
    df = pd.read_csv(df_path)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # Prepare genre and album targets
    logger.info("Preparing genre and album targets...")

    genre_mapping_path = os.path.join(config_dir, "index_genre_mapping.json")
    genre_target_len = prepare_genre_targets(df, genre_mapping_path)
    
    album_mapping_path = os.path.join(config_dir, "index_album_mapping.json")
    album_target_len = prepare_album_targets(df, album_mapping_path)

    logger.info(f"Genre target length: {genre_target_len}, Album target length: {album_target_len}")

    # Generate index.tsv for preprocessing_librosa.py (sample_id audio_path)
    logger.info("Generating index.tsv...")

    output_file_path = os.path.join(output_dir, "index.tsv")
    df['audio_path'] = df['sample_id'].apply(lambda sample_id: os.path.join(audio_dir, f"{sample_id}.mp3"))
    df[["sample_id", "audio_path"]].to_csv(output_file_path, sep='\t', index=False, header=False)

    logger.info(f"Index file saved to {output_file_path}")

    # Split the data into train, validation, and test sets; save the corresponding .tsv files with targets
    logger.info("Splitting data into train, validation, and test sets...")
    train_df, val_df, test_df = get_train_val_test_split(df)

    def save_targets(df, target_column, file_name):
        file_path = os.path.join(output_dir, file_name)
        df[["sample_id", target_column]].to_csv(file_path, sep='\t', index=False, header=False)

    save_targets(train_df, "genre_targets", "train_genre_targets.tsv")
    save_targets(train_df, "album_targets", "train_album_targets.tsv")

    save_targets(val_df, "genre_targets", "val_genre_targets.tsv")
    save_targets(val_df, "album_targets", "val_album_targets.tsv")

    save_targets(test_df, "genre_targets", "test_genre_targets.tsv")
    save_targets(test_df, "album_targets", "test_album_targets.tsv")

    logger.info("Data split and targets saved.")

    return genre_target_len, album_target_len


def prepare_genre_targets(df: pd.DataFrame, genre_mapping_path: str = None):
    # Get all unique tags from the title_tags column
    unique_tags = sorted(set(tag for tags in df['title_tags'].apply(eval) for tag in tags))

    # Create a mapping of tag to index
    tag_to_index = {tag: idx for idx, tag in enumerate(unique_tags)}

    # Add the genre_targets column
    df['genre_targets'] = df['title_tags'].apply(
        lambda tags: [1 if i in [tag_to_index[tag] for tag in eval(tags)] else 0 for i in range(len(unique_tags))]
    )

    # Save the index-genre mapping to a JSON file
    if genre_mapping_path:
        with open(genre_mapping_path, "w", encoding="utf-8") as f:
            json.dump(tag_to_index, f, ensure_ascii=False, indent=4)

    return len(unique_tags)


def prepare_album_targets(df: pd.DataFrame, album_mapping_path: str = None):
    # Get all unique album IDs
    unique_album_ids = sorted(df['album_id'].unique())

    # Create a mapping of album_id to index
    album_id_to_index = {int(album_id): idx for idx, album_id in enumerate(unique_album_ids)}

    # Add the album_targets column
    df['album_targets'] = df['album_id'].apply(
        lambda album_id: [1 if i == album_id_to_index[album_id] else 0 for i in range(len(unique_album_ids))]
    )

    # Save the index-album_id mapping to a JSON file
    if album_mapping_path:
        with open(album_mapping_path, "w", encoding="utf-8") as f:
            json.dump(album_id_to_index, f, ensure_ascii=False, indent=4)

    return len(unique_album_ids)


def get_train_val_test_split(df):
    # TODO: modify this
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return train_df, val_df, test_df


if __name__ == "__main__":
    generate_index_files(
        df_path=config.DATA_CSV, 
        output_dir=os.path.join(config.DATA_FOLDER, "index", config.DATASET),
        config_dir=config.CONFIGS_FOLDER,
        audio_dir=config.AUDIO_FOLDER
    )