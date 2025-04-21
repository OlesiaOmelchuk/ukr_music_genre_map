import os
import librosa
from joblib import Parallel, delayed
import json
import config_file
import argparse
import pickle
import numpy as np
from pathlib import Path
from pydub import AudioSegment
import time
from loguru import logger

DEBUG = False

# Configure logger to save logs to a file with date and time in a 'log' directory
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"preprocess_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
logger.add(log_file, level="DEBUG")


def compute_audio_repr(audio_file, audio_repr_file, config):

    # audio, sr = librosa.load(audio_file, sr=config['resample_sr'])

    # Load audio file with pydub to correctly handle .mp3
    audio = AudioSegment.from_file(audio_file).set_channels(1).set_frame_rate(config['resample_sr'])
    audio = np.array(audio.get_array_of_samples())
    sr = config['resample_sr']

    audio = audio / np.max(np.abs(audio))  # Normalize audio

    logger.debug(f"Audio file: {audio_file}, Sample rate: {sr}, Length: {len(audio)}")

    if len(audio) < config['resample_sr'] * 120:
        logger.warning(f"Audio file shorter than 2 min: {audio_file}")

    if config['type'] == 'waveform':
        audio_repr = audio
        audio_repr = np.expand_dims(audio_repr, axis=1)

    elif config['spectrogram_type'] == 'mel':
        audio_repr = librosa.feature.melspectrogram(y=audio, sr=sr,
                                                    hop_length=config['hop'],
                                                    n_fft=config['n_fft'],
                                                    n_mels=config['n_mels']).T
    # Compute length
    logger.debug(f"Audio representation shape: {audio_repr.shape}")
    length = audio_repr.shape[0]

    # Transform to float16 (to save storage, and works the same)
    audio_repr = audio_repr.astype(np.float16)

    # Write results:
    if audio_repr_file is not None:
        with open(audio_repr_file, "wb") as f:
            pickle.dump(audio_repr, f)  # audio_repr shape: NxM

    return length


def do_process(files, index, config):

    try:
        [id, audio_file, audio_repr_file] = files[index]
        os.makedirs(os.path.dirname(audio_repr_file), exist_ok=True)
        # compute audio representation (pre-processing)
        length = compute_audio_repr(audio_file, audio_repr_file, config)
        # index.tsv writing
        with open(os.path.join(config_file.DATA_FOLDER, config['audio_representation_folder'], "index_" + str(config['machine_i']) + ".tsv"), "a", encoding="utf-8") as fw:
            fw.write("%s\t%s\t%s\n" % (id, audio_repr_file[len(config_file.DATA_FOLDER):], audio_file))
        logger.info(str(index) + '/' + str(len(files)) + ' Computed: %s' % audio_file)

    except Exception as e:
        ferrors = open(os.path.join(config_file.DATA_FOLDER, config['audio_representation_folder'], "errors" + str(config['machine_i']) + ".txt"), "a")
        ferrors.write(audio_file + "\n")
        ferrors.write(str(e))
        ferrors.close()
        logger.error('Error computing audio representation: ', audio_file)
        logger.error(str(e))


def process_files(files, config):

    if DEBUG:
        logger.warning('WARNING: Parallelization is not used!')
        for index in range(0, len(files)):
            do_process(files, index, config)

    else:
        Parallel(n_jobs=config['num_processing_units'])(
            delayed(do_process)(files, index, config) for index in range(0, len(files)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('configurationID', help='ID of the configuration dictionary')
    args = parser.parse_args()
    config = config_file.config_preprocess[args.configurationID]

    config['audio_representation_folder'] = "audio_representation/%s__%s/" % (config['identifier'], config['type'])

    logger.info(f"Hop length: {config['hop']}, FFT size: {config['n_fft']}, Number of Mel bands: {config['n_mels']}")
    
    # set audio representations folder
    representations_dir = os.path.join(config_file.DATA_FOLDER, config['audio_representation_folder'])
    if not os.path.exists(representations_dir):
        os.makedirs(representations_dir)
    else:
        print("WARNING: already exists a folder with this name!"
              "\nThis is expected if you are splitting computations into different machines.."
              "\n..because all these machines are writing to this folder. Otherwise, check your config_file!")

    # list audios to process: according to 'index_file'
    files_to_convert = []
    with open(os.path.join(config["index_file"]), encoding="utf-8") as f:
        for line in f.readlines():
            id, audio_path = line.strip().split("\t")
            audio_repr = os.path.splitext(os.path.basename(audio_path))[0] + ".pk"  # .npy or .pk
            files_to_convert.append(
                (
                    id, 
                    audio_path,
                    os.path.join(representations_dir, audio_repr)
                )
            )

    # compute audio representation
    if config['machine_i'] == config['n_machines'] - 1:
        process_files(files_to_convert[int(len(files_to_convert) / config['n_machines']) * (config['machine_i']):], config)
        # we just save parameters once! In the last thread run by n_machine-1!
        json.dump(config, open(os.path.join(representations_dir, "config.json"), "w"))
    else:
        first_index = int(len(files_to_convert) / config['n_machines']) * (config['machine_i'])
        second_index = int(len(files_to_convert) / config['n_machines']) * (config['machine_i'] + 1)
        assigned_files = files_to_convert[first_index:second_index]
        process_files(assigned_files, config)

    logger.info("Audio representation folder: " + config_file.DATA_FOLDER + config['audio_representation_folder'])
