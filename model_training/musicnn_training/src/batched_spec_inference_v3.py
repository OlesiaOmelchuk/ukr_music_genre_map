import argparse
import json
import os
import time
import random
import pescador
import numpy as np
import tensorflow
import models
import config_file, shared
import pickle
from tensorflow.python.framework import ops
from loguru import logger
import shutil
import tensorflow.compat.v1 as tf

from train import tf_define_model_and_cost


tf.disable_v2_behavior()


# Configure logger to save logs to a file with date and time in a 'log' directory
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{time.strftime('inference_%Y-%m-%d_%H-%M-%S')}.log")
logger.add(log_file, level="DEBUG")


def data_gen(id, audio_repr_path, gt, pack):
    [config, sampling_strategy, segment_len_frames, num_musicnn_segments] = pack
    
    # Load audio representation
    audio_rep = pickle.load(open(os.path.join(config_file.DATA_FOLDER, audio_repr_path), 'rb'))
    
    # Apply preprocessing
    if config['pre_processing'] == 'logEPS':
        audio_rep = np.log10(audio_rep + np.finfo(float).eps)
    elif config['pre_processing'] == 'logC':
        audio_rep = np.log10(10000 * audio_rep + 1)

    # Ensure we have enough frames
    try:
        assert audio_rep.shape[0] >= segment_len_frames, \
            f"Audio is too short ({audio_rep.shape[0]} frames), needs at least {segment_len_frames} frames"
    except AssertionError as e:
        logger.error(f"Audio representation for ID {id} is too short: {e}")
        logger.error(f"Audio representation shape: {audio_rep.shape}")
        return None
    
    # Take random 2-minute crop #TODO: random or first/last
    if sampling_strategy == 'random':
        max_start = audio_rep.shape[0] - segment_len_frames
        start_frame = random.randint(0, max_start)
    elif sampling_strategy == 'first':
        start_frame = 0
    else:
        raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}. Use 'random' or 'first'.")
    cropped = audio_rep[start_frame:start_frame + segment_len_frames, :]
    
    # Split into 3-second segments # TODO: optimize
    segments = []
    for seg_idx in range(num_musicnn_segments):
        start = seg_idx * config['xInput']
        end = start + config['xInput']
        segments.append(cropped[start:end, :])
    
    # Stack segments into [num_musicnn_segments, xInput, n_mels]
    x = np.stack(segments)

    # logger.debug(f'Data shape: {x.shape}')
    
    yield {
        'X': x,          # Shape: [num_musicnn_segments, xInput, n_mels]
        'Y': gt,         # Original ground truth
        'ID': id,
    }


if __name__ == '__main__':
    # load config parameters defined in 'config_file.py'
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration',
                        help='ID in the config_file dictionary')
    args = parser.parse_args()
    config = config_file.config_train[args.configuration]

    config_inference = config_file.config_inference[args.configuration]

    # replace config with inference config
    config['model_type'] = config_inference['model_type']
    config['save_filename'] = config_inference['save_filename']
    config['gt_val'] = config_inference['gt_val']
    config['load_model'] = config_inference['load_model']

    # load config parameters used in 'preprocess_librosa.py',
    config_json = os.path.join(config_file.DATA_FOLDER, config['audio_representation_folder'], 'config.json')
    with open(config_json, "r") as f:
        params = json.load(f)
    config['audio_rep'] = params

    # set patch parameters
    if config['audio_rep']['type'] == 'waveform':
        raise ValueError('Waveform-based training is not implemented')

    elif config['audio_rep']['spectrogram_type'] == 'mel':
        config['xInput'] = config['n_frames']
        config['yInput'] = config['audio_rep']['n_mels']

    # load audio representation paths
    file_index = os.path.join(config_file.DATA_FOLDER, config['audio_representation_folder'], 'index.tsv')
    [audio_repr_paths, id2audio_repr_path] = shared.load_id2path(file_index)

    # load validation data
    file_ground_truth_val = os.path.join(config['gt_val'])
    [ids_val, id2gt_val] = shared.load_id2gt(file_ground_truth_val)

    # Generate zero vector targets if 'model_type' is 'album'
    if len(list(id2gt_val.values())[0]) != config['num_classes_dataset']:
        id2gt_val = {id: np.zeros(config['num_classes_dataset']) for id in ids_val}
        logger.info(f"Generated zero vector targets for 'album' model type with shape {config['num_classes_dataset']}")

    # set output
    config['classes_vector'] = list(range(config['num_classes_dataset']))

    logger.info(f'# Val: {len(ids_val)}')
    logger.info(f"# Classes: {config['classes_vector']}")

    # save experimental settings
    features_folder = os.path.join(config_file.DATA_FOLDER, 'inference', args.configuration)
    if not os.path.exists(features_folder):
        os.makedirs(features_folder)
    json.dump(config, open(os.path.join(features_folder, 'config.json'), 'w'))

    logger.info(f'Config file saved: {str(config)}')

    # define the musicnn segment length and number of such segments in the input audio segment of length 'segment_len'
    musicnn_segment_len = 3 # in sec; TODO: make configurable via config (n_frames, hop_size, etc.)
    num_musicnn_segments = int(config['segment_len'] / musicnn_segment_len)
    segment_len_frames = num_musicnn_segments * config['xInput']
    config['num_musicnn_segments'] = num_musicnn_segments

    # tensorflow: define model and cost
    [x, y_, is_train, y, normalized_y, cost, calculate_accuracy, accuracy, feature_vectors_dense] = tf_define_model_and_cost(config)

    sess = tf.InteractiveSession()
    tf.keras.backend.set_session(sess)

    # pescador val: define streamer
    val_pack = [config, config['sampling_strategy'], segment_len_frames, num_musicnn_segments]
    val_streams = [pescador.Streamer(data_gen, id, id2audio_repr_path[id], id2gt_val[id], val_pack) for id in ids_val]
    val_mux_stream = pescador.ChainMux(val_streams, mode='exhaustive')
    val_batch_streamer = pescador.Streamer(pescador.buffer_stream, val_mux_stream, buffer_size=config['val_batch_size'], partial=True)
    val_batch_streamer = pescador.ZMQStreamer(val_batch_streamer)
    logger.info('Successfully created val streamer!')

    # tensorflow: create a session to run the tensorflow graph
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    if config['load_model'] == None: # restore model weights from previously saved model
        logger.info('No pre-trained model to load!')
        raise ValueError('No pre-trained model to load!')
    
    saver.restore(sess, config['load_model']) # end with /!
    logger.info(f"Pre-trained model loaded from {config['load_model']}")

    all_ids = []
    all_features = []
    all_predictions = []
    all_accuracy = []
    array_val_cost = []

    logger.info('Starting inference...')
    for val_batch in val_batch_streamer:
        test_y_norm, test_cost, test_accuracy, test_feature_vectors = sess.run([normalized_y, cost, accuracy, feature_vectors_dense],
                            feed_dict={x: val_batch['X'], y_: val_batch['Y'], is_train: False, calculate_accuracy: True if "genre" not in args.configuration else False})
        array_val_cost.append(test_cost)
        all_ids.extend(list(val_batch['ID']))
        all_features.append(test_feature_vectors)
        all_predictions.append(test_y_norm)
        all_accuracy.append(test_accuracy)
        
    val_cost = np.mean(array_val_cost)
    logger.info(f"Validation cost: {val_cost}")

    val_accuracy = np.mean(all_accuracy)
    logger.info(f"Validation accuracy: {val_accuracy}")

    # Save features
    all_features = np.vstack(all_features)
    all_predictions = np.vstack(all_predictions)

    logger.info(f"Features shape: {all_features.shape}")
    logger.info(f"Predictions shape: {all_predictions.shape}")
    logger.info(f"IDs shape: {len(all_ids)}")

    # Save to compressed NPZ in the features folder
    output_file = os.path.join(features_folder, f'{config["save_filename"]}.npz')
    np.savez_compressed(
        output_file,
        ids=np.array(all_ids),
        features=all_features,
        predictions=all_predictions
    )

    logger.info(f"Batch data saved to {output_file}")

    # Copy the log file to the features folder
    log_output_file = os.path.join(features_folder, os.path.basename(log_file))
    shutil.copy(log_file, log_output_file)
    logger.info(f"Log file copied to {log_output_file}")
