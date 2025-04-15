import tensorflow as tf
from loguru import logger
import argparse
import json
import os
from pydub import AudioSegment
import numpy as np
import librosa
import random
import pickle

import models as models
import shared as shared
import config_file as config_file
from models_backend import positional_encoding
from train import multi_head_attention


def tf_define_model(config):
    # tensorflow: define the model
    num_musicnn_segments = config['num_musicnn_segments']  # 120 seconds / 3-second segments
    penultinate_units = 200  # e.g., 200 units in the last layer
    segment_frames = config['xInput']  # e.g., 187 frames (3 sec)
    n_mels = config['yInput']  # e.g., 96 mel bins
    feature_vec_dim = config['feature_vector_dim'] # size of the penultimate layer of the model; will be used as a feature vector for the song

    with tf.name_scope('model'):
        # Input placeholders [batch, num_musicnn_segments, 187, 96]
        x = tf.compat.v1.placeholder(tf.float32, [None, num_musicnn_segments, segment_frames, n_mels])
        # y_ = tf.compat.v1.placeholder(tf.float32, [None, config['num_classes_dataset']])
        is_train = tf.compat.v1.placeholder(tf.bool)

        # Process all segments in parallel
        batch_size = tf.shape(x)[0]
        x_reshaped = tf.reshape(x, [-1, segment_frames, n_mels])  # [batch*num_musicnn_segments, 187, 96]

        # Original segment-level model
        # TODO: remove the final projection layer from the model (and return penultimate for example)
        with tf.variable_scope('musicnn'):
            segment_logits = models.model_number(x_reshaped, is_train, config)  # [batch*num_musicnn_segments, penultinate_units]

        logger.debug(f'Segment logits shape: {segment_logits.get_shape()}')

        # Reshape back to [batch, num_musicnn_segments, penultinate_units]
        segment_logits = tf.reshape(segment_logits, [batch_size, num_musicnn_segments, penultinate_units])  # [batch, num_musicnn_segments, penultinate_units]

        logger.debug(f'Segment logits shape: {segment_logits.get_shape()}')

        # Apply positional encoding to segment logits
        pos_embedding = positional_encoding(segment_logits.get_shape().as_list())
        segment_logits = tf.add(segment_logits, pos_embedding)  # [batch, num_musicnn_segments, penultinate_units]
        logger.debug(f'Segment logits with positional encoding shape: {segment_logits.get_shape()}')

        # TODO: batch normalization (?)

        # Calculate attention and add to segment logits
        attention_output = multi_head_attention(segment_logits, num_heads=1, d_model=penultinate_units, d_k=penultinate_units, d_v=penultinate_units) # [batch, num_musicnn_segments, penultinate_units]
        logger.debug(f'attention_output shape: {attention_output.get_shape()}')
        segment_logits = tf.add(segment_logits, attention_output)  # [batch, num_musicnn_segments, penultinate_units]

        # Aggregate segment-level features into song-level feature vector
        with tf.variable_scope('aggregation'):
            penultinate_units_agg = tf.compat.v1.layers.flatten(segment_logits)  # [batch, num_musicnn_segments*penultinate_units]
            penultinate_units_agg = tf.compat.v1.layers.batch_normalization(penultinate_units_agg, training=is_train)
            penultinate_units_agg_dropout = tf.compat.v1.layers.dropout(penultinate_units_agg, rate=0.5, training=is_train)
            feature_vectors_dense = tf.compat.v1.layers.dense(
                inputs=penultinate_units_agg_dropout,
                units=feature_vec_dim,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
            )  # [batch, feature_vec_dim]

        # Finally, apply a dense layer to project from penultinate_units to num_classes
        feature_vectors_dense = tf.compat.v1.layers.batch_normalization(feature_vectors_dense, training=is_train)
        feature_vectors_dense_dropout = tf.compat.v1.layers.dropout(feature_vectors_dense, rate=0.5, training=is_train)
        y = tf.compat.v1.layers.dense(
            inputs=feature_vectors_dense_dropout,
            units=config['num_classes_dataset'],
            activation=None,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
        ) # [batch, num_classes_dataset]

        normalized_y = tf.nn.sigmoid(y)

    return [x, is_train, y, normalized_y, feature_vectors_dense]


def data_gen(audio_file, config):
    # Load audio file with pydub to correctly handle .mp3
    audio = AudioSegment.from_file(audio_file).set_channels(1).set_frame_rate(config['audio_rep']['resample_sr'])
    audio = np.array(audio.get_array_of_samples())
    sr = config['audio_rep']['resample_sr']
    num_musicnn_segments = config['num_musicnn_segments']
    segment_len_frames = config['segment_len_frames']

    audio = audio / np.max(np.abs(audio))  # Normalize audio

    if config['audio_rep']['type'] == 'waveform':
        audio_rep = audio
        audio_rep = np.expand_dims(audio_rep, axis=1)

    elif config['audio_rep']['spectrogram_type'] == 'mel':
        audio_rep = librosa.feature.melspectrogram(y=audio, sr=sr,
                                                    hop_length=config['audio_rep']['hop'],
                                                    n_fft=config['audio_rep']['n_fft'],
                                                    n_mels=config['audio_rep']['n_mels']).T

    # Apply preprocessing
    if config['pre_processing'] == 'logEPS':
        audio_rep = np.log10(audio_rep + np.finfo(float).eps)
    elif config['pre_processing'] == 'logC':
        audio_rep = np.log10(10000 * audio_rep + 1)

    # Ensure we have enough frames
    assert audio_rep.shape[0] >= segment_len_frames, \
           f"Audio is too short ({audio_rep.shape[0]} frames), needs at least {segment_len_frames} frames"
    
    # Take random 2-minute crop
    if config["sampling_strategy"] == 'random':
        max_start = audio_rep.shape[0] - segment_len_frames
        start_frame = random.randint(0, max_start)
    elif config["sampling_strategy"] == 'first':
        start_frame = 0
    else:
        raise ValueError(f"Unsupported sampling strategy: {config['sampling_strategy']}. Use 'random' or 'first'.")
    cropped = audio_rep[start_frame:start_frame + segment_len_frames, :]
    
    # Split into 3-second segments # TODO: optimize
    segments = []
    for seg_idx in range(num_musicnn_segments):
        start = seg_idx * config['xInput']
        end = start + config['xInput']
        segments.append(cropped[start:end, :])
    
    # Stack segments into [num_musicnn_segments, xInput, n_mels]
    x = np.stack(segments)

    x = np.reshape(x, [1, num_musicnn_segments, config['xInput'], config['yInput']])

    logger.debug(f'Data shape: {x.shape}')
    
    return x

def extractor(file_name, config):
    # load config parameters used in 'preprocess_librosa.py',
    config_json = config_file.DATA_FOLDER + config['audio_representation_folder'] + 'config.json'
    with open(config_json, "r") as f:
        params = json.load(f)
    config['audio_rep'] = params

    # set patch parameters
    if config['audio_rep']['type'] == 'waveform':
        raise ValueError('Waveform-based training is not implemented')

    elif config['audio_rep']['spectrogram_type'] == 'mel':
        config['xInput'] = config['n_frames']
        config['yInput'] = config['audio_rep']['n_mels']

    # define the musicnn segment length and number of such segments in the input audio segment of length 'segment_len'
    musicnn_segment_len = 3 # in sec; TODO: make configurable via config (n_frames, hop_size, etc.)
    num_musicnn_segments = int(config['segment_len'] / musicnn_segment_len)
    segment_len_frames = num_musicnn_segments * config['xInput']
    config['num_musicnn_segments'] = num_musicnn_segments
    config['segment_len_frames'] = segment_len_frames

    # tensorflow: define model and
    [x, is_train, y, normalized_y, feature_vectors_dense] = tf_define_model(config)

    # tensorflow: loading model
    sess = tf.InteractiveSession()
    tf.keras.backend.set_session(sess)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # config['load_model'] = os.path.join("output", "experiments", "1744595687genre_test_v2") # !!!FAILS
    config['load_model'] = os.path.join("output", "experiments", "1744595060genre_test_v2/")
    saver.restore(sess, config['load_model'])
    logger.info(f'Pre-trained model loaded!')

    # prepare data
    input_data = data_gen(file_name, config)

    # extract features
    extract_vector = [y, normalized_y, feature_vectors_dense]

    tf_out = sess.run(
        extract_vector, 
        feed_dict={
            x: input_data,
            is_train: False
        }
    )
    
    _y, _normalized_y, _feature_vectors_dense = tf_out

    logger.debug(f"y: {_y}")
    logger.debug(f"normalized_y: {_normalized_y}")
    logger.debug(f"feature_vectors_dense: {_feature_vectors_dense}")

    return _y, _normalized_y, _feature_vectors_dense


if __name__ == "__main__":
    # load config parameters defined in 'config_file.py'
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration',
                        help='ID in the config_file dictionary')
    args = parser.parse_args()
    config = config_file.config_train[args.configuration]

    audio_path = os.path.join(config_file.AUDIO_FOLDER, "Я тону.mp3")
    extractor(audio_path, config)
