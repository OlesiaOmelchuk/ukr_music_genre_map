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

import tensorflow.compat.v1 as tf

from train import tf_define_model_and_cost

tf.disable_v2_behavior()


class MusicFeatureExtractor:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir.endswith('/') else checkpoint_dir + '/'
        self.config = self.load_config()
        self.sess = self.load_model()

    def load_config(self):
        config_json = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_json, "r") as f:
            config = json.load(f)

        musicnn_segment_len = 3 # in sec; TODO: make configurable via config (n_frames, hop_size, etc.)
        num_musicnn_segments = int(config['segment_len'] / musicnn_segment_len)
        segment_len_frames = num_musicnn_segments * config['xInput']
        config['num_musicnn_segments'] = num_musicnn_segments
        config['segment_len_frames'] = segment_len_frames

        return config

    def load_model(self):
        [x, y_, is_train, y, normalized_y, cost, calculate_accuracy, accuracy, feature_vectors_dense] = tf_define_model_and_cost(self.config)
        sess = tf.InteractiveSession()
        tf.keras.backend.set_session(sess)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, self.checkpoint_dir)
        logger.info(f'Pre-trained model loaded!')

        self.config['x'] = x
        self.config['normalized_y'] = normalized_y
        self.config['feature_vectors_dense'] = feature_vectors_dense
        self.config['is_train'] = is_train

        return sess
    
    def extract_features(self, audio_file):
        # Prepare data
        input_data = self.prepare_data(audio_file)

        # Extract features
        extract_vector = [self.config['normalized_y'], self.config['feature_vectors_dense']]
        tf_out = self.sess.run(
            extract_vector, 
            feed_dict={
                self.config['x']: input_data,
                self.config['is_train']: False
            }
        )
        
        _normalized_y, _feature_vectors_dense = tf_out

        logger.debug(f"normalized_y: {_normalized_y}")
        logger.debug(f"feature_vectors_dense: {_feature_vectors_dense}")

        return {
            'normalized_y': _normalized_y,
            'feature_vectors': _feature_vectors_dense
        }
    
    def prepare_data(self, audio_file):
        # Load audio file with pydub to correctly handle .mp3
        audio = AudioSegment.from_file(audio_file).set_channels(1).set_frame_rate(self.config['audio_rep']['resample_sr'])
        audio = np.array(audio.get_array_of_samples())
        sr = self.config['audio_rep']['resample_sr']
        num_musicnn_segments = self.config['num_musicnn_segments']
        segment_len_frames = self.config['segment_len_frames']

        audio = audio / np.max(np.abs(audio))
        audio_rep = librosa.feature.melspectrogram(y=audio, sr=sr,
                                                    hop_length=self.config['audio_rep']['hop'],
                                                    n_fft=self.config['audio_rep']['n_fft'],
                                                    n_mels=self.config['audio_rep']['n_mels']).T
        
        # Apply preprocessing
        if self.config['pre_processing'] == 'logEPS':
            audio_rep = np.log10(audio_rep + np.finfo(float).eps)
        elif self.config['pre_processing'] == 'logC':
            audio_rep = np.log10(10000 * audio_rep + 1)

        # Ensure we have enough frames
        assert audio_rep.shape[0] >= segment_len_frames, \
            f"Audio is too short ({audio_rep.shape[0]} frames), needs at least {segment_len_frames} frames"
        
        # Take random segment
        if self.config["sampling_strategy"] == 'random':
            max_start = audio_rep.shape[0] - segment_len_frames
            start_frame = random.randint(0, max_start)
        elif self.config["sampling_strategy"] == 'first':
            start_frame = 0
        else:
            raise ValueError(f"Unsupported sampling strategy: {self.config['sampling_strategy']}. Use 'random' or 'first'.")
        cropped = audio_rep[start_frame:start_frame + segment_len_frames, :]

        segments = []
        for seg_idx in range(num_musicnn_segments):
            start = seg_idx * self.config['xInput']
            end = start + self.config['xInput']
            segments.append(cropped[start:end, :])

        x = np.stack(segments)
        x = np.reshape(x, [1, num_musicnn_segments, self.config['xInput'], self.config['yInput']])
        logger.debug(f'Data shape: {x.shape}')

        return x
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str, help='Path to the audio file to process.')
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default=os.path.join('deliverables', 'trained_weights', 'ukr_songs_v1_2k_album_v5_continue'), 
        help='Path to the checkpoint directory.'
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint
    audio_path = args.audio_path

    extractor = MusicFeatureExtractor(checkpoint_dir)
    features = extractor.extract_features(audio_path)
    print(features)
