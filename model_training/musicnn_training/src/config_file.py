import os 

DATA_CSV = os.path.join('data', 'final_dataset_v1.csv')
DATA_FOLDER =  'output/'
CONFIGS_FOLDER = 'configs'
AUDIO_FOLDER = 'audio'
INDEX_FOLDER = os.path.join('data', 'train_val_test_split_v1')

DATASET = 'ukr_songs_v1_2k'

config_preprocess = {
    'ukr_songs_v1_2k_spec': {
        'identifier': DATASET,                      # name for easy identification
        'audio_folder': AUDIO_FOLDER,               # end it with / -> this is an absolute path!
        'n_machines': 1,                          # parallelizing this process through 'n_machines'
        'machine_i': 0,                           # id number of the machine which is running this script (from 0 to n_machines-1)
        'num_processing_units': 1,               # number of parallel processes in every machine
        'type': 'time-freq',                      # kind of audio representation: 'time-freq' (only recommended option)
        'spectrogram_type': 'mel',                # 'mel' (only option) - parameters below should change according to this type
        'resample_sr': 16000,                     # sampling rate (original or the one to be resampled)
        'hop': 256,                               # hop size of the STFT
        'n_fft': 512,                             # frame size (number of freq bins of the STFT)
        'n_mels': 96,                             # number of mel bands
        'index_file': os.path.join(INDEX_FOLDER, "index.tsv"),  # list of audio representations to be computed
    }
}


epochs = 300
lr = 0.00001
batch_size = 4

config_train = {
    'ukr_songs_v1_2k_genre': {
        'name_run': 'genre_full_train_v1_2k',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join(INDEX_FOLDER, 'train_genre_targets.tsv'),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_genre_targets.tsv'),

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': epochs,                            # maximum number of epochs before stopping training
        'batch_size': batch_size,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': lr,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'sigmoid_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 39,                # the number of unique genre tags
        'val_batch_size': batch_size
    },

    'ukr_songs_v1_2k_album': {
        'name_run': '',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join(INDEX_FOLDER, 'train_album_targets.tsv'),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_album_targets.tsv'),

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250420_181925_ukr_songs_v1_2k_album') + '/',                       # set to None or absolute path to the model
        'epochs': epochs,                            # maximum number of epochs before stopping training
        'batch_size': batch_size,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': lr * 10,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': batch_size
    },

        'ukr_songs_v1_2k_album_sigmoid': {
        'name_run': '',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join(INDEX_FOLDER, 'train_album_targets.tsv'),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_album_targets.tsv'),

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250420_181925_ukr_songs_v1_2k_album') + '/',                       # set to None or absolute path to the model
        'epochs': epochs,                            # maximum number of epochs before stopping training
        'batch_size': batch_size,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': lr * 10,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'sigmoid_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': batch_size
    }
}

