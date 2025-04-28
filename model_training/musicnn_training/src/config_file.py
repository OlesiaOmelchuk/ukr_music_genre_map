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

    'ukr_songs_v1_2k_genre_v2': {
        'name_run': 'genre_full_train_v1_2k_v2',
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

    'ukr_songs_v1_2k_genre_v3': {
        'name_run': 'genre_v3_meanmax',
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
        'aggregation': 'meanmaxpooling',                # 'meanpooling' or 'flatten' (default)
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': 2000,                            # maximum number of epochs before stopping training
        'batch_size': 8,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 1e-4,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'sigmoid_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 39,                # the number of unique genre tags
        'val_batch_size': 8
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

    'ukr_songs_v1_2k_album_v2': {
        'name_run': 'album_v2',
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
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': epochs,                            # maximum number of epochs before stopping training
        'batch_size': batch_size,                         # batch size during training
        'weight_decay': 1e-4,                     # None or value for the regularization parameter
        'learning_rate': lr * 10,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': batch_size
    },

    'ukr_songs_v1_2k_album_v2_continue': {
        'name_run': 'album_v2',
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
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250423_160502_ukr_songs_v1_2k_album_v2') + '/',                       # set to None or absolute path to the model
        'epochs': epochs,                            # maximum number of epochs before stopping training
        'batch_size': batch_size,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': lr,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': batch_size
    },

    'ukr_songs_v1_2k_album_v3': {
        'name_run': 'album_v3',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join(INDEX_FOLDER, 'train_album_targets.tsv'),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_album_targets.tsv'),
        'gt_train_subset': os.path.join(INDEX_FOLDER, 'train_album_targets_subset.tsv'), # for testing purposes

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'aggregation': 'meanpooling',                # 'meanpooling' or 'flatten' (default)
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': 2000,                            # maximum number of epochs before stopping training
        'batch_size': 6,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 1e-4,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': 6
    },

    'ukr_songs_v1_2k_album_v4': {
        'name_run': 'album_v4_no_dropout',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join(INDEX_FOLDER, 'train_album_targets.tsv'),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_album_targets.tsv'),
        'gt_train_subset': os.path.join(INDEX_FOLDER, 'train_album_targets_subset.tsv'), # for testing purposes

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'aggregation': 'meanpooling',                # 'meanpooling' or 'flatten' (default)
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': 2000,                            # maximum number of epochs before stopping training
        'batch_size': 8,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 1e-4,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': 8
    },

    'ukr_songs_v1_2k_album_v5': {
        'name_run': 'album_v5_meanmax',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join(INDEX_FOLDER, 'train_album_targets.tsv'),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_album_targets.tsv'),
        'gt_train_subset': os.path.join(INDEX_FOLDER, 'train_album_targets_subset.tsv'), # for testing purposes

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'aggregation': 'meanmaxpooling',                # 'meanpooling' or 'flatten' (default)
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': 2000,                            # maximum number of epochs before stopping training
        'batch_size': 8,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 1e-4,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': 8
    },

    'ukr_songs_v1_2k_album_v5_continue': {
        'name_run': 'album_v5_meanmax_continue',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join(INDEX_FOLDER, 'train_album_targets.tsv'),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_album_targets.tsv'),
        'gt_train_subset': os.path.join(INDEX_FOLDER, 'train_album_targets_subset.tsv'), # for testing purposes

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'aggregation': 'meanmaxpooling',                # 'meanpooling' or 'flatten' (default)
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250424_115123_ukr_songs_v1_2k_album_v5') + '/',                       # set to None or absolute path to the model
        'epochs': 2000,                            # maximum number of epochs before stopping training
        'batch_size': 8,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 1e-4,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': 8
    },

    'ukr_songs_v1_2k_album_v6': {
        'name_run': 'album_v6_attention_pooling',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join(INDEX_FOLDER, 'train_album_targets.tsv'),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_album_targets.tsv'),
        'gt_train_subset': os.path.join(INDEX_FOLDER, 'train_album_targets_subset.tsv'), # for testing purposes

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'aggregation': 'attention',                # 'meanpooling' or 'flatten' (default)
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': 2000,                            # maximum number of epochs before stopping training
        'batch_size': 8,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 1e-4,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': 8
    },

    'ukr_songs_v1_2k_album_v7': {
        'name_run': 'album_v7_no_weight_decay',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join(INDEX_FOLDER, 'train_album_targets.tsv'),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_album_targets.tsv'),
        'gt_train_subset': os.path.join(INDEX_FOLDER, 'train_album_targets_subset.tsv'), # for testing purposes

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'aggregation': 'attention',                # 'meanpooling' or 'flatten' (default)
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': 2000,                            # maximum number of epochs before stopping training
        'batch_size': 8,                         # batch size during training
        'weight_decay': None,                     # None or value for the regularization parameter
        'learning_rate': 1e-4,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': 8
    },

    'ukr_songs_v1_2k_album_v8': {
        'name_run': 'album_v8_dim_256',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join(INDEX_FOLDER, 'train_album_targets.tsv'),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_album_targets.tsv'),
        'gt_train_subset': os.path.join(INDEX_FOLDER, 'train_album_targets_subset.tsv'), # for testing purposes

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'aggregation': 'attention',                # 'meanpooling' or 'flatten' (default)
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': 2000,                            # maximum number of epochs before stopping training
        'batch_size': 8,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 1e-4,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 256,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': 8
    },

    'ukr_songs_v1_2k_album_v9': {
        'name_run': 'album_v8_dim_256_dropout',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join(INDEX_FOLDER, 'train_album_targets.tsv'),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_album_targets.tsv'),
        'gt_train_subset': os.path.join(INDEX_FOLDER, 'train_album_targets_subset.tsv'), # for testing purposes

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'aggregation': 'attention',                # 'meanpooling' or 'flatten' (default)
        'feat_vec_dropout': True,                # True or False
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': 2000,                            # maximum number of epochs before stopping training
        'batch_size': 8,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 1e-4,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 256,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': 8
    },

    'ukr_songs_v1_2k_album_v10': {
        'name_run': 'album_v8_dim_256_lr_1e-3',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join(INDEX_FOLDER, 'train_album_targets.tsv'),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_album_targets.tsv'),
        'gt_train_subset': os.path.join(INDEX_FOLDER, 'train_album_targets_subset.tsv'), # for testing purposes

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'aggregation': 'attention',                # 'meanpooling' or 'flatten' (default)
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': 2000,                            # maximum number of epochs before stopping training
        'batch_size': 8,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 1e-3,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 256,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': 8
    },

    'ukr_songs_v1_2k_album_v10_continue': {
        'name_run': 'album_v10_continue_lr_1e-4',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join(INDEX_FOLDER, 'train_album_targets.tsv'),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_album_targets.tsv'),
        'gt_train_subset': os.path.join(INDEX_FOLDER, 'train_album_targets_subset.tsv'), # for testing purposes

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'aggregation': 'attention',                # 'meanpooling' or 'flatten' (default)
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250424_160333_ukr_songs_v1_2k_album_v10') + '/',                       # set to None or absolute path to the model
        'epochs': 2000,                            # maximum number of epochs before stopping training
        'batch_size': 8,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 1e-4,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 256,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': 8
    },

    'ukr_songs_v1_2k_album_v11': {
        'name_run': 'album_v6_lr_1e-3',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join(INDEX_FOLDER, 'train_album_targets.tsv'),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_album_targets.tsv'),
        'gt_train_subset': os.path.join(INDEX_FOLDER, 'train_album_targets_subset.tsv'), # for testing purposes

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'aggregation': 'attention',                # 'meanpooling' or 'flatten' (default)
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': 2000,                            # maximum number of epochs before stopping training
        'batch_size': 8,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 1e-3,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': 8
    },
}

config_inference = {
    'ukr_songs_v1_2k_genre': {
        'model_type': 'genre',
        'save_filename': 'ukr_songs_v1_2k_genre',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_val': os.path.join(INDEX_FOLDER, 'test_genre_targets.tsv'),

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'first',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250420_072717_ukr_songs_v1_2k_genre') + '/',                       # set to None or absolute path to the model
        'loss_function': 'sigmoid_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 39,                # the number of unique genre tags
        'val_batch_size': batch_size
    },

    'ukr_songs_v1_2k_genre': {
        'model_type': 'genre',
        'save_filename': 'val_ukr_songs_v1_2k_genre',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_genre_targets.tsv'),

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'first',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250420_072717_ukr_songs_v1_2k_genre') + '/',                       # set to None or absolute path to the model
        'loss_function': 'sigmoid_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 39,                # the number of unique genre tags
        'val_batch_size': batch_size
    },

    'ukr_songs_v1_2k_album_sigmoid': {
        'model_type': 'album',
        'save_filename': '20250420_191325_ukr_songs_v1_2k_album_sigmoid',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_val': os.path.join(INDEX_FOLDER, 'test_album_targets.tsv'),

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'first',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250420_191325_ukr_songs_v1_2k_album_sigmoid') + '/',                       # set to None or absolute path to the model
        'loss_function': 'sigmoid_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': batch_size
    },

    'ukr_songs_v1_2k_album': {
        'model_type': 'album',
        'save_filename': '20250420_201313_ukr_songs_v1_2k_album',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_val': os.path.join(INDEX_FOLDER, 'test_album_targets.tsv'),

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'first',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250420_201313_ukr_songs_v1_2k_album') + '/',                       # set to None or absolute path to the model
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': batch_size
    },

    'ukr_songs_v1_2k_genre_v2': {
        'model_type': 'genre',
        'save_filename': 'ukr_songs_v1_2k_genre_v2',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_val': os.path.join(INDEX_FOLDER, 'test_genre_targets.tsv'),

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'first',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250423_162437_ukr_songs_v1_2k_genre_v2') + '/',                       # set to None or absolute path to the model
        'loss_function': 'sigmoid_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 39,                # the number of unique genre tags
        'val_batch_size': batch_size
    },

        'ukr_songs_v1_2k_genre_v2': {
        'model_type': 'genre',
        'save_filename': 'val_ukr_songs_v1_2k_genre_v2',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_val': os.path.join(INDEX_FOLDER, 'val_genre_targets.tsv'),

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'first',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250423_162437_ukr_songs_v1_2k_genre_v2') + '/',                       # set to None or absolute path to the model
        'loss_function': 'sigmoid_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 39,                # the number of unique genre tags
        'val_batch_size': batch_size
    },

    'ukr_songs_v1_2k_album_v2': {
        'model_type': 'album_v2',
        'save_filename': 'ukr_songs_v1_2k_album_v2_dense_attention',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_val': os.path.join(INDEX_FOLDER, 'test_album_targets.tsv'),

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'first',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250423_160502_ukr_songs_v1_2k_album_v2') + '/',                       # set to None or absolute path to the model
        'loss_function': 'softmax_cross_entropy', # 'sigmoid_cross_entropy' or 'softmax_cross_entropy' (for multi-label or single-label classification)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 305,                # the number of unique genre tags
        'val_batch_size': batch_size
    },
    'ukr_songs_v1_2k_album_v3': {
        'model_type': 'album_v3',
        'save_filename': 'ukr_songs_v1_2k_album_v3',
        'gt_val': os.path.join(INDEX_FOLDER, 'test_album_targets.tsv'),                    # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250424_091658_ukr_songs_v1_2k_album_v3') + '/',                       # set to None or absolute path to the model
    },
    'ukr_songs_v1_2k_album_v4': {
        'model_type': 'album_v4',
        'save_filename': 'ukr_songs_v1_2k_album_v4',
        'gt_val': os.path.join(INDEX_FOLDER, 'test_album_targets.tsv'),                    # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250424_101827_ukr_songs_v1_2k_album_v4') + '/',                       # set to None or absolute path to the model
    },
    'ukr_songs_v1_2k_album_v5': {
        'model_type': 'album_v5',
        'save_filename': 'ukr_songs_v1_2k_album_v5',
        'gt_val': os.path.join(INDEX_FOLDER, 'test_album_targets.tsv'),                    # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250424_115123_ukr_songs_v1_2k_album_v5') + '/',                       # set to None or absolute path to the model
    },
    'ukr_songs_v1_2k_album_v6': {
        'model_type': 'album_v6',
        'save_filename': 'ukr_songs_v1_2k_album_v6',
        'gt_val': os.path.join(INDEX_FOLDER, 'test_album_targets.tsv'),                    # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250424_125609_ukr_songs_v1_2k_album_v6') + '/',                       # set to None or absolute path to the model
    },
    'ukr_songs_v1_2k_album_v6': {
        'model_type': 'album_v6',
        'save_filename': 'ukr_songs_v1_2k_album_v6__best_accuracy',
        'gt_val': os.path.join(INDEX_FOLDER, 'test_album_targets.tsv'),                    # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250424_125609_ukr_songs_v1_2k_album_v6', 'best_val_accuracy') + '/',                       # set to None or absolute path to the model
    },
    'ukr_songs_v1_2k_album_v8': {
        'model_type': 'album_v8',
        'save_filename': 'ukr_songs_v1_2k_album_v8',
        'gt_val': os.path.join(INDEX_FOLDER, 'test_album_targets.tsv'),                    # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250424_143504_ukr_songs_v1_2k_album_v8') + '/',                       # set to None or absolute path to the model
    },
    'ukr_songs_v1_2k_album_v8': {
        'model_type': 'album_v8',
        'save_filename': 'ukr_songs_v1_2k_album_v8__best_accuracy',
        'gt_val': os.path.join(INDEX_FOLDER, 'test_album_targets.tsv'),                    # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250424_143504_ukr_songs_v1_2k_album_v8', 'best_val_accuracy') + '/',                       # set to None or absolute path to the model
    },
    'ukr_songs_v1_2k_album_v10': {
        'model_type': 'album_v10',
        'save_filename': 'ukr_songs_v1_2k_album_v10',
        'gt_val': os.path.join(INDEX_FOLDER, 'test_album_targets.tsv'),                    # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250424_160333_ukr_songs_v1_2k_album_v10') + '/',                       # set to None or absolute path to the model
    },
    'ukr_songs_v1_2k_album_v6': {
        'model_type': 'album_v6',
        'save_filename': 'train_ukr_songs_v1_2k_album_v6_best_accuracy',
        'gt_val': os.path.join(INDEX_FOLDER, 'train_album_targets.tsv'),                    # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250424_125609_ukr_songs_v1_2k_album_v6', 'best_val_accuracy') + '/',                       # set to None or absolute path to the model
    },
    'ukr_songs_v1_2k_album_v5': {
        'model_type': 'album_v5',
        'save_filename': 'train_ukr_songs_v1_2k_album_v5',
        'gt_val': os.path.join(INDEX_FOLDER, 'train_album_targets.tsv'),                    # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250424_115123_ukr_songs_v1_2k_album_v5') + '/',                       # set to None or absolute path to the model
    },
    'ukr_songs_v1_2k_album_v5_continue': {
        'model_type': 'album_v5_continue',
        'save_filename': 'val_ukr_songs_v1_2k_album_v5_continue',
        'gt_val': os.path.join(INDEX_FOLDER, 'val_album_targets.tsv'),                    # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250426_101210_ukr_songs_v1_2k_album_v5_continue', 'best_val_accuracy') + '/',                       # set to None or absolute path to the model
    },
    'ukr_songs_v1_2k_genre_v3': {
        'model_type': 'genre_v3',
        'save_filename': 'val_ukr_songs_v1_2k_genre_v3',
        'gt_val': os.path.join(INDEX_FOLDER, 'val_genre_targets.tsv'),                    # number of the model as in models.py
        'load_model': os.path.join(DATA_FOLDER, 'experiments', '20250426_161037_ukr_songs_v1_2k_genre_v3') + '/',                       # set to None or absolute path to the model
    },
}