import os 

DATA_CSV = os.path.join('data', 'final_dataset_v1.csv')
DATA_FOLDER =  'output/'
CONFIGS_FOLDER = 'configs'
AUDIO_FOLDER = 'audio'

DATASET = 'ukr_songs_v1_2k'

config_preprocess = {
    'mtt_spec': {
        'identifier': 'mtt',                      # name for easy identification
        'audio_folder': '/home/jpons/audio/mtt/', # end it with / -> this is an absolute path!
        'n_machines': 1,                          # parallelizing this process through 'n_machines'
        'machine_i': 0,                           # id number of the machine which is running this script (from 0 to n_machines-1)
        'num_processing_units': 20,               # number of parallel processes in every machine
        'type': 'time-freq',                      # kind of audio representation: 'time-freq' (only recommended option)
        'spectrogram_type': 'mel',                # 'mel' (only option) - parameters below should change according to this type
        'resample_sr': 16000,                     # sampling rate (original or the one to be resampled)
        'hop': 256,                               # hop size of the STFT
        'n_fft': 512,                             # frame size (number of freq bins of the STFT)
        'n_mels': 96,                             # number of mel bands
        'index_file': 'index/mtt/index_mtt.tsv',  # list of audio representations to be computed
    },
    'msd_spec': {
        'identifier': 'msd',                      # name for easy identification
        'audio_folder': '/home/jpons/audio/MSD/millionsong-audio/mp3/', # end it with / -> this is an absolute path!
        'n_machines': 1,                          # parallelizing this process through 'n_machines'
        'machine_i': 0,                           # id number of the machine which is running this script (from 0 to n_machines-1)
        'num_processing_units': 20,               # number of parallel processes in every machine
        'type': 'time-freq',                      # kind of audio representation: 'time-freq' (only recommended option)
        'spectrogram_type': 'mel',                # 'mel' (only option) - parameters below should change according to this type
        'resample_sr': 16000,                     # sampling rate (original or the one to be resampled)
        'hop': 256,                               # hop size of the STFT
        'n_fft': 512,                             # frame size (number of freq bins of the STFT)
        'n_mels': 96,                             # number of mel bands
        'index_file': 'index/msd/index_msd.tsv',  # list of audio representations to be computed
    },
    'lastfm_sample_spec': {
        'identifier': DATASET,                      # name for easy identification
        'audio_folder': 'data\\audio\\', # end it with / -> this is an absolute path!
        'n_machines': 1,                          # parallelizing this process through 'n_machines'
        'machine_i': 0,                           # id number of the machine which is running this script (from 0 to n_machines-1)
        'num_processing_units': 1,               # number of parallel processes in every machine
        'type': 'time-freq',                      # kind of audio representation: 'time-freq' (only recommended option)
        'spectrogram_type': 'mel',                # 'mel' (only option) - parameters below should change according to this type
        'resample_sr': 16000,                     # sampling rate (original or the one to be resampled)
        'hop': 256,                               # hop size of the STFT
        'n_fft': 512,                             # frame size (number of freq bins of the STFT)
        'n_mels': 96,                             # number of mel bands
        'index_file': f"index/{DATASET}/index_{DATASET}.tsv",  # list of audio representations to be computed
    },
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
        'index_file': os.path.join("index", DATASET, "index.tsv"),  # list of audio representations to be computed
    }
}


config_train = {
    'spec': {
        'name_run': '',
        # which data?
        'audio_representation_folder': 'audio_representation/'+DATASET+'__time-freq/',
        'gt_train': 'index/'+DATASET+'/train_gt_'+DATASET+'.tsv',
        'gt_val': 'index/'+DATASET+'/val_gt_'+DATASET+'.tsv',

        # input setup?
        'n_frames': 187,                          # length of the input (integer)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None
        'pad_short': 'repeat-pad',                # 'zero-pad' or 'repeat-pad'
        'train_sampling': 'random',               # 'overlap_sampling' or 'random'. How to sample patches from the audio?
        'param_train_sampling': 1,                # if mode_sampling='overlap_sampling': param_sampling=hop_size
                                                  # if mode_sampling='random': param_sampling=number of samples
        # learning parameters?
        'model_number': 11,                       # number of the model as in models.py
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': 600,                            # maximum number of epochs before stopping training
        'batch_size': 32,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 0.001,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)

        # experiment settings?
        'num_classes_dataset': 50,
        'val_batch_size': 32
    },
    
    'genre_test': {
        'name_run': '',
        # which data?
        'audio_representation_folder': 'audio_representation/'+DATASET+'__time-freq/',
        'gt_train': 'index/'+DATASET+'/train_gt_'+DATASET+'.tsv',
        'gt_val': 'index/'+DATASET+'/val_gt_'+DATASET+'.tsv',

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': 1,                            # maximum number of epochs before stopping training
        'batch_size': 1,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 0.001,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 50,                # the number of unique genre tags
        'val_batch_size': 1
    },

    'genre_test_v2': {
        'name_run': '',
        # which data?
        'audio_representation_folder': 'audio_representation/'+DATASET+'__time-freq/',
        'gt_train': 'index/'+ DATASET +'/train_album_targets.tsv',
        'gt_val': 'index/'+ DATASET + '/val_album_targets.tsv',

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': 1,                            # maximum number of epochs before stopping training
        'batch_size': 1,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 0.001,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 122,                # the number of unique genre tags (435 albums/122 genres)
        'val_batch_size': 1
    },

    'ukr_songs_v1_2k_genre': {
        'name_run': '',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join('index', DATASET, 'train_genre_targets.tsv'),
        'gt_val': os.path.join('index', DATASET, 'val_genre_targets.tsv'),

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': 100,                            # maximum number of epochs before stopping training
        'batch_size': 4,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 0.001,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 122,                # the number of unique genre tags (435 albums/122 genres)
        'val_batch_size': 4
    },

    'ukr_songs_v1_2k_album': {
        'name_run': '',
        # which data?
        'audio_representation_folder': os.path.join('audio_representation', f"{DATASET}__time-freq"),
        'gt_train': os.path.join('index', DATASET, 'train_album_targets.tsv'),
        'gt_val': os.path.join('index', DATASET, 'val_album_targets.tsv'),

        # input setup?
        'segment_len': 60,                         # length of the audio segment (in seconds)
        'sampling_strategy': 'random',            # 'first' or 'random'. How to sample an audio segment?
        'n_frames': 187,                          # length of the input segment to the Musicnn model (in frames) 
                                                        # (187 frames = 3 sec in the current setup)
        'pre_processing': 'logC',                 # 'logEPS', 'logC' or None     #TODO

        # learning parameters?
        'model_number': 111,                       # number of the model as in models.py
        'load_model': None,                       # set to None or absolute path to the model
        'epochs': 100,                            # maximum number of epochs before stopping training
        'batch_size': 8,                         # batch size during training
        'weight_decay': 1e-5,                     # None or value for the regularization parameter
        'learning_rate': 0.001,                   # learning rate
        'optimizer': 'Adam',                      # 'SGD_clip', 'SGD', 'Adam'
        'patience': 75,                           # divide by two the learning rate after the number of 'patience' epochs (integer)

        # experiment settings?
        'feature_vector_dim': 128,                # size of the penultimate layer of the model; will be used as a feature vector for the song
        'num_classes_dataset': 435,                # the number of unique genre tags (435 albums/122 genres)
        'val_batch_size': 8
    }
}

