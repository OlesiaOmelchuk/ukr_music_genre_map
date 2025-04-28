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
import wandb

# # disable eager mode for tf.v1 compatibility with tf.v2
# tf.compat.v1.disable_eager_execution()

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from models_backend import positional_encoding


# Configure logger to save logs to a file with date and time in a 'log' directory
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{time.strftime('train_%Y-%m-%d_%H-%M-%S')}.log")
logger.add(log_file, level="DEBUG")


def multi_head_attention(inputs, num_heads, d_model, d_k, d_v, scope="multihead_attention"):
    """
    Args:
        inputs: Tensor of shape [batch_size, seq_len, d_model]
        num_heads: Number of attention heads
        d_model: Original embedding dimension (must be divisible by num_heads)
        d_k: Key/Query dimension per head
        d_v: Value dimension per head
    Returns:
        output: Tensor of shape [batch_size, seq_len, d_model]
    """
    with tf.variable_scope(scope):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Linear projections for Q, K, V (with learnable weights)
        Q = tf.layers.dense(inputs, d_model, name="q_proj")  # [batch, seq_len, d_model]
        K = tf.layers.dense(inputs, d_model, name="k_proj")  # [batch, seq_len, d_model]
        V = tf.layers.dense(inputs, d_model, name="v_proj")  # [batch, seq_len, d_model]

        # Split into multiple heads (reshape + transpose)
        Q = tf.reshape(Q, [batch_size, seq_len, num_heads, d_k])  # [batch, seq_len, num_heads, d_k]
        Q = tf.transpose(Q, [0, 2, 1, 3])  # [batch, num_heads, seq_len, d_k]
        
        K = tf.reshape(K, [batch_size, seq_len, num_heads, d_k])  # [batch, seq_len, num_heads, d_k]
        K = tf.transpose(K, [0, 2, 1, 3])  # [batch, num_heads, seq_len, d_k]
        
        V = tf.reshape(V, [batch_size, seq_len, num_heads, d_v])  # [batch, seq_len, num_heads, d_v]
        V = tf.transpose(V, [0, 2, 1, 3])  # [batch, num_heads, seq_len, d_v]

        # Scaled dot-product attention
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(d_k, tf.float32))  # [batch, num_heads, seq_len, seq_len]
        attn_weights = tf.nn.softmax(scores, axis=-1)  # [batch, num_heads, seq_len, seq_len]
        attn_output = tf.matmul(attn_weights, V)  # [batch, num_heads, seq_len, d_v]

        # Merge heads back
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])  # [batch, seq_len, num_heads, d_v]
        attn_output = tf.reshape(attn_output, [batch_size, seq_len, num_heads * d_v])  # [batch, seq_len, d_model]

        # Final linear projection
        output = tf.layers.dense(attn_output, d_model, name="out_proj")  # [batch, seq_len, d_model]
        
        return output
    
def tf_define_model_and_cost(config):
    # tensorflow: define the model
    num_musicnn_segments = config['num_musicnn_segments']  # 120 seconds / 3-second segments
    penultinate_units = 200  # e.g., 200 units in the last layer
    segment_frames = config['xInput']  # e.g., 187 frames (3 sec)
    n_mels = config['yInput']  # e.g., 96 mel bins
    feature_vec_dim = config['feature_vector_dim'] # size of the penultimate layer of the model; will be used as a feature vector for the song

    with tf.name_scope('model'):
        # Input placeholders [batch, num_musicnn_segments, 187, 96]
        x = tf.compat.v1.placeholder(tf.float32, [None, num_musicnn_segments, segment_frames, n_mels])
        y_ = tf.compat.v1.placeholder(tf.float32, [None, config['num_classes_dataset']])
        is_train = tf.compat.v1.placeholder(tf.bool)
        calculate_accuracy = tf.compat.v1.placeholder(tf.bool)

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
        segment_logits = tf.compat.v1.layers.batch_normalization(segment_logits, training=is_train)

        # Calculate attention and add to segment logits
        attention_output = multi_head_attention(segment_logits, num_heads=1, d_model=penultinate_units, d_k=penultinate_units, d_v=penultinate_units) # [batch, num_musicnn_segments, penultinate_units]
        logger.debug(f'attention_output shape: {attention_output.get_shape()}')
        segment_logits = tf.add(segment_logits, attention_output)  # [batch, num_musicnn_segments, penultinate_units]

        # Aggregate segment-level features into song-level feature vector
        with tf.variable_scope('aggregation'):
            if 'aggregation' in config and config['aggregation'] == 'attention':
                attention_scores = tf.compat.v1.layers.dense(
                    segment_logits, 
                    units=1,  # Scalar score per segment
                    activation=None,
                    name='attention_scores'
                )  # [batch, num_segments, 1]
                attention_weights = tf.nn.softmax(attention_scores, axis=1)  # Normalize across segments
                penultinate_units_agg = tf.reduce_sum(segment_logits * attention_weights, axis=1)  # Weighted sum
            elif 'aggregation' in config and config['aggregation'] == 'meanpooling':
                penultinate_units_agg = tf.reduce_mean(segment_logits, axis=1)
            elif 'aggregation' in config and config['aggregation'] == 'maxpooling':
                penultinate_units_agg = tf.reduce_max(segment_logits, axis=1)
            elif 'aggregation' in config and config['aggregation'] == 'meanmaxpooling':
                mean_pool = tf.reduce_mean(segment_logits, axis=1)
                max_pool = tf.reduce_max(segment_logits, axis=1)
                penultinate_units_agg = tf.concat([mean_pool, max_pool], axis=1)
            else:
                penultinate_units_agg = tf.compat.v1.layers.flatten(segment_logits)  # [batch, num_musicnn_segments*penultinate_units]

            penultinate_units_agg = tf.compat.v1.layers.batch_normalization(penultinate_units_agg, training=is_train)
            penultinate_units_agg_dropout = tf.compat.v1.layers.dropout(penultinate_units_agg, rate=0.5, training=is_train)

            feature_vectors_dense = tf.compat.v1.layers.dense(
                inputs=penultinate_units_agg_dropout,
                units=feature_vec_dim,
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling()
                # kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
            )  # [batch, feature_vec_dim]

        # Finally, apply a dense layer to project from penultinate_units to num_classes
        feature_vectors_dense = tf.compat.v1.layers.batch_normalization(feature_vectors_dense, training=is_train)

        if 'feat_vec_dropout' in config and config['feat_vec_dropout'] == True:
            feature_vectors_dense = tf.compat.v1.layers.dropout(feature_vectors_dense, rate=0.5, training=is_train)

        # Add L2 normalization here
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors_dense, axis=1)
        # feature_vectors_normalized = tf.keras.layers.LayerNormalization()(feature_vectors_dense)
                                                     
        y = tf.compat.v1.layers.dense(
            inputs=feature_vectors_normalized,
            units=config['num_classes_dataset'],
            activation=None,
            kernel_initializer=tf.keras.initializers.VarianceScaling()
            # kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
        ) # [batch, num_classes_dataset]

        if config['loss_function'] == 'sigmoid_cross_entropy':
            normalized_y = tf.nn.sigmoid(y)
        elif config['loss_function'] == 'softmax_cross_entropy':
            normalized_y = tf.nn.softmax(y)
        else:
            raise ValueError(f"Unsupported loss function: {config['loss_function']}. Use 'sigmoid_cross_entropy' or 'softmax_cross_entropy'.")
    logger.info(f'Number of parameters of the model: {str(shared.count_params(tf.trainable_variables()))}')

    # tensorflow: define cost function
    with tf.name_scope('metrics'):
        # if you use softmax_cross_entropy be sure that the output of your model has linear units!
        if config['loss_function'] == 'softmax_cross_entropy':
            cost = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_, y)
        elif config['loss_function'] == 'sigmoid_cross_entropy':
            cost = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_, logits=y)
        else:
            raise ValueError(f"Unsupported loss function: {config['loss_function']}. Use 'softmax_cross_entropy' or 'sigmoid_cross_entropy'.")
        
        logger.info(f'Loss function: {config["loss_function"]}')

        if 'weight_decay' in config and config['weight_decay'] != None:
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'kernel' in v.name ])
            cost = cost + config['weight_decay']*lossL2
            logger.info(f'L2 norm, weight decay!')

    # logger.info fall trainable variables, for debugging
    model_vars = [v for v in tf.global_variables()]
    for variables in model_vars:
        logger.debug(variables)

    def compute_accuracy(y, y_):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    accuracy = tf.cond(
        calculate_accuracy,
        true_fn=lambda: compute_accuracy(normalized_y, y_),  # Your accuracy computation
        false_fn=lambda: tf.constant(-1.0)  # Dummy value (won't be used)
    )

    return [x, y_, is_train, y, normalized_y, cost, calculate_accuracy, accuracy, feature_vectors_normalized]


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

    # load training data
    file_ground_truth_train = os.path.join(config['gt_train'])
    [ids_train, id2gt_train] = shared.load_id2gt(file_ground_truth_train)

    # load validation data
    file_ground_truth_val = os.path.join(config['gt_val'])
    [ids_val, id2gt_val] = shared.load_id2gt(file_ground_truth_val)

    # load train subset (for metrics calculation)
    if 'gt_train_subset' in config and config['gt_train_subset'] != None:
        file_ground_truth_train_subset = os.path.join(config['gt_train_subset'])
        [ids_train_subset, id2gt_train_subset] = shared.load_id2gt(file_ground_truth_train_subset)
    else:
        ids_train_subset = None
        id2gt_train_subset = None

    # set output
    config['classes_vector'] = list(range(config['num_classes_dataset']))

    logger.info(f'# Train: {len(ids_train)}')
    logger.info(f'# Val: {len(ids_val)}')
    logger.info(f"# Classes: {config['classes_vector']}")

    # save experimental settings
    experiment_id = time.strftime('%Y%m%d_%H%M%S') + '_' + args.configuration
    model_folder = os.path.join(config_file.DATA_FOLDER, 'experiments', str(experiment_id))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    json.dump(config, open(os.path.join(model_folder, 'config.json'), 'w'))

    legacy_model_folder = model_folder
    if not legacy_model_folder.endswith('/'):
        legacy_model_folder += '/'

    legacy_model_folder_best_accuracy = os.path.join(model_folder, 'best_val_accuracy')
    if not os.path.exists(legacy_model_folder_best_accuracy):
        os.makedirs(legacy_model_folder_best_accuracy)
    if not legacy_model_folder_best_accuracy.endswith('/'):
        legacy_model_folder_best_accuracy += '/'

    logger.info(f'Config file saved: {str(config)}')

    # define the musicnn segment length and number of such segments in the input audio segment of length 'segment_len'
    musicnn_segment_len = 3 # in sec; TODO: make configurable via config (n_frames, hop_size, etc.)
    num_musicnn_segments = int(config['segment_len'] / musicnn_segment_len)
    segment_len_frames = num_musicnn_segments * config['xInput']
    config['num_musicnn_segments'] = num_musicnn_segments

    # tensorflow: define model and cost
    [x, y_, is_train, y, normalized_y, cost, calculate_accuracy, accuracy, _] = tf_define_model_and_cost(config)

    # tensorflow: define optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # needed for batchnorm
    with tf.control_dependencies(update_ops):
        lr = tf.placeholder(tf.float32)
        if config['optimizer'] == 'SGD_clip':
            optimizer = tf.train.GradientDescentOptimizer(lr)
            gradients, variables = zip(*optimizer.compute_gradients(cost))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            train_step = optimizer.apply_gradients(zip(gradients, variables))
        elif config['optimizer'] == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(lr)
            train_step = optimizer.minimize(cost)
        elif config['optimizer'] == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            train_step = optimizer.minimize(cost)

    sess = tf.InteractiveSession()
    tf.keras.backend.set_session(sess)

    logger.info(f'EXPERIMENT: {str(experiment_id)}')

    # pescador train: define streamer
    train_pack = [config, config['sampling_strategy'], segment_len_frames, num_musicnn_segments]
    train_streams = [pescador.Streamer(data_gen, id, id2audio_repr_path[id], id2gt_train[id], train_pack) for id in ids_train]
    train_mux_stream = pescador.StochasticMux(train_streams, n_active=config['batch_size']*2, rate=None, mode='exhaustive')
    train_batch_streamer = pescador.Streamer(pescador.buffer_stream, train_mux_stream, buffer_size=config['batch_size'], partial=True)
    train_batch_streamer = pescador.ZMQStreamer(train_batch_streamer)
    logger.info('Successfully created train streamer!')

    # pescador val: define streamer
    val_pack = [config, config['sampling_strategy'], segment_len_frames, num_musicnn_segments]
    val_streams = [pescador.Streamer(data_gen, id, id2audio_repr_path[id], id2gt_val[id], val_pack) for id in ids_val]
    val_mux_stream = pescador.ChainMux(val_streams, mode='exhaustive')
    val_batch_streamer = pescador.Streamer(pescador.buffer_stream, val_mux_stream, buffer_size=config['val_batch_size'], partial=True)
    val_batch_streamer = pescador.ZMQStreamer(val_batch_streamer)
    logger.info('Successfully created val streamer!')

    # pescador train subset: define streamer
    if ids_train_subset != None:
        train_subset_pack = [config, config['sampling_strategy'], segment_len_frames, num_musicnn_segments]
        train_subset_streams = [pescador.Streamer(data_gen, id, id2audio_repr_path[id], id2gt_train_subset[id], train_subset_pack) for id in ids_train_subset]
        train_subset_mux_stream = pescador.ChainMux(train_subset_streams, mode='exhaustive')
        train_subset_batch_streamer = pescador.Streamer(pescador.buffer_stream, train_subset_mux_stream, buffer_size=config['val_batch_size'], partial=True)
        train_subset_batch_streamer = pescador.ZMQStreamer(train_subset_batch_streamer)
        logger.info('Successfully created train subset streamer!')
    else:
        train_subset_batch_streamer = None
        logger.info('No train subset streamer created!')

    # tensorflow: create a session to run the tensorflow graph
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if config['load_model'] != None: # restore model weights from previously saved model
        saver.restore(sess, config['load_model']) # end with /!
        logger.info(f"Pre-trained model loaded from {config['load_model']}")

    # writing headers of the train_log.tsv
    fy = open(os.path.join(model_folder, 'train_log.tsv'), 'a')
    fy.write('Epoch\ttrain_cost\tval_cost\tepoch_time\tlearing_rate\n')
    fy.close()

    # training
    k_patience = 0
    cost_best_model = np.Inf
    accuracy_best_model = 0
    tmp_learning_rate = config['learning_rate']

    # Initialize wandb
    wandb.init(project="ukr_genre_map_thesis", 
            config=config,
            name=f"exp_{experiment_id}",
            notes="-")

    logger.info(f'Training started..')
    for i in range(config['epochs']):
        logger.info(f'Epoch {i+1}')
        # training: do not train first epoch, to see random weights behaviour
        start_time = time.time()
        array_train_cost = []
        if i != 0:
            for train_batch in train_batch_streamer:
                tf_start = time.time()
                _, train_cost = sess.run([train_step, cost],
                                         feed_dict={x: train_batch['X'], y_: train_batch['Y'], lr: tmp_learning_rate, is_train: True, calculate_accuracy: False})
                array_train_cost.append(train_cost)

        # validation
        array_val_cost = []
        array_val_accuracy = []
        for val_batch in val_batch_streamer:
            val_cost, val_accuracy = sess.run([cost, accuracy],
                                feed_dict={x: val_batch['X'], y_: val_batch['Y'], is_train: False, calculate_accuracy: True if "genre" not in args.configuration else False})
            array_val_cost.append(val_cost)
            array_val_accuracy.append(val_accuracy)
        
        # metrics on test subset (if provided)
        if train_subset_batch_streamer != None:
            array_train_subset_cost = []
            array_train_subset_accuracy = []
            for train_subset_batch in train_subset_batch_streamer:
                train_subset_cost, train_subset_accuracy = sess.run([cost, accuracy],
                                        feed_dict={x: train_subset_batch['X'], y_: train_subset_batch['Y'], is_train: False, calculate_accuracy: True if "genre" not in args.configuration else False})
                array_train_subset_cost.append(train_subset_cost)
                array_train_subset_accuracy.append(train_subset_accuracy)

            # Log metrics to wandb
            # wandb.log({
            #     "train/subset_cost": np.mean(array_train_subset_cost),
            #     "train/subset_accuracy": np.mean(array_train_subset_accuracy)
            # })

        # Keep track of average loss of the epoch
        train_cost = np.mean(array_train_cost)
        val_accuracy = np.mean(array_val_accuracy)
        val_cost = np.mean(array_val_cost)
        train_subset_cost = np.mean(array_train_subset_cost) if train_subset_batch_streamer else None
        train_subset_accuracy = np.mean(array_train_subset_accuracy) if train_subset_batch_streamer else None
        epoch_time = time.time() - start_time
        fy = open(os.path.join(model_folder, 'train_log.tsv'), 'a')
        fy.write('%d\t%g\t%g\t%gs\t%g\n' % (i+1, train_cost, val_cost, epoch_time, tmp_learning_rate))
        fy.close()

        # Log to wandb
        wandb.log({
            "epoch": i+1,
            "train/cost": train_cost,
            "val/cost": val_cost,
            "val/accuracy": val_accuracy,
            "train/subset_cost": train_subset_cost,
            "train/subset_accuracy": train_subset_accuracy,
            "time/epoch": epoch_time,
            "hyperparams/learning_rate": tmp_learning_rate,
            "patience": k_patience
        })

        # Decrease the learning rate after not improving in the validation set
        # if config['patience'] and k_patience >= config['patience']:
        #     logger.info(f'Changing learning rate from {tmp_learning_rate} to {tmp_learning_rate / 2}')
        #     tmp_learning_rate = tmp_learning_rate / 2
        #     k_patience = 0

        #     # Log learning rate change
        #     wandb.log({
        #         "hyperparams/learning_rate_change": tmp_learning_rate,
        #         "patience_reset": 0
        #     })

        # Early stopping: keep the best model in validation set
        if val_cost >= cost_best_model:
            k_patience += 1
            logger.info(f'Epoch %d, train cost %g, val cost %g,'
                  'epoch-time %gs, lr %g, time-stamp %s' %
                  (i+1, train_cost, val_cost, epoch_time, tmp_learning_rate,
                   str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))))

        else:
            # save model weights to disk
            save_path = saver.save(sess, legacy_model_folder)
            logger.info(f'Epoch %d, train cost %g, val cost %g, '
                  'epoch-time %gs, lr %g, time-stamp %s - [BEST MODEL]'
                  ' saved in: %s' %
                  (i+1, train_cost, val_cost, epoch_time,tmp_learning_rate,
                   str(time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())), save_path))
            cost_best_model = val_cost

            logger.info(f'Best model (validation loss) saved in: {save_path}')

            # Log best model information
            # wandb.log({
            #     "best_model/val_cost": val_cost,
            #     "best_model/epoch": i+1,
            #     "best_model/save_path": save_path
            # })

            # Optionally save model to wandb
            # wandb.save(save_path + '*')  # Saves all model files

        # Save the best model based on validation accuracy
        if val_accuracy >= accuracy_best_model:
            accuracy_best_model = val_accuracy
            save_path_best_accuracy = saver.save(sess, legacy_model_folder_best_accuracy)
            logger.info(f'Best model (validation accuracy) saved in: {save_path_best_accuracy}')

            # wandb.log({
            #     "best_accuracy_model/val_accuracy": val_accuracy,
            #     "best_accuracy_model/epoch": i+1,
            #     "best_accuracy_model/save_path": save_path_best_accuracy
            # })

        # Save model every 50 epochs
        if i % 50 == 0:
            epoch_model_folder = os.path.join(model_folder, f'epoch_{i}')
            os.makedirs(epoch_model_folder, exist_ok=True)
            epoch_save_path = saver.save(sess, epoch_model_folder + '/')
            logger.info(f'Model saved at epoch {i} in: {epoch_save_path}')

            # Log periodic model save
            # wandb.log({
            #     "checkpoint/epoch": i,
            #     "checkpoint/path": epoch_save_path
            # })
            # wandb.save(epoch_model_folder + '/*')

    logger.info(f'EVALUATE EXPERIMENT -> {str(experiment_id)}')

    # Mark run as completed
    wandb.finish()