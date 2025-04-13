import argparse
import json
import os
import time
import random
import pescador
import numpy as np
import tensorflow as tf
import models
import config_file, shared
import pickle
from tensorflow.python.framework import ops

from models_backend import positional_encoding


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
        # output = tf.layers.dense(attn_output, d_model, name="out_proj")  # [batch, seq_len, d_model]
        
        return attn_output
    
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

        # Process all segments in parallel
        batch_size = tf.shape(x)[0]
        x_reshaped = tf.reshape(x, [-1, segment_frames, n_mels])  # [batch*num_musicnn_segments, 187, 96]

        # Original segment-level model
        # TODO: remove the final projection layer from the model (and return penultimate for example)
        with tf.variable_scope('musicnn'):
            segment_logits = models.model_number(x_reshaped, is_train, config)  # [batch*num_musicnn_segments, penultinate_units]

        print('Segment logits shape:', segment_logits.get_shape())

        # Reshape back to [batch, num_musicnn_segments, penultinate_units]
        segment_logits = tf.reshape(segment_logits, [batch_size, num_musicnn_segments, penultinate_units])  # [batch, num_musicnn_segments, penultinate_units]

        print('Segment logits shape:', segment_logits.get_shape())

        # Apply positional encoding to segment logits
        pos_embedding = positional_encoding(segment_logits.get_shape().as_list())
        segment_logits = tf.add(segment_logits, pos_embedding)  # [batch, num_musicnn_segments, penultinate_units]
        print('Segment logits with positional encoding shape:', segment_logits.get_shape())

        # TODO: batch normalization (?)

        # Calculate attention and add to segment logits
        attention_output = multi_head_attention(segment_logits, num_heads=1, d_model=penultinate_units, d_k=penultinate_units, d_v=penultinate_units) # [batch, num_musicnn_segments, penultinate_units]
        print('attention_output shape:', attention_output.get_shape())
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
        print(normalized_y.get_shape())
    print('Number of parameters of the model: ' + str(shared.count_params(tf.trainable_variables()))+'\n')

    # tensorflow: define cost function
    with tf.name_scope('metrics'):
        # if you use softmax_cross_entropy be sure that the output of your model has linear units!
        cost = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_, logits=y)
        if config['weight_decay'] != None:
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'kernel' in v.name ])
            cost = cost + config['weight_decay']*lossL2
            print('L2 norm, weight decay!')

    # print all trainable variables, for debugging
    model_vars = [v for v in tf.global_variables()]
    for variables in model_vars:
        print(variables)

    return [x, y_, is_train, y, normalized_y, cost]


def data_gen(id, audio_repr_path, gt, pack):
    [config, sampling_strategy, segment_len_frames, num_musicnn_segments] = pack
    
    # Load audio representation
    audio_rep = pickle.load(open(config_file.DATA_FOLDER + audio_repr_path, 'rb'))
    
    # Apply preprocessing
    if config['pre_processing'] == 'logEPS':
        audio_rep = np.log10(audio_rep + np.finfo(float).eps)
    elif config['pre_processing'] == 'logC':
        audio_rep = np.log10(10000 * audio_rep + 1)

    # Ensure we have enough frames
    assert audio_rep.shape[0] >= segment_len_frames, \
           f"Audio is too short ({audio_rep.shape[0]} frames), needs at least {segment_len_frames} frames"
    
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

    print('Data shape:', x.shape)
    
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

    # load audio representation paths
    file_index = config_file.DATA_FOLDER + config['audio_representation_folder'] + 'index.tsv'
    [audio_repr_paths, id2audio_repr_path] = shared.load_id2path(file_index)

    # load training data
    file_ground_truth_train = config_file.DATA_FOLDER + config['gt_train']
    [ids_train, id2gt_train] = shared.load_id2gt(file_ground_truth_train)

    # load validation data
    file_ground_truth_val = config_file.DATA_FOLDER + config['gt_val']
    [ids_val, id2gt_val] = shared.load_id2gt(file_ground_truth_val)

    # set output
    config['classes_vector'] = list(range(config['num_classes_dataset']))

    print('# Train:', len(ids_train))
    print('# Val:', len(ids_val))
    print('# Classes:', config['classes_vector'])

    # save experimental settings
    experiment_id = str(shared.get_epoch_time()) + args.configuration
    model_folder = config_file.DATA_FOLDER + 'experiments/' + str(experiment_id) + '/'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    json.dump(config, open(model_folder + 'config.json', 'w'))
    print('\nConfig file saved: ' + str(config))

    # define the musicnn segment length and number of such segments in the input audio segment of length 'segment_len'
    musicnn_segment_len = 3 # in sec; TODO: make configurable via config (n_frames, hop_size, etc.)
    num_musicnn_segments = int(config['segment_len'] / musicnn_segment_len)
    segment_len_frames = num_musicnn_segments * config['xInput']
    config['num_musicnn_segments'] = num_musicnn_segments

    # tensorflow: define model and cost
    [x, y_, is_train, y, normalized_y, cost] = tf_define_model_and_cost(config)

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

    print('\nEXPERIMENT: ', str(experiment_id))
    print('-----------------------------------')

    # pescador train: define streamer
    train_pack = [config, config['sampling_strategy'], segment_len_frames, num_musicnn_segments]
    train_streams = [pescador.Streamer(data_gen, id, id2audio_repr_path[id], id2gt_train[id], train_pack) for id in ids_train]
    train_mux_stream = pescador.StochasticMux(train_streams, n_active=config['batch_size']*2, rate=None, mode='exhaustive')
    train_batch_streamer = pescador.Streamer(pescador.buffer_stream, train_mux_stream, buffer_size=config['batch_size'], partial=True)
    train_batch_streamer = pescador.ZMQStreamer(train_batch_streamer)

    # pescador val: define streamer
    val_pack = [config, config['sampling_strategy'], segment_len_frames, num_musicnn_segments]
    val_streams = [pescador.Streamer(data_gen, id, id2audio_repr_path[id], id2gt_val[id], val_pack) for id in ids_val]
    val_mux_stream = pescador.ChainMux(val_streams, mode='exhaustive')
    val_batch_streamer = pescador.Streamer(pescador.buffer_stream, val_mux_stream, buffer_size=config['val_batch_size'], partial=True)
    val_batch_streamer = pescador.ZMQStreamer(val_batch_streamer)

    # tensorflow: create a session to run the tensorflow graph
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if config['load_model'] != None: # restore model weights from previously saved model
        saver.restore(sess, config['load_model']) # end with /!
        print('Pre-trained model loaded!')

    # writing headers of the train_log.tsv
    fy = open(model_folder + 'train_log.tsv', 'a')
    fy.write('Epoch\ttrain_cost\tval_cost\tepoch_time\tlearing_rate\n')
    fy.close()

    # training
    k_patience = 0
    cost_best_model = np.Inf
    tmp_learning_rate = config['learning_rate']
    print('Training started..')
    for i in range(config['epochs']):
        print('Epoch %d' % (i))
        # training: do not train first epoch, to see random weights behaviour
        start_time = time.time()
        array_train_cost = []
        if i != 0:
            for train_batch in train_batch_streamer:
                tf_start = time.time()
                _, train_cost = sess.run([train_step, cost],
                                         feed_dict={x: train_batch['X'], y_: train_batch['Y'], lr: tmp_learning_rate, is_train: True})
                array_train_cost.append(train_cost)

        # validation
        array_val_cost = []
        for val_batch in val_batch_streamer:
            val_cost = sess.run([cost],
                                feed_dict={x: val_batch['X'], y_: val_batch['Y'], is_train: False})
            array_val_cost.append(val_cost)

        # Keep track of average loss of the epoch
        train_cost = np.mean(array_train_cost)
        val_cost = np.mean(array_val_cost)
        epoch_time = time.time() - start_time
        fy = open(model_folder + 'train_log.tsv', 'a')
        fy.write('%d\t%g\t%g\t%gs\t%g\n' % (i+1, train_cost, val_cost, epoch_time, tmp_learning_rate))
        fy.close()

        # Decrease the learning rate after not improving in the validation set
        if config['patience'] and k_patience >= config['patience']:
            print('Changing learning rate!')
            tmp_learning_rate = tmp_learning_rate / 2
            print(tmp_learning_rate)
            k_patience = 0

        # Early stopping: keep the best model in validation set
        if val_cost >= cost_best_model:
            k_patience += 1
            print('Epoch %d, train cost %g, val cost %g,'
                  'epoch-time %gs, lr %g, time-stamp %s' %
                  (i+1, train_cost, val_cost, epoch_time, tmp_learning_rate,
                   str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))))

        else:
            # save model weights to disk
            save_path = saver.save(sess, model_folder)
            print('Epoch %d, train cost %g, val cost %g, '
                  'epoch-time %gs, lr %g, time-stamp %s - [BEST MODEL]'
                  ' saved in: %s' %
                  (i+1, train_cost, val_cost, epoch_time,tmp_learning_rate,
                   str(time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())), save_path))
            cost_best_model = val_cost

    print('\nEVALUATE EXPERIMENT -> '+ str(experiment_id))
