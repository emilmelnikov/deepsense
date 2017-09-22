#!/usr/bin/env python


import logging
import os
import tensorflow as tf


TRAIN_SIZE = 119080
EVAL_SIZE = 1193


tf.flags.DEFINE_string('datadir', 'data', 'data directory')
tf.flags.DEFINE_string('hparams', '', 'hyperparameters')
tf.flags.DEFINE_string('modeldir', 'model', 'model directory')
tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'running mode')
tf.flags.DEFINE_string('log', 'INFO', 'log level')
tf.flags.DEFINE_integer('cpplog', '2', 'log level for C++ backend')
FLAGS = tf.flags.FLAGS


tf.logging.set_verbosity(logging.getLevelName(FLAGS.log))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(FLAGS.cpplog)


def get_default_hparams():
    return tf.contrib.training.HParams(
        learning_rate=1e-3,
        batch_size=64,
        epochs=1_000,
        kernel_sizes=dict(sensor=[3, 3, 3], fused=[8, 6, 4]),
        filters=64,
        paddings=dict(sensor='valid', fused='same'),
        conv_keep_prob=0.8,
    )


def parse_fn(serialized):
    inputs = tf.parse_example(serialized, features=dict(
        acc=tf.FixedLenFeature([20, 10, 6], tf.float32),
        gyro=tf.FixedLenFeature([20, 10, 6], tf.float32),
        label=tf.FixedLenFeature([], tf.int64),
    ))
    label = inputs['label']
    del inputs['label']
    return inputs, dict(classes=label)


def make_input_fn(filenames, batch_size):
    """Create input_fn for training and/or evaluation."""
    return lambda: (tf.contrib.data.TFRecordDataset(filenames).
                    shuffle(buffer_size=10_000).
                    batch(batch_size).
                    map(parse_fn).
                    repeat().
                    make_one_shot_iterator().
                    get_next())


def conv_layer(inputs, training, kernel_size, igroup, filters, padding,
               keep_prob):
    """Single convolutional layer.

    * Convolution (does not touch time steps)
    * Batch normalization
    * Activation (ReLU)
    * Dropout (identical masks for all time steps and features)
    """
    with tf.name_scope(str(igroup)):
        layer = tf.layers.conv2d(inputs, filters, [1, kernel_size],
                                 padding=padding, use_bias=False)
        layer = tf.layers.batch_normalization(layer, training=training)
        layer = tf.nn.relu(layer)
        shape = tf.shape(layer)
        layer = tf.layers.dropout(layer, rate=1-keep_prob,
                                  noise_shape=[shape[0], 1, 1, shape[3]],
                                  training=training)
    return layer


def conv_layer_n(inputs, training, subnet_type, subnet_name, params):
    """Group of N convolutional layers."""
    subnet = inputs
    with tf.name_scope(subnet_name or subnet_type):
        for i, kernel_size in enumerate(params.kernel_sizes[subnet_type], 1):
            subnet = conv_layer(
                subnet, training=training, kernel_size=kernel_size, igroup=i,
                filters=params.filters, padding=params.paddings[subnet_type],
                keep_prob=params.conv_keep_prob)
    return subnet


def seq_layer(inputs, training, layers=2, units=128, keep_prob=0.5):
    """Sequential layer of N GRU cells with dropout after each output.

    Returns
    -------
        RNN output.
    """
    cells = []
    for _i in range(layers):
        cell = tf.nn.rnn_cell.GRUCell(units)
        if training:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=keep_prob)
        cells.append(cell)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    rnn, _rnn_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    return rnn


def get_network(sensors, training, params):
    """Create network graph.

    Arguments
    ---------
    sensors : dict
        Mapping from string sensor names to Tensor sensor readings;
        each Tensor has shape
        (batch_size, time_intervals, dft_bins, n_features).
    training : bool
        Whether to create a model for training.
    params : HParams
        Hyperparameters.

    Returns
    -------
        Last network layer before fully-connected layer.
    """

    # Convolve each sensor input independently.
    sensor_subnets = []
    for sensor_name, sensor in sensors.items():
        subnet = conv_layer_n(sensor, training, 'sensor', sensor_name, params)
        sensor_subnets.append(subnet)

    # Stack convolved outputs along the n_features axis.
    # We treat features from individual outputs as if they come
    # from a single sensor.
    fused = tf.concat(sensor_subnets, axis=3)

    # Convolve outputs from individual sensor subnets.
    fused = conv_layer_n(fused, training, 'fused', '', params)

    # Reshape fused output into (batch_size, time_intervals, n_features).
    # RNN requires 1D feature vector for each time step.
    fused = tf.reshape(fused, [-1, 20, 256])

    # Fold time intervals with recurrent subnet.
    seq = seq_layer(fused, training)
    return seq


def model_fn(features, labels, mode, params, config):
    """Model for Estimator.

    Currently, `features` is dict from strings to Tensors
    (i.e. FeatureColumns are not supported yet).
    """

    # Get the last inner layer.
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    outputs = get_network(features, training, params)

    # Create the output layer (Head) without the softmax part.
    outputs = tf.reduce_mean(outputs, axis=1)
    logits = tf.layers.dense(outputs, 6)

    classes = labels['classes']

    # Define the loss (not needed when predicting).
    loss = None
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        loss = tf.losses.sparse_softmax_cross_entropy(classes, logits)

    # Create train op based on loss (needed only when training).
    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = (tf.train.AdamOptimizer(params.learning_rate).
                    minimize(loss, global_step=tf.train.get_global_step()))

    # Predictions are also needed in train and eval modes for computing metrics.
    predictions = dict(
        classes=tf.argmax(logits, axis=1),
    )

    # Track metrics (not needed when predicting).
    eval_metric_ops = None
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        eval_metric_ops = dict(
            accuracy=tf.metrics.accuracy(classes, predictions['classes']),
        )

    return tf.estimator.EstimatorSpec(
        mode, predictions=predictions, loss=loss, train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def experiment_fn(config, params):
    """Create Experiment with make_input_fn and model_fn."""
    train_filenames = [os.path.join(FLAGS.datadir, 'train.tfrecords')]
    eval_filenames = [os.path.join(FLAGS.datadir, 'eval.tfrecords')]
    return tf.contrib.learn.Experiment(
        tf.estimator.Estimator(model_fn, config=config, params=params),
        make_input_fn(train_filenames, params.batch_size),
        make_input_fn(eval_filenames, params.batch_size),
        train_steps=(params.epochs * TRAIN_SIZE) // params.batch_size,
        eval_steps=EVAL_SIZE // params.batch_size,
    )


def main(_args):
    """Run experiment_fn with RunConfig and HParams."""
    tf.contrib.learn.learn_runner.run(
        experiment_fn,
        schedule=FLAGS.schedule,
        run_config=tf.contrib.learn.RunConfig(model_dir=FLAGS.modeldir),
        hparams=get_default_hparams().parse(FLAGS.hparams),
    )


if __name__ == '__main__':
    tf.app.run(main)
