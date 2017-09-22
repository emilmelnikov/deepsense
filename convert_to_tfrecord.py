#!/usr/bin/env python

import os
import tarfile
import tensorflow as tf


tf.flags.DEFINE_string('src', 'sepHARData_a.tar.gz', 'original dataset')
tf.flags.DEFINE_string('destdir', 'data', 'output directory')
FLAGS = tf.flags.FLAGS


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def serialize(writer, contents):
    values = [float(val) for val in contents.split(b',')]
    features = tf.train.Features(feature=dict(
        acc=float_feature(values[:1200]),
        gyro=float_feature(values[1200:2400]),
        label=int64_feature([values[2400:].index(1)]),
    ))
    example = tf.train.Example(features=features)
    writer.write(example.SerializeToString())


def examples(archive):
    for ientry, item in enumerate(archive):
        if not item.isfile():
            continue

        prefix, basename = os.path.split(item.name)
        if basename.startswith('.') or not basename.endswith('.csv'):
            continue

        _prefix, dataset_type = os.path.split(prefix)
        if dataset_type not in ('train', 'eval'):
            continue

        with archive.extractfile(item) as buf:
            contents = buf.read()
        yield ientry, contents, dataset_type


def convert(archive, train_writer, eval_writer):
    train_examples = 0
    eval_examples = 0

    for ientry, contents, dataset_type in examples(archive):
        if dataset_type == 'train':
            serialize(train_writer, contents)
            train_examples += 1
        elif dataset_type == 'eval':
            serialize(eval_writer, contents)
            eval_examples += 1

        print('\rtrain:{:<8} eval:{:<8} examples:{:<8} entries:{:<8}'.
              format(train_examples, eval_examples,
                     train_examples + eval_examples, ientry + 1),
              end='')

    print()


def main():
    os.makedirs(FLAGS.destdir, mode=0o755, exist_ok=True)
    train_file = os.path.join(FLAGS.destdir, 'train.tfrecords')
    eval_file = os.path.join(FLAGS.destdir, 'eval.tfrecords')

    with tarfile.open(FLAGS.src) as archive, \
         tf.python_io.TFRecordWriter(train_file) as train_writer, \
         tf.python_io.TFRecordWriter(eval_file) as eval_writer:
        convert(archive, train_writer, eval_writer)


if __name__ == '__main__':
    main()
