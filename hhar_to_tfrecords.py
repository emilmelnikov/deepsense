#!/usr/bin/env python


import collections
import typeing
import logging
import os
import zipfile
import tensorflow as tf
import pandas as pd


log = logging.getLogger()


tf.flags.DEFINE_string(
    'src',
    'Activity_recognition_exp.zip',
    'path to activity recognition dataset archive',
)
tf.flags.DEFINE_string(
    'destdir',
    'data',
    'output directory',
)
FLAGS = tf.flags.FLAGS


class Record(typing.NamedTuple):
    time: int
    x: float
    y: float
    z: float
    user_id: int


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def serialize(writer, record):
    values = [float(val) for val in contents.split(b',')]
    features = tf.train.Features(feature=dict(
        x=float_feature(values[:1200]),
        y=float_feature(values[1200:2400]),
        label=int64_feature([values[2400:].index(1)]),
    ))
    example = tf.train.Example(features=features)
    writer.write(example.SerializeToString())


def parse_records(archive, filename):
    records = []
    with archive.open(filename, 'r') as fd:
        # Skip CSV header:
        # Index,Arrival_Time,Creation_Time,x,y,z,User,Model,Device,gt
        fd.readline()
        for line in fd:
            values = line.strip().split(b',')
            record = Record(
                time=int(values[2]),
                x=float(values[3]),
                y=float(values[4]),
                z=float(values[5]),
                user_id=ord(values[6])-ord('a'),
            )
            records.append(record)
    return records


class IntervalBuffer:
    """Store items and partition them based on time intervals."""

    def __init__(self, interval, start_time=0):
        self._items = []
        self._frames = collections.deque()
        self._interval = interval
        self._start_time = start_time

    def append(self, item, time):
        while time - self._start_time >= self._interval:
            self._frames.append(self._items)
            self._items = []
            self._start_time += self._interval
        self._items.append(item)

    def popframe(self):
        return self._frames.popleft() if self._frames else None

    def __iter__(self):
        return iter(self._frames)


def convert(writer, archive):
    with archive.open('Watch_accelerometer.csv') as fd:
        acc = pd.read_csv(fd)
    with archive.open('Watch_gyroscope.csv') as fd:
        gyro = pd.read_csv(fd)


    acc_recs = parse_records(archive, 'Watch_accelerometer.csv')
    gyro_recs = parse_records(archive, 'Watch_gyroscope.csv')

    # Creation time seems to be specififed in nanoseconds.
    # Interval time is 250 milliseconds.
    buf = IntervalBuffer(250_000_000, start_time=acc_recs[0].time)

    for acc in acc_recs:
        buf.append(acc, acc.time)

    for frame in buf:
        pass


def main(_args):
    os.makedirs(FLAGS.destdir, mode=0o755, exist_ok=True)
    data_file = os.path.join(FLAGS.destdir, 'data.tfrecords')
    with zipfile.ZipFile(FLAGS.src) as archive, \
         tf.python_io.TFRecordWriter(data_file) as writer:
        convert(writer, archive)


if __name__ == '__main__':
    tf.app.run()
