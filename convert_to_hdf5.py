#!/usr/bin/env python

"""Convert CSV-based dataset into the HDF5 format."""


import argparse
import os
import h5py


def convert_dataset(csvdir, output):
    nfiles = 119080
    nvals = 2406
    with h5py.File(output, 'w') as fd:
        data = fd.create_dataset('deepsense/train', (nfiles, nvals), dtype='f4')
        for i, csvfile in enumerate(os.listdir(csvdir)):
            print('\r{:0.2%}'.format((i+1) / nfiles), end='')
            with open(os.path.join(csvdir, csvfile)) as fd:
                for j, val in enumerate(fd.read().split(',')):
                    data[i, j] = float(val)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csvdir', help='directory with CSV dataset')
    ap.add_argument('output', help='output file name')
    args = ap.parse_args()
    convert_dataset(args.csvdir, args.output)


if __name__ == '__main__':
    main()
