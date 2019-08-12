import argparse
import logging
import gzip
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, IterableDataset, DataLoader

from helpers.file_functions import list_files, get_last_line

class DtDatasetHdd(IterableDataset):
    ''' Loads a dense trajectory dataset in memory. '''

    def __init__(self, dt_path, labels_path, use_gzip=True):

        # bag info
        self.dt_path = dt_path
        self.labels_path = labels_path
        self.use_gzip = use_gzip

        self.fh = None
        if self.use_gzip:
            self.fh = gzip.open(dt_path)
        else:
            self.fh = open(dt_path)

        self.labels = np.loadtxt(labels_path)[:,None]

        # get the last ID, this should be the size of the dataset
        # ll = np.fromstring(get_last_line(dt_path), sep='\t')
        # self.num_frames = ll[0].item()

        self.last_dt = None

    def __iter__(self):
        self.fh.seek(0)
        self.i = 0
        self.last_dt = None
        return self

    def __next__(self):
        if self.last_dt is None:
            dt = np.fromstring(next(self.fh), sep=',')
        else:
            dt = self.last_dt

        # print(dt[2])

        if dt[2] == self.i:
            bag = list()
            bag.append(dt)
            while dt[2] == self.i:
                bag.append(dt)
                dt = np.fromstring(next(self.fh), sep=',')
            self.last_dt = dt
            bag = np.stack(bag)
        else:
            bag = np.array([])

        self.i += 1
        return bag, np.squeeze(self.labels[self.i, :])

    def __del__(self):
        self.fh.close()

def main(args):
    dataset = DtDatasetHdd(args.input, args.labels, use_gzip=False)
    for X, Y in dataset:
        print(X.shape)
        print(Y.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the MILES classifier.')

    parser.add_argument('--input', required=True,
                        help='input folder with the dense trajectories')
    parser.add_argument('--labels', required=True,
                        help='input folder with labels')

    parser.add_argument('-d', '--debug', help="Print lots of debugging statements",
                        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('-v', '--verbose', help="Be verbose",
                        action="store_const", dest="loglevel", const=logging.INFO)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    main(args)