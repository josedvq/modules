import argparse
import logging
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class DtDataset(Dataset):
    ''' Loads a dense trajectory dataset in memory. '''

    def __init__(self, dt_path, labels_path, bag_idx, feature_idx):

        # bag info
        self.dt_path = dt_path
        self.labels_path = labels_path

        # load labels
        self.labels = np.loadtxt(labels_path, delimiter=',')

        # load DTs
        bags = np.loadtxt(dt_path, delimiter=',')
        last_elem_idx = np.where(np.diff(bags[:, bag_idx]))[0]
        self.bags = np.split(bags[:, feature_idx], last_elem_idx + 1)

    def __getitem__(self, idx):
        return self.bags[idx]

    def __del__(self):
        return len(self.bags)

def main(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the MILES classifier.')

    parser.add_argument('--input', required=True,
                        help='input folder with the dense trajectories')
    parser.add_argument('--labels_path', required=True,
                        help='input folder with labels')
    parser.add_argument('--output', required=True,
                        help='output folder for the dataset')
    parser.add_argument('--labels-out', required=True,
                        help='output folder for the dataset')

    parser.add_argument('--num-frames', type=int, default=26400)
    parser.add_argument('--window-size', type=int, default=20)
    parser.add_argument('--sample', type=int, default=200)

    parser.add_argument('-d', '--debug', help="Print lots of debugging statements",
                        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('-v', '--verbose', help="Be verbose",
                        action="store_const", dest="loglevel", const=logging.INFO)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    main(args)