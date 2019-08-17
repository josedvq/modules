import argparse
import logging
import gzip
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, IterableDataset, DataLoader

from dts.feature_map import get_non_rotated_fmap

class DtDatasetHdd(IterableDataset):
    ''' Loads a dense trajectory dataset in memory. '''

    def __init__(self, dt_path, feature_idx, labels_path=None, use_gzip=True, num_bags=200):

        # bag info
        self.dt_path = dt_path
        self.feature_idx = feature_idx
        self.labels_path = labels_path
        self.use_gzip = use_gzip
        self.num_bags = num_bags

        # self.fh = None
        # if self.use_gzip:
        #     self.fh = gzip.open(dt_path)
        # else:
        #     self.fh = open(dt_path)

        # self.labels = None
        # if self.labels_path:
        #     self.labels = np.loadtxt(labels_path)[:,None]

        # get the last ID, this should be the size of the dataset
        # ll = np.fromstring(get_last_line(dt_path), sep='\t')
        # self.num_frames = ll[0].item()

        # self.last_dt = None

    # def __iter__(self):
    #     self.fh.seek(0)
    #     self.i = 0
    #     self.last_dt = None
    #     return self

    # def __next__(self):
    #     if self.last_dt is None:
    #         dt = np.fromstring(next(self.fh), sep=',')
    #     else:
    #         dt = self.last_dt

    #     if dt[2] == self.i:
    #         bag = list()
    #         bag.append(dt[self.feature_idx])
    #         while dt[2] == self.i:
    #             bag.append(dt[self.feature_idx])
    #             dt = np.fromstring(next(self.fh), sep=',')
    #         self.last_dt = dt
    #         bag = np.stack(bag)
    #     else:
    #         bag = np.array([])

    #     self.i += 1
    #     if self.labels is not None:
    #         return bag, np.squeeze(self.labels[self.i, :])
    #     else: 
    #         return bag

    # def __del__(self):
    #     self.fh.close()

    def sample_lines(self, rate):
        lines = list()
        for line in self.fh:
            if np.random.sample() < rate:
                dt = np.fromstring(line, sep=',')
                lines.append(dt[self.feature_idx])

        return np.stack(lines)

    def sample_lines_fast(self, rate):
        arrs = [np.empty((1000, len(self.feature_idx)))]
        cnt = 0

        if self.use_gzip:
            fh = gzip.open(self.dt_path)
        else:
            fh = open(self.dt_path)

        for line in fh:

            if np.random.sample() < rate:
                if cnt == 1000:
                    print('  buffer filled')
                    arrs.append(np.empty((1000, len(self.feature_idx))))
                    cnt = 0
                arrs[-1][cnt, :] = (np.fromstring(line,
                                                  sep=','))[self.feature_idx]
                cnt += 1
        fh.close()

        arrs[-1] = arrs[-1][:cnt, :]
        D = np.vstack(arrs)

        return D

    def sample_lines_faster(self, rate):
        ds = np.loadtxt(self.dt_path, delimiter=',')
        print(ds.shape)
        idxs = np.random.choice(len(ds), int(rate*len(ds)), replace=False)
        print(idxs.shape)
        return ds[idxs,:]


    def get_full_dataset(self):
        trajs = np.loadtxt(self.dt_path, delimiter=',')

        def get_next_bag_idx(i):
            if i == len(trajs) - 1:
                return self.num_bags
            else:
                 return trajs[i+1, 2]

        bags = list()

        i = 0
        bag = np.empty((20000, len(self.feature_idx)))
        bag_i = 0
        bag_idx = 0

        while bag_idx < self.num_bags:

            # print(trajs[i, 2])
            if i < len(trajs) and trajs[i, 2] == bag_idx:
                while i < len(trajs) and trajs[i, 2] == bag_idx:
                    bag[bag_i] = trajs[i, self.feature_idx]
                    bag_i += 1

                    # update
                    i += 1
                bags.append(bag[:bag_i, :])
                bag_i = 0
            else: 
                bags.append(np.zeros((0, len(self.feature_idx))))

            bag_idx += 1
        return bags

    def get_full_dataset_legacy(self):
        bags = np.loadtxt(self.dt_path, delimiter=',')

        last_elem_idx = np.where(np.diff(bags[:, 2]))[0]
        bags = np.split(bags[:, self.feature_idx], last_elem_idx + 1)

        return bags

def main(args):
    fmap = get_non_rotated_fmap(15)
    dataset = DtDatasetHdd(args.input, fmap['traj'], use_gzip=False)
    # ds1 = dataset.get_full_dataset_legacy()
    ds2 = dataset.get_full_dataset()

    # print('len(ds1): {:d}'.format(len(ds1)))
    print('len(ds2): {:d}'.format(len(ds2)))

    total_len = 0
    for bag in ds2:
        total_len += len(bag)
        print(len(bag))

    print(total_len)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the MILES classifier.')

    parser.add_argument('--input', required=True,
                        help='input folder with the dense trajectories')
    parser.add_argument('--labels', required=False,
                        help='input folder with labels')

    parser.add_argument('-d', '--debug', help="Print lots of debugging statements",
                        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('-v', '--verbose', help="Be verbose",
                        action="store_const", dest="loglevel", const=logging.INFO)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    main(args)