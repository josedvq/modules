import argparse
import logging
import gzip
import random
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader

from bag import Bag
from helpers.file_functions import list_files, get_last_line

class DtFileDataset(Dataset):
    ''' Represents a bag or window. '''

    def __init__(self, dt_path, labels, bags):

        # bag info
        self.dt_path = dt_path
        self.bags = bags
        self.labels = labels

        self.fh = gzip.open(dt_path, 'rt')
        if type(labels) == str:
            self.labels = np.loadtxt(labels)
        elif type(labels) == np.ndarray:
            self.labels = labels
        else:
            raise 'weird type for labels'

        self.num_frames = len(self.labels)

        # ll = np.fromstring(get_last_line(dt_path), sep='\t')
        # self.num_frames = ll[0].item()


    def __getitem__(self, idx):
        raise 'not implemented'

    def __del__(self):
        self.fh.close()

    def get_curr_frame_trajectories(self):
        # move to the right frame in the three files
        frame_trajectories = list()
        dt = np.fromstring(next(self.fh), sep='\t')
        frame_num = int(dt[0])

        while int(dt[0]) == frame_num:
            frame_trajectories.append(dt)
            dt = np.fromstring(next(self.fh), sep='\t')
            
            if int(dt[0]) % 100 == 0:
                logging.debug(
                    'dt[{:d}][0] = {:.0f}'.format(frame_num, dt[0]))

        return frame_num-15, frame_trajectories

    def output_bag(self, fout, lout, curr_bag):
        self.bags[curr_bag].bag_idx = curr_bag
        self.bags[curr_bag].set_labels(self.labels[:,None][self.bags[curr_bag].frames, :])
        self.bags[curr_bag].save_labels(lout)
        self.bags[curr_bag].save_trajectories(fout)
        print('bag output. id={:d}, trajectories={:d}'.format(curr_bag, len(self.bags[curr_bag].trajectories)))

    def to_dataset_no_bbs(self, fout, lout):

        curr_bag = 0
        frame_num = 0
        while frame_num < self.num_frames:

            try:
                # get traj from next frame in the file (there might be jumps)
                frame_num, frame_trajectories = self.get_curr_frame_trajectories()
                # print('{:d}: {:d}'.format(frame_num, len(frame_trajectories)))

                # add trajectories to bags that contain the frame
                b = curr_bag
                while b < len(self.bags) and frame_num in self.bags[b].frames:
                    self.bags[b].trajectories += frame_trajectories
                    b += 1
            except StopIteration:
                frame_num = self.num_frames
                frame_trajectories = list()

            # output bags if neccesary
            while curr_bag < len(self.bags) and frame_num > max(self.bags[curr_bag].frames):
                logging.debug('bag filled')
                self.output_bag(fout, lout, curr_bag)
                
                self.bags[curr_bag] = None
                curr_bag += 1

            

    def to_dataset_bbs(self, fout):
        raise 'not implemented'

    def to_dataset(self, output_path, labels_path):
        assert len(self.bags) > 0
        with open(output_path, 'w') as fout:
            with open(labels_path, 'w') as lout:
                if self.bags[0].bbs is None:
                    self.to_dataset_no_bbs(fout, lout)
                else:
                    self.to_dataset_bbs(fout, lout)

def create_non_overlaping_bags(window_size, num_frames):
    ini = 0
    bags = list()
    while ini + window_size < num_frames - window_size:
        bags.append(Bag(
            frames=range(max(0, ini), min(num_frames, ini + window_size)),
            subject_idx=0,
            bag_idx=0
        ))
        ini += window_size

    return bags

def main(args):

    # # create some bags
    # bags = list()
    # bags.append(Bag(
    #     frames=range(5,15),
    #     subject_idx=0,
    #     bag_idx=0
    # ))

    # dataset = DtFileDataset(args.input, bags=bags)

    # dataset.to_dataset(args.output)

    # get files in folder

    # create and sample bags
    traj_files = list_files(args.input)
    labels = np.loadtxt(args.labels, delimiter=',')

    assert len(traj_files) == labels.shape[1]

    for subject, traj_file in enumerate(traj_files):
        bags = create_non_overlaping_bags(args.window_size, args.num_frames)
        bags = [bags[i] for i in sorted(random.sample(range(len(bags)), args.sample))]

        dataset = DtFileDataset(traj_file, labels[:,subject], bags=bags)

        out_file = os.path.join(args.output, os.path.basename(traj_file))
        labels_out_file = os.path.join(args.labels_out, os.path.splitext(os.path.basename(traj_file))[0])
        dataset.to_dataset(out_file, labels_out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the MILES classifier.')

    parser.add_argument('--input', required=True,
                        help='input folder with the dense trajectories')
    parser.add_argument('--labels', required=True,
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