import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class Bag:
    ''' Represents a bag or window. '''

    def __init__(self, frames=None, subject_idx=None, bag_num=None, bag_idx=None):

        # bag info
        self.frames = frames
        self.subject_idx = subject_idx
        self.bag_num = bag_num
        self.bag_idx = bag_idx
        self.bbs = None

        self.trajectories = list()
        self.cluster_labels = None
        self.labels = None

    def set_labels(self, frame_labels):
        majority_labels = np.round(np.mean(frame_labels,axis=0)).astype(int)
        miles_labels = (np.sum(frame_labels,axis=0) > 5).astype(int)
        self.labels = np.concatenate((majority_labels, miles_labels))

    def save_trajectories(self, fout):
        for i, t in enumerate(self.trajectories):
            cluster_label = 0
            if self. cluster_labels is not None:
                cluster_label = self.cluster_labels[i]
            fout.write(str(t[0]) + ',' + str(self.subject_idx) + ',' + str(self.bag_num) + ',' + str(cluster_label) + ',' + ','.join(map(str, t[1:])) + '\n')

    def save_labels(self, fout):
        np.savetxt(fout,self.labels[None,:])

    def save_info(self, fout):
        fout.write(str(self.bag_num) + ', ' + str(self.bag_idx) + ', ' + str(self.subject_idx) + ', ' + str(min(self.frames)) + ', ' + str(max(self.frames)) + ', ' + str(len(self.trajectories)) + '\n')


    def kmeans_cluster(self, k=20, cluster_features=None):
        if cluster_features is None:
            cluster_features = range(0,len(self.trajectories[0]))

        # holds output
        new_bag = Bag()
        new_bag.frames = self.frames
        new_bag.subject_idx = self.subject_idx
        new_bag.bag_num = self.bag_num
        new_bag.bag_idx = self.bag_idx
        new_bag.labels = np.copy(self.labels)

        # cluster trajectories if necessary
        if len(self.trajectories) > k:

            original_trajectories = np.array(self.trajectories)

            # only for clustering, get and scale features
            trajectories_for_clustering = original_trajectories[:, cluster_features]
            trajectories_for_clustering = StandardScaler().fit_transform(trajectories_for_clustering)
            kmeans = KMeans(n_clusters=k).fit(trajectories_for_clustering)
            self.cluster_labels = kmeans.labels_
            # get the full cluster centers
            cluster_centers = np.empty(
                (k, original_trajectories.shape[1]))
            # for each cluster
            for l in range(0, k):
                # get the ids of the trajectories in the cluster
                cluster_idxs = np.nonzero(self.cluster_labels == l)[0]
                # get the mean of the cluster on the full trajectories
                cluster_trajectories = original_trajectories[cluster_idxs, :]
                cluster_centers[l, :] = np.mean(cluster_trajectories, axis=0)


            # kmeans.cluster_centers_
            new_bag.trajectories = [t for t in cluster_centers]
            new_bag.cluster_labels = np.array(list(range(0,k)))

        else:
            raise 'not implemented'

        return new_bag


def main(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the MILES classifier.')

    parser.add_argument('-d', '--debug', help="Print lots of debugging statements",
                        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('-v', '--verbose', help="Be verbose",
                        action="store_const", dest="loglevel", const=logging.INFO)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    main(args)
