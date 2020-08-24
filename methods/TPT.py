# -*- coding: utf-8 -*-

import os
import numpy as np
import time
import random
import argparse
import logging

import matplotlib.pyplot as plt
from scipy import stats
import scipy.signal
from hmmlearn import hmm
# from emd import emd
from pyemd import emd as emd_pele
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from joblib import Memory, Parallel, delayed
import scipy.spatial.distance

from accel.feature_extraction import extract_psd_features

memory = Memory('./savefile', verbose=0)


## Compute the EMD kernel
## X1 and X2 are lists with data from multiple participants as elements
@memory.cache
def compute_emd_original(X1, X2, dist):
    sz1 = len(X1)
    sz2 = len(X2)
    D = np.zeros((sz1, sz2))
    for i in range(0, sz1):
        res = Parallel(n_jobs=-1,
                       verbose=10)(delayed(emd)(X1[i], X2[j], distance=dist)
                                   for j in range(i, sz2))
        D[i, i:] = res
        # for j in range(i,sz2):
        #     startT = time.time()
        #     D[i,j] = (emd(X1[i],X2[j],distance=dist))
        #     endT = time.time() - startT
        #     print('EMD took ' + str(endT) + ' seconds.')
    D = D + np.transpose(np.triu(D, k=1))
    #D2 = np.exp((-1/np.mean(D[np.nonzero(D)]))*D)
    return D


## Compute the EMD kernel when X1 only has one element
@memory.cache
def compute_emd_original_1d(X1, X2, dist):
    assert len(X1) == 1
    sz2 = len(X2)
    res = Parallel(n_jobs=-1,
                   verbose=10)(delayed(emd)(X1[0], X2[j], distance=dist)
                               for j in range(0, sz2))
    return res


def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal
    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.
    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.
    Examples
    --------
    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])
    See Also
    --------
    pieces : Calculate number of pieces available by sliding
    """
    if axis >= data.ndim:
        raise ValueError("Axis value out of range")

    if stepsize < 1:
        raise ValueError("Stepsize may not be zero or negative")

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis")

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize +
                           1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(data,
                                              shape=shape,
                                              strides=strides)

    if copy:
        return strided.copy()
    else:
        return strided


def my_emd(D1, D2, distance='sqeuclidean', num_clusters=40):
    # cluster each dataset
    cls_d1 = KMeans(n_clusters=num_clusters).fit(D1)
    cls_d2 = KMeans(n_clusters=num_clusters).fit(D2)

    clusters_d1 = cls_d1.cluster_centers_
    clusters_d2 = cls_d2.cluster_centers_

    # get the histograms
    hist_d1 = np.zeros(num_clusters)
    hist_d2 = np.zeros(num_clusters)
    for i in range(0, num_clusters):
        hist_d1[i] = sum(cls_d1.labels_ == i)
        hist_d2[i] = sum(cls_d2.labels_ == i)

    assert sum(hist_d1) == sum(hist_d2)

    # concatenate for emd calculation
    emd_hist_1 = np.hstack(
        (hist_d1, np.array([0 for i in range(0, num_clusters)])))
    emd_hist_2 = np.hstack(
        (np.array([0 for i in range(0, num_clusters)]), hist_d2))

    # get the distance matrix
    centers_hist = np.vstack((clusters_d1, clusters_d2))

    distance_matrix = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(centers_hist, metric=distance))

    # run emd
    dist = emd_pele(emd_hist_1, emd_hist_2, distance_matrix)
    return dist


## Compute the EMD kernel
## X1 and X2 are lists with data from multiple participants as elements
@memory.cache
def compute_emd(X1, X2, dist):
    sz1 = len(X1)
    sz2 = len(X2)
    D = np.zeros((sz1, sz2))
    for i in range(0, sz1):
        # res = Parallel(n_jobs=-1, verbose=10)(delayed(my_emd)(X1[i], X2[j], distance=dist) for j in range(i,sz2))
        # D[i,i:] = res
        for j in range(i, sz2):
            startT = time.time()
            D[i, j] = (my_emd(X1[i], X2[j], distance=dist))
            endT = time.time() - startT
            # print('EMD took ' + str(endT) + ' seconds.')
    D = D + np.transpose(np.triu(D, k=1))
    return D


## Compute the EMD kernel when X1 only has one element
@memory.cache
def compute_emd_1d(X1, X2, dist):
    assert len(X1) == 1
    sz2 = len(X2)
    D = np.zeros(sz2)
    for j in range(0, sz2):
        D[j] = (my_emd(X1[0], X2[j], distance=dist))
    # res = Parallel(n_jobs=-1, verbose=10)(delayed(emd)(X1[0], X2[j], distance=dist) for j in range(0,sz2))
    return D





def get_annotation(annotations, l):
    assert l < 9
    assert annotations.shape[0] % 9 == 0

    ekin_subjects = [2,21,36,53,74,3,22,38,54,75,5,23,39,56,76,6,24,40,57,77,8,25,42,59,78,9,26,43,60,80,10,27,44,65,81,12,28,45,66,82,13,29,46,67,83,14,30,47,69,85,15,31,49,70,87,16,32,50,71,88,17,34,51,72,89,20,35,52,73,92]
    ekin_subjects = sorted(ekin_subjects)
    assert len(ekin_subjects) == 70
    ekin_subjects = np.array(ekin_subjects) - 1

    selected_label = annotations[l::9,:]

    filtered_labels = selected_label[ekin_subjects,:]

    return filtered_labels


def get_labels(annotations, window_size=64, stride=64):
    '''
    Returns labels from input annotations
    :param annotations: ndarray, of shape (num_participants, num_frames)
    '''
    print('extracting labels')

    all_labels = list()

    sz = annotations.shape[1]
    for p in range(0, annotations.shape[0]):
        curr_labels = list()

        for j in range(0, sz - window_size, stride):
            curr_annotations = annotations[p, j:j + window_size]

            label = None
            if sum(curr_annotations == 1) > sum(curr_annotations == 0):
                label = 1
            else:
                label = 0

            curr_labels.append(label)

        all_labels.append(np.array(curr_labels))

    return all_labels


@memory.cache
def get_features(accel_signals, window_size=64, stride=64):
    '''
    Returns features and labels from input data
    :param accel_signals: ndarray, of shape (num_participants, num_frames, num_features)
    
    :returns all_features: list of list of bags (lists) of features, shape (num_participants, num_bags, num_features)
    :returns all_labels: list of array with labels, shape (num_participants, num_bags, num_labels)
    '''
    print('extracting data')

    all_features = list()
    for p in range(0, accel_signals.shape[0]):

        scaled_signals = np.squeeze(StandardScaler().fit_transform(
            accel_signals[p, :, :]))
        sz = scaled_signals.shape[0]
        curr_features = list()

        for j in range(0, sz - window_size, stride):
            # labels = annotations[p, j:j + window_size]
            # label = labels.mean() >= 0.5

            curr_features.append(
                feature_extractor(scaled_signals[j:j + window_size, :]))

        all_features.append(np.vstack(curr_features))

    return all_features


# @memory.cache
def get_trajectories(accel_signals,
                     window_size=64,
                     stride=64,
                     trajectory_length=12,
                     trajectory_stride=4):
    '''
    :param accel_signals: ndarray, of shape (num_participants, num_windows, num_features)
    :returns:
    features: list of list of bags (lists) of features, shape (num_participants, num_bags, num_local_features, num_dimensions)
    labels: list of array with labels, shape (num_participants, num_bags)
    '''

    print('extracting data')
    all_features = list()
    # for each person
    for p in range(0, accel_signals.shape[0]):

        # magnitude_signal = np.linalg.norm(accel_signals[p], axis=1)
        # scaled_signal = magnitude_signal / np.sum(magnitude_signal)
        # sz = scaled_signal.shape[0]
        person_signal = accel_signals[p]
        sz = person_signal.shape[0]
        curr_features = list()

        # slide the window
        for j in range(0, sz - window_size, stride):

            windows = sliding_window(person_signal[j:j + window_size, :],
                                     size=trajectory_length,
                                     stepsize=trajectory_stride,
                                     axis=0)

            assert windows.ndim == 3
            new_windows = np.empty(
                (windows.shape[0], windows.shape[1] * windows.shape[2]))
            for i in range(0, len(windows)):
                new_windows[i, 0::3] = windows[i, 0]
                new_windows[i, 1::3] = windows[i, 1]
                new_windows[i, 2::3] = windows[i, 2]

            curr_features.append(new_windows)

        all_features.append(curr_features)

    return all_features


def test_features(args):

    # load accel files
    accel_file_path = os.path.join(os.getcwd(), args.input)
    accel = np.loadtxt(accel_file_path, dtype=np.float64,
                       delimiter=',')[:-1, :]

    num_participants = accel.shape[1] / 3
    accel = np.stack(np.split(accel, num_participants, axis=1))
    print('Loaded accel file. shape: {:s}'.format(str(accel.shape)))

    # load labels
    labels_file_path = os.path.join(os.getcwd(), args.labels)
    labels = np.transpose(
        np.loadtxt(labels_file_path, dtype=np.int8, delimiter=','))
    print('Loaded labels file. shape: {:s}'.format(str(labels.shape)))

    # check that stuff makes sense
    # assert accel.shape[0] == labels.shape[0]

    X, Y = get_data(accel, labels)
    fig, ax = plt.subplots()

    all_X = np.vstack(X)
    df = pd.DataFrame(all_X[:, 0:6], columns=['a', 'b', 'c', 'd', 'e', 'f'])
    pd.plotting.scatter_matrix(df, alpha=0.2, ax=ax)
    # ax.hist(all_X[:,args.feature], bins=80, range=(-5,5))
    # ax.grid()
    plt.show()


def test_feature_extractor(args):
    # load accel files
    accel_file_path = os.path.join(os.getcwd(), args.input)
    accel = np.loadtxt(accel_file_path, dtype=np.float64,
                       delimiter=',')[0:-1, :]
    print('Loaded accel file. shape: {:s}'.format(str(accel.shape)))

    assert accel.shape[1] % 3 == 0

    num_participants = accel.shape[1] / 3
    accel = np.stack(np.split(accel, num_participants, axis=1))
    print(accel.shape)

    # load labels
    labels_file_path = os.path.join(os.getcwd(), args.labels)
    labels = np.transpose(
        np.loadtxt(labels_file_path, dtype=np.int8, delimiter=','))
    print('Loaded labels file. shape: {:s}'.format(str(labels.shape)))

    X, Y = get_data(accel, labels)
    print(len(X))
    print(X[0].shape)
    print(Y[0].shape)

    for i in range(0, len(X)):
        output_file_path = os.path.join(os.getcwd(), args.output,
                                        'participant{:d}.csv'.format(i))
        D = np.hstack((Y[i], X[i]))
        np.savetxt(output_file_path, D, delimiter=',')


def test_psd(args):
    accel_file_path = os.path.join(os.getcwd(), args.input)

    accel = np.loadtxt(accel_file_path, dtype=np.float32, delimiter=',')

    print(accel.shape)
    #psds = feature_extractor(accel[:,1:])

    while True:
        ini = np.random.randint(0, accel.shape[0] - args.window_size)

        signal = accel[ini:ini + args.window_size, 1]
        f, psd = scipy.signal.periodogram(x=signal,
                                          fs=args.fs,
                                          nfft=args.window_size,
                                          return_onesided=True)
        print(f.shape)
        print(psd.shape)

        print(f)
        print(psd)
        ids = np.nonzero(
            np.logical_or.reduce((f == 0, f == 0.125, f == 0.25, f == 0.5,
                                  f == 1, f == 2, f == 4, f == 8)))
        print(ids[0])

        plt.subplot(311)
        plt.plot(list(range(0, len(signal))), signal)
        plt.subplot(312)
        plt.plot(f, psd)

        # binning the psd
        binned_psd = np.zeros(args.bins)

        # if the bins have power of two bounds
        if 2**(args.bins - 1) == args.window_size:
            print('here')
            binned_psd[0] = psd[0]
            binned_psd[1] = psd[1]
            i = 2
            j = 2
            bin_size = 1
            while bin_size <= args.window_size / 4:
                for k in range(0, bin_size):
                    binned_psd[i] += psd[j]
                    j += 1
                i += 1
                bin_size *= 2

        else:
            bin_bounds = np.logspace(0, np.log10(args.window_size / 2),
                                     args.bins - 1)
            binned_psd[0] = psd[0]
            i = 1
            for j in range(0, len(bin_bounds)):
                while i <= bin_bounds[j]:
                    print(j)
                    binned_psd[j + 1] += psd[i]
                    i += 1

        print(psd)
        print(binned_psd)

        plt.subplot(313)
        plt.bar(range(0, len(binned_psd)), binned_psd)
        plt.show()

        a = raw_input('press key to recalc periodogram.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs some simple code using the feature extractor.')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_psd_test = subparsers.add_parser('psd')
    parser_psd_test.add_argument('--input',
                                 '-i',
                                 required=True,
                                 help='input accel file')

    parser_psd_test.add_argument('-w',
                                 '--window-size',
                                 type=float,
                                 default=64,
                                 help="window size")
    parser_psd_test.add_argument('-f',
                                 '--fs',
                                 type=float,
                                 default=20,
                                 help="Sampling frequency")
    parser_psd_test.add_argument('-n',
                                 '--bins',
                                 type=int,
                                 default=6,
                                 help="Frequency bins")
    parser_psd_test.set_defaults(func=test_psd)

    parser_fe_test = subparsers.add_parser('extractor')
    parser_fe_test.add_argument('--input',
                                '-i',
                                required=True,
                                help='accel input files')
    parser_fe_test.add_argument('--labels',
                                '-l',
                                required=True,
                                help='annotation file')
    parser_fe_test.add_argument('--output',
                                '-o',
                                required=True,
                                help='folder to write text files')
    parser_fe_test.set_defaults(func=test_feature_extractor)

    parser_fe_test = subparsers.add_parser('features')
    parser_fe_test.add_argument('--input',
                                '-i',
                                required=True,
                                help='accel input files')
    parser_fe_test.add_argument('--labels',
                                '-l',
                                required=True,
                                help='annotation file')
    parser_fe_test.add_argument('-f',
                                '--feature',
                                type=int,
                                default=22,
                                help="feature to show")
    parser_fe_test.set_defaults(func=test_features)

    parser.add_argument('-d',
                        '--debug',
                        help="Print lots of debugging statements",
                        action="store_const",
                        dest="loglevel",
                        const=logging.DEBUG,
                        default=logging.WARNING)
    parser.add_argument('-v',
                        '--verbose',
                        help="Be verbose",
                        action="store_const",
                        dest="loglevel",
                        const=logging.INFO)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    args.func(args)
