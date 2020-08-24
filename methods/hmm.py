import argparse
import logging

import numpy as np
from sklearn.mixture import GaussianMixture

# seq: sequences with shape [num_participants, num_samples, num_features]
# state integer labels, with shape [num_participants, num_samples]
def train_hmm(seq, labels, num_states):
    assert len(seq) == len(labels)
    num_participants = len(seq)
    num_features = seq[0].shape[1]

    # hmm params
    startprob = np.zeros(num_states)
    # each row is P(S_t+1 | S_t)
    transmat = np.zeros((num_states,num_states))
    # means has shape [num_states, num_features]
    # covar has shape [num_states, num_features, num_features]
    state_features = [[] for i in range(0,num_states)]
    means = np.zeros((num_states,num_features))
    covars = np.zeros((num_states,num_features))

    for p, (curr_seq, curr_labels) in enumerate(zip(seq,labels)):
        num_samples = curr_seq.shape[0]

        assert curr_labels.shape[0] == num_samples

        # calculate start probabilities
        for s in range(0,num_states):
            startprob[s] += np.count_nonzero(curr_labels==s)
        
        # calculate state transition probabilities by counting all the state transitions
        for t in range(0,num_samples-1):
            transmat[curr_labels[t],curr_labels[t+1]] += 1

        # calculate observation probabilities by fitting a normal distrib per state observations        
        for s in range(0,num_states):
            # get the features from the state
            s_idxs = np.where(curr_labels == s)[0]
            s_feat = curr_seq[s_idxs,:]

            state_features[s].append(s_feat)

    # normalize startprob vector
    startprob /= np.sum(startprob)

    # normalize transmat along rows
    totals = np.sum(transmat,axis=1)
    transmat /= totals[:,np.newaxis]

    # calc mean and var from state features
    for s in range(0,num_states):
        s_feat = np.vstack(state_features[s])
        means[s,:] = np.mean(s_feat,axis=0)
        covars[s,:] = np.var(s_feat,axis=0)

    return startprob, transmat, means, covars


# seq: sequences with shape [num_participants, num_samples, num_features]
# state integer labels, with shape [num_participants, num_samples]
def train_gmm_hmm(seq, labels, num_states, num_mix):
    assert len(seq) == len(labels)
    num_participants = len(seq)
    num_features = seq[0].shape[1]

    # hmm params
    startprob = np.zeros(num_states)
    # each row is P(S_t+1 | S_t)
    transmat = np.zeros((num_states,num_states))
    # means has shape [num_states, num_features]
    # covar has shape [num_states, num_features, num_features]
    state_features = [[] for i in range(0,num_states)]

    for p, (curr_seq, curr_labels) in enumerate(zip(seq,labels)):
        num_samples = curr_seq.shape[0]

        assert curr_labels.shape[0] == num_samples

        # calculate start probabilities
        for s in range(0,num_states):
            startprob[s] += np.count_nonzero(curr_labels==s)
        
        # calculate state transition probabilities by counting all the state transitions
        for t in range(0,num_samples-1):
            transmat[curr_labels[t],curr_labels[t+1]] += 1

        # calculate observation probabilities by fitting a normal distrib per state observations        
        for s in range(0,num_states):
            # get the features from the state
            s_idxs = np.where(curr_labels == s)[0]
            s_feat = curr_seq[s_idxs,:]

            state_features[s].append(s_feat)

    # normalize startprob vector
    startprob /= np.sum(startprob)

    # normalize transmat along rows
    totals = np.sum(transmat,axis=1)
    transmat /= totals[:,np.newaxis]

    # train a GMM for each state
    weights = np.zeros((num_states,num_mix))
    means = np.zeros((num_states,num_mix,num_features))
    covars = np.zeros((num_states,num_mix,num_features))
    for s in range(0,num_states):
        s_feat = np.vstack(state_features[s])
        gmm = GaussianMixture(n_components=num_mix,covariance_type='diag').fit(s_feat)
        weights[s,:] = gmm.weights_
        means[s,:,:] = gmm.means_
        covars[s,:,:] = gmm.covariances_

    return startprob, transmat, weights, means, covars


def main(args):
    X = np.array([[
        [2,3],
        [4,5],
        [-2,-3],
        [-4,-5],
        [-2,-5],
        [-4,-3],
        [2,5],
        [4,3]
    ]])

    Y = np.array([[1,1,0,0,0,0,1,1]])

    startprob, transmat, means, covars = train_hmm(X,Y,2)

    print(startprob)
    print(transmat)
    print(means)
    print(covars)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the HMM learning.')

    parser.add_argument('-d', '--debug', help="Print lots of debugging statements",
                        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('-v', '--verbose', help="Be verbose",
                        action="store_const", dest="loglevel", const=logging.INFO)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    main(args)