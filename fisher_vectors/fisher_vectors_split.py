''' Fisher Vectors implementation
'''
import argparse
import logging
import os
import shutil
import math
import pickle

import joblib
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from memory_profiler import profile
from tqdm import tqdm

from fisher_vectors.improved_fisher_var import FisherVectorGMM
from helpers.tqdm import tqdm_joblib
from utils.descriptors import DescriptorMap, MotherDescriptorMap

class FisherVectors:
    ''' Fisher Vector's implementation.
    '''

    def __init__(self,
                 pca_components=0.95,
                 num_gmm_samples=100000,
                 num_gmm_components=256,
                 tmp_path='tmp'):
        self.pca_components = pca_components
        self.num_gmm_samples = num_gmm_samples
        self.num_gmm_components = num_gmm_components
        self.tmp_path = tmp_path
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)

        self.ss1 = None
        self.pca = None
        self.fv_gmm = None

        self.ss2 = None
        self.lr = None
        self.svm = None

        self.cv_results = None
        self.best_c = None
        self.scaled_c = None


    # GENERAL TRAINING
    def train_fv_gmm(self, X, num_gmm_components):
        ''' Trains the first stage of the FVs algo, up to the GMM
        '''
        # need to calculate FVs
        print('  sampling training set for GMM fit set')
        ss1 = StandardScaler().fit(X)
        X = ss1.transform(X)
        print('  calculating PCA')
        pca = PCA(n_components=self.pca_components, whiten=True).fit(X)
        num_final_features = pca.n_components_
        X = pca.transform(X)
        print('  components kept: {:d}, explained variance = {:f}'.format(
            num_final_features, np.sum(pca.explained_variance_ratio_)))

        print('  fitting GMM')
        fv_gmm = FisherVectorGMM(n_kernels=num_gmm_components).fit(
            X=X, verbose=True)
        print('  trained FV GMM')

        return {'ss':ss1, 'pca':pca, 'fv_gmm':fv_gmm}

    def train_all_fv_gmms(self, X, fmap):
        '''
        Trains separate gmms for different feature groups and different feature descriptors
        '''
        
        self.models = list()

        # for each feature group
        for g, X_g in enumerate(X):
            group_model = dict()

            # for each feature descriptor
            for desc_name, desc_idxs in fmap:
                print('Training FV GMM for group {:d}: {:s}'.format(g, desc_name))
                group_model[desc_name] = self.train_fv_gmm(X_g[:,desc_idxs], self.num_gmm_components)

            self.models.append(group_model)

    def compute_fvs(self, bags):
        assert len(bags) > 0
        assert self.ss1 is not None
        assert self.pca is not None
        assert self.fv_gmm is not None

        print('  bags: {:d}'.format(len(bags)))

        # transform the features
        for i,b in enumerate(bags):
            if b.size != 0:
                bags[i] = self.pca.transform(self.ss1.transform(b))
            else:
                bags[i] = np.zeros((0, self.pca.n_components_))
        # bags = [self.pca.transform(self.ss1.transform(b)) for b in bags]
        fvs = self.fv_gmm.predict(bags, normalized=True)
        fvs = fvs.reshape(((fvs.shape[0], fvs.shape[1] * fvs.shape[2])))

        return fvs

    def train_fv_gmm_and_compute_fvs(self, X, fvs_path):
        X_gmm = np.vstack(X)
        X_gmm = X_gmm[np.random.choice(
            len(X_gmm), size=self.num_gmm_samples, replace=False), :]
        self.train_fv_gmm(X_gmm)

        X = self.compute_fvs(X)
        print('  done!')

        if fvs_path:
            joblib.dump(X, fvs_path)

        return X

    def train_linear_svc(self, X, Y, C):
        svc = LinearSVC(loss='hinge', penalty='l2', max_iter=5000,
                        C=C).fit(X, Y)

        # Platt calibration on the train set
        scores = svc.decision_function(X).reshape(-1, 1)
        lr = LogisticRegression(C=1000000000,
                                solver='liblinear').fit(scores, Y)

        return svc, lr

    def train_linear_svc_with_sgd(self, X, Y, C):
        alpha = 1/C
        num_epochs = 8
        batch_size = 1000
        
        indices = range(0, len(X))
        batches = [indices[i:i+batch_size] for i in range(0, len(X), batch_size)]
        svc = SGDClassifier(loss='hinge', penalty='l2', alpha=alpha)

        for e in range(0, num_epochs):
            # perm = np.arange(0, len(X))
            # np.random.shuffle(perm)
            # X = X[perm]
            # Y = Y[perm]
            rng_state = np.random.get_state()
            np.random.shuffle(X)
            np.random.set_state(rng_state)
            np.random.shuffle(Y)

            for batch in batches:
                svc.partial_fit(X[batch[0]: batch[-1]], Y[batch[0]: batch[-1]], classes=np.unique(Y))
                # if svc.loss_curve_ is not None:
                #     print(svc.loss_curve_)

        # Platt calibration on the train set
        scores = svc.decision_function(X).reshape(-1, 1)
        lr = LogisticRegression(C=1000000000,
                                solver='liblinear').fit(scores, Y)

        return svc, lr

    def train_cv_linear_svc(self, X, Y, G=None, num_folds=4, c_values=None, use_sgd=True):
        '''
        Trains a Linear SVC using Platt scaling and optimizing for AUC using cross-validation.
        :param X: input data with shape (num_elements, num_features)
        :param Y: labels with shape (num_elements)
        :param G: group assignment for the elements
        :returns: trained classifier
        '''
        if c_values is None:
            if use_sgd:
                c_values = np.logspace(-2, 0, 8, base=10)
            else:
                c_values = np.logspace(-15, -1, 20, base=10)

        colnames = ['C'] + [
            'split{:d}_test_score'.format(i) for i in range(num_folds)
        ] + ['mean_test_score'
             ] + ['split{:d}_train_score'.format(i)
                  for i in range(num_folds)] + ['mean_train_score']
        cv_results = pd.DataFrame(np.nan,
                                  index=range(0, len(c_values)),
                                  columns=colnames)
        sum_train_auc = np.zeros(len(c_values))
        sum_test_auc = np.zeros(len(c_values))
        best_c = None

        if G is not None:
            cv_split = GroupKFold(n_splits=num_folds).split(X, Y, G)
        else:
            cv_split = KFold(n_splits=num_folds).split(X)

        for g, (inner_train_idxs, inner_test_idxs) in enumerate(cv_split):
            X_inner_train, X_inner_test = X[inner_train_idxs], X[
                inner_test_idxs]
            Y_inner_train, Y_inner_test = Y[inner_train_idxs], Y[
                inner_test_idxs]

            for i, c in enumerate(c_values):
                # print('  testing c={:.2E}'.format(c))
                cv_results.loc[i, 'C'] = c

                if use_sgd:
                    svc, lr = self.train_linear_svc_with_sgd(X_inner_train, Y_inner_train,c)
                else:
                    svc, lr = self.train_linear_svc(X_inner_train, Y_inner_train, c)

                scores = svc.decision_function(X_inner_train).reshape(-1, 1)

                # np.savetxt('coef/coef_{:.2E}.csv'.format(c), svc.coef_)

                # get training set results
                proba = lr.predict_proba(scores)
                pred = lr.predict(scores)
                fold_auc = roc_auc_score(Y_inner_train, proba[:, 1])
                precision, recall, f1, support = precision_recall_fscore_support(
                    Y_inner_train, pred)

                # print('  Training results')
                # print('{: >4}'.format('auc: {:f}'.format(fold_auc)))

                cv_results.loc[i, 'split{:d}_train_score'.format(g)] = fold_auc
                sum_train_auc[i] += fold_auc

                # now test
                proba = lr.predict_proba(
                    svc.decision_function(X_inner_test).reshape(-1, 1))
                pred = lr.predict(
                    svc.decision_function(X_inner_test).reshape(-1, 1))
                fold_auc = roc_auc_score(Y_inner_test, proba[:, 1])
                precision, recall, f1, support = precision_recall_fscore_support(
                    Y_inner_test, pred)

                # print('  Test results')
                # print('{: >4}'.format('auc: {:f}'.format(fold_auc)))

                cv_results.loc[i, 'split{:d}_test_score'.format(g)] = fold_auc
                sum_test_auc[i] += fold_auc

            X_inner_train = None
            X_inner_test = None
            Y_inner_train = None
            Y_inner_test = None

        cv_results.loc[:, 'mean_train_score'] = sum_train_auc / num_folds
        cv_results.loc[:, 'mean_test_score'] = sum_test_auc / num_folds

        best_cv_run = cv_results.loc[cv_results['mean_test_score'].idxmax()]

        self.best_c = best_cv_run['C']
        self.scaled_c = self.best_c * ((num_folds - 1) / num_folds)
        # print('  best C: {:f}, scaled C: {:f}'.format(self.best_c, self.scaled_c))

        if use_sgd:
            svc, clf = self.train_linear_svc_with_sgd(X, Y, self.best_c)
        else:    
            svc, clf = self.train_linear_svc(X, Y, self.scaled_c)

        return svc, clf, cv_results

    def train_svm_and_lr(self, X, Y, G=None, svm_c=None):
        ''' Trains the second stage of the FVs algo, the SVM.
        '''

        # print('  X.shape: {:s}'.format(str(X.shape)))
        # print('  Y.shape: {:s}'.format(str(Y.shape)))
        # print('num_bags: {:d} ({:d} positive)'.format(len(Y), int(np.sum(Y))))

        if svm_c is not None and type(svm_c) == float:
            self.svm, self.lr = self.train_linear_svc_with_sgd(X, Y, svm_c)
        else:
            self.svm, self.lr, self.cv_results = self.train_cv_linear_svc(X, Y, G, c_values=svm_c)

    def predict_proba_from_fvs(self, X):
        return self.lr.predict_proba(
            self.svm.decision_function(X).reshape(-1, 1))
            # self.svm.decision_function(self.ss2.transform(X)).reshape(-1, 1))

    def predict_from_fvs(self, X):
        return self.lr.predict(
            self.svm.decision_function(self.ss2.transform(X)).reshape(-1, 1))
    
    # INTERFACE FROM DATA
    def train(self, X, Y, G=None, svm_c=None, fvs_path=None):
        ''' Trains a FV model using CV and the supplied features X and labels Y
        :param X: input list of bags, each bag of shape (num_trajectories, num_features)
        '''
        X = self.train_fv_gmm_and_compute_fvs(X, fvs_path)
        assert len(X) == len(Y)

        self.train_svm_and_lr(X, Y, G, svm_c)

    def get_fv_map(self):
        dmaps = list()
        offset = 0
        for g, g_models in enumerate(self.models):
            dmap = dict()
            for desc_name, models in g_models.items():
                sz = 2 * models['pca'].n_components_ * models['fv_gmm'].n_kernels#self.num_gmm_components
                dmap[desc_name] = range(offset, offset + sz)
                offset += sz
            dmaps.append(dmap)
        return MotherDescriptorMap(dmaps)

    # INTERFACE FROM DATASET
    def compute_batch(self, batch_num, dataset, output, idxs, fvmap):

        bags = [dataset[i] for i in idxs] # loads examples in memory
        batch_fvs = np.empty((len(idxs), len(fvmap)))

        # for each group
        for g in range(dataset.num_groups):
            # for each descriptor
            for desc_name, desc_idxs in dataset.fmap:
                # put features together
                X = list()
                for ex in bags:
                    b = ex.get_group_fea(g)[:, desc_idxs]
                    if b.size != 0:
                        b = self.models[g][desc_name]['pca'].transform(self.models[g][desc_name]['ss'].transform(b))
                    else:
                        b = np.zeros((0, self.models[g][desc_name]['pca'].n_components_))
                    X.append(b)

                cols = fvmap[g][desc_name]
                fvs = self.models[g][desc_name]['fv_gmm'].predict(X, normalized=True)
                fvs = fvs.reshape(((fvs.shape[0], fvs.shape[1] * fvs.shape[2])))

                batch_fvs[:, cols] = fvs
        
        output[idxs,:] = batch_fvs

    def compute_fvs_from_dataset(self, dataset):
        assert self.models is not None

        batch_size = 50 # do batch_size fvs at a time
        examples = range(0, len(dataset))
        batches = [np.array(examples[i:i + batch_size]) for i in range(0, len(examples), batch_size)]
        fvmap = self.get_fv_map()
        nmap_path = os.path.join(self.tmp_path, 'fvs.dat')
        output = np.memmap(nmap_path, dtype='float64', mode='w+', shape=(len(dataset),len(fvmap)))
        
        with tqdm_joblib(tqdm(desc="Computing FVs", total=len(batches))) as progress_bar:
            joblib.Parallel(n_jobs=4)(joblib.delayed(self.compute_batch)(i, dataset, output, batch, fvmap) for i, batch in enumerate(batches))

        # X = np.empty(output.shape)
        # X[:] = output[:]
        # del output
        return np.memmap(nmap_path, dtype='float64', mode='r', shape=(len(dataset),len(fvmap))) # return read-only memmap

    def train_fv_gmm_and_compute_fvs_from_dataset(self, dataset, load_models=False):
        if load_models and dataset.is_pickled('models'):
            print('  loading ss1, pca and gmm.')
            self.models = dataset.get_pickled('models')
        else:
            # sample for GMM training
            # sample_rate = 1.5 * self.num_gmm_samples / (len(dataset) * 800)
            if dataset.hastxt_intermediate('samples'):
                X_s = dataset.loadtxt_intermediate('samples')
            else:
                X_s = dataset.sample(60)
            X_s = [X[np.random.choice(len(X), min(len(X), self.num_gmm_samples), replace=False), :] for X in X_s]

            self.train_all_fv_gmms(X_s, dataset.fmap)
            X_s = None
            dataset.pickle_intermediate('models', self.models)

        print(self.get_fv_map())
        X = self.compute_fvs_from_dataset(dataset)

        return X

    def fit_from_fvs(self, X, Y, svm_c=None):
        # Normalize FVs
        # self.ss2 = StandardScaler().fit(X)
        # X = self.ss2.transform(X)

        self.train_svm_and_lr(X, Y, None, svm_c)


    def fit_from_dataset(self, dataset, svm_c=None, store=True):
        # get the fvs
        X = dataset.loadtxt_intermediate('fvs')
        if X is None:
            X = self.train_fv_gmm_and_compute_fvs_from_dataset(dataset, store)

        # get the labels
        Y = dataset.get_labels()

        self.fit_from_fvs(X, Y, svm_c)

    def predict_proba_from_dataset(self, dataset, save_name='fvs'):
        X = dataset.loadtxt_intermediate(save_name)
        if X is None:
            print('  calculating FVs..')
            X = self.compute_fvs_from_dataset(dataset)
            print('  done!')

        dataset.savetxt_intermediate(save_name, X)

        return self.predict_proba_from_fvs(X)

    # GENERAL USER INTERFACE
    def fit(self, X, Y=None, G=None, store=True, **kwargs):
        # store = True will store the intermediate FVs
        if type(X) == torch.utils.data.Dataset and Y is None and G is None:
            self.fit_from_dataset(X, **kwargs)
        elif type(X) == list and type(Y) == list or type(Y) == np.array:
            self.fit_from_data(X, Y, G, **kwargs)
        else:
            raise 'not implemented'

    def predict_proba(self, X):
        print('  calculating FVs..')
        X = self.compute_fvs(X)
        print('  done!')

        return self.predict_proba_from_fvs(X)

    # SAVING / LOADING the model
    def save_models(self, models_path):
        pickle.dump(self.models, open(models_path, 'wb'))

    def load_models(self, models_path):
        self.models = pickle.load(open(models_path, 'rb'))

def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the MILES classifier.')

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
    main(args)
