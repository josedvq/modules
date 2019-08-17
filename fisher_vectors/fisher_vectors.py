''' Fisher Vectors implementation
'''
import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

from fisher_vectors.improved_fisher_var import FisherVectorGMM

def extern_compute_fvs_from_bags(ss1, pca, fv_gmm, bags):
    assert len(bags) > 0
    assert ss1 is not None
    assert pca is not None
    assert fv_gmm is not None

    print('  bags: {:d}'.format(len(bags)))

    # transform the features
    # bags = [pca.transform(ss1.transform(b)) for b in bags]
    for i, b in enumerate(bags):
        if b.size != 0:
            bags[i] = pca.transform(ss1.transform(b))

    fvs = fv_gmm.predict(bags, normalized=True)
    fvs = fvs.reshape(((fvs.shape[0], fvs.shape[1] * fvs.shape[2])))

    return fvs

def extern_compute_fvs_from_dataset(ss1, pca, fv_gmm, dataset):

    bags = dataset.get_full_dataset()
    fvs = extern_compute_fvs_from_bags(ss1, pca, fv_gmm, bags)

    return fvs

class FisherVectors:
    ''' Fisher Vector's implementation.
    '''

    def __init__(self,
                 pca_components=0.95,
                 num_gmm_samples=100000,
                 num_gmm_components=256):
        self.pca_components = pca_components
        self.num_gmm_samples = num_gmm_samples
        self.num_gmm_components = num_gmm_components

        self.ss1 = None
        self.pca = None
        self.fv_gmm = None

        self.ss2 = None
        self.lr = None
        self.svm = None

        self.cv_results = None

    def train_linear_svc(self, X, Y, C):
        svc = LinearSVC(loss='hinge', penalty='l2', max_iter=5000,
                        C=C).fit(X, Y)

        # Platt calibration on the train set
        scores = svc.decision_function(X).reshape(-1, 1)
        lr = LogisticRegression(C=1000000000,
                                solver='liblinear').fit(scores, Y)

        return svc, lr

    def train_cv_linear_svc(self, X, Y, G=None, num_folds=4, c_values=None):
        '''
        Trains a Linear SVC using Platt scaling and optimizing for AUC using cross-validation.
        :param X: input data with shape (num_elements, num_features)
        :param Y: labels with shape (num_elements)
        :param G: group assignment for the elements
        :returns: trained classifier
        '''
        if c_values is None:
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
                print('  testing c={:.2E}'.format(c))
                cv_results.loc[i, 'C'] = c

                svc, lr = self.train_linear_svc(X_inner_train, Y_inner_train,
                                                c)
                scores = svc.decision_function(X_inner_train).reshape(-1, 1)

                # np.savetxt('coef/coef_{:.2E}.csv'.format(c), svc.coef_)

                # get training set results
                proba = lr.predict_proba(scores)
                pred = lr.predict(scores)
                fold_auc = roc_auc_score(Y_inner_train, proba[:, 1])
                precision, recall, f1, support = precision_recall_fscore_support(
                    Y_inner_train, pred)

                print('  Training results')
                print('{: >4}'.format('auc: {:f}'.format(fold_auc)))

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

                print('  Test results')
                print('{: >4}'.format('auc: {:f}'.format(fold_auc)))

                cv_results.loc[i, 'split{:d}_test_score'.format(g)] = fold_auc
                sum_test_auc[i] += fold_auc

        cv_results.loc[:, 'mean_train_score'] = sum_train_auc / num_folds
        cv_results.loc[:, 'mean_test_score'] = sum_test_auc / num_folds

        best_cv_run = cv_results.loc[cv_results['mean_test_score'].idxmax()]
        print('best CV run:')
        print(best_cv_run)

        best_c = best_cv_run['C']
        scaled_c = best_c * ((num_folds - 1) / num_folds)
        print('  best C: {:f}, scaled C: {:f}'.format(best_c, scaled_c))
        svc = LinearSVC(loss='hinge', penalty='l2', max_iter=5000,
                        C=scaled_c).fit(X, Y)

        scores = svc.decision_function(X).reshape(-1, 1)
        clf = LogisticRegression(C=1000000000,
                                 solver='liblinear').fit(scores, Y)
        print('  done!')

        return svc, clf, cv_results

    def sample_fast(self, file_paths, rate, num_samples, feature_idx):

        print('  sampling trajectories')
        print('  rate: {:f}'.format(rate))
        arrs = [np.empty((10000, len(feature_idx)))]
        cnt = 0

        for f in file_paths:
            print('  {:s}'.format(f))
            for line in open(f):
                if np.random.sample() < rate:
                    if cnt == 10000:
                        print('  buffer filled')
                        arrs.append(np.empty((10000, len(feature_idx))))
                        cnt = 0
                    arrs[-1][cnt, :] = (np.fromstring(line,
                                                      sep=','))[feature_idx]
                    cnt += 1

        arrs[-1] = arrs[-1][:cnt, :]
        D = np.vstack(arrs)

        assert len(D) >= num_samples
        D = D[np.random.choice(len(D), num_samples, replace=False), :]
        return D

    def sample_datasets_fast(self, datasets, rate, num_samples):
        print('  sampling dataset')
        print('  rate: {:f}'.format(rate))

        samples = list()
        for i, d in enumerate(datasets):
            ds_samples = d.sample_lines_fast(rate)
            print('  sampled {:d} lines from {:s}'.format(len(ds_samples), os.path.basename(d.dt_path)))
            samples.append(ds_samples)

        D = np.vstack(samples)

        assert len(D) >= num_samples
        D = D[np.random.choice(len(D), num_samples, replace=False), :]
        return D

    def compute_fvs_from_bags(self, bags):
        assert len(bags) > 0
        assert self.ss1 is not None
        assert self.pca is not None
        assert self.fv_gmm is not None

        print('  bags: {:d}'.format(len(bags)))

        # transform the features
        bags = [self.pca.transform(self.ss1.transform(b)) for b in bags]
        fvs = self.fv_gmm.predict(bags, normalized=True)
        fvs = fvs.reshape(((fvs.shape[0], fvs.shape[1] * fvs.shape[2])))

        return fvs

    def compute_fvs_from_file(self,
                              file_path,
                              bag_idx,
                              feature_idx,
                              group_id=None):
        assert os.path.isfile(file_path)
        print('  {:s}'.format(file_path))

        bags = np.loadtxt(file_path, delimiter=',')
        # col_idx = np.r_[2,18:58]
        # arr = arr[:,col_idx]
        last_elem_idx = np.where(np.diff(bags[:, bag_idx]))[0]
        bags = np.split(bags[:, feature_idx], last_elem_idx + 1)
        groups = None
        if group_id is not None:
            groups = np.full(len(bags), group_id)

        print('  bags: {:d}'.format(len(bags)))

        # transform the features
        fvs = self.compute_fvs_from_bags(bags)

        return fvs, groups

    def compute_fvs_from_files(self, file_paths, bag_idx, feature_idx):
        # get the len of feature_idx
        assert len(file_paths) > 0

        res = joblib.Parallel(n_jobs=10, verbose=10)(
            joblib.delayed(self.compute_fvs_from_file)(
                file_paths[j], bag_idx, feature_idx, group_id=j)
            for j in range(0, len(file_paths)))
        all_fvs, all_groups = zip(*res)

        return np.vstack(all_fvs), np.concatenate(all_groups)

    def compute_fvs_from_dataset(self, dataset):

        bags = dataset.get_full_dataset()

        fvs = self.compute_fvs_from_bags(bags)

        return fvs

    def compute_fvs_from_datasets(self, datasets):

        all_fvs = joblib.Parallel(n_jobs=10, verbose=10)(
            joblib.delayed(extern_compute_fvs_from_dataset)(self.ss1, self.pca, self.fv_gmm, ds) for ds in datasets)

        return all_fvs

    def train_fv_gmm(self, X):
        ''' Trains the first stage of the FVs algo, up to the GMM
        '''
        # need to calculate FVs
        print('  sampling training set for GMM fit set')

        print('  normalizing features')

        self.ss1 = StandardScaler().fit(X)
        X = self.ss1.transform(X)
        print('  calculating PCA')
        self.pca = PCA(n_components=self.pca_components, whiten=True).fit(X)
        num_final_features = self.pca.n_components_
        X = self.pca.transform(X)
        print('  components kept: {:d}, explained variance = {:f}'.format(
            num_final_features, np.sum(self.pca.explained_variance_ratio_)))

        print('  fitting GMM')
        self.fv_gmm = FisherVectorGMM(n_kernels=self.num_gmm_components).fit(
            X=X, verbose=True)
        print('  trained FV GMM')

    def train_fv_gmm_from_datasets(self, datasets, sample_rate=0.01):
        X_gmm = self.sample_datasets_fast(datasets, sample_rate, self.num_gmm_samples)
        self.train_fv_gmm(X_gmm)

    def train_fv_gmm_and_compute_fvs(self, X, fvs_path):
        X_gmm = np.vstack(X)
        X_gmm = X_gmm[np.random.choice(
            len(X_gmm), size=self.num_gmm_samples, replace=False), :]
        self.train_fv_gmm(X_gmm)

        X = self.compute_fvs_from_bags(X)
        print('  done!')

        if fvs_path:
            joblib.dump(X, fvs_path)

        return X

    def train_fv_gmm_and_compute_fvs_from_files(self, file_paths, bag_idx, feature_idx, fvs_path=None, sample_rate=0.01):
        X_gmm = self.sample_fast(file_paths, sample_rate, self.num_gmm_samples, feature_idx)
        self.train_fv_gmm(X_gmm)

        X, G = self.compute_fvs_from_files(file_paths, bag_idx, feature_idx)

        if fvs_path:
            joblib.dump(X, fvs_path)

        return X, G

    def train_svm_and_lr(self, X, Y, G=None, svm_c=None):
        ''' Trains the second stage of the FVs algo, the SVM.
        '''
        print(X)
        print(Y)
        self.ss2 = StandardScaler().fit(X)
        X = self.ss2.transform(X)

        print('  X.shape: {:s}'.format(str(X.shape)))
        print('  Y.shape: {:s}'.format(str(Y.shape)))
        print(X)
        print(Y)
        print('num_bags: {:d} ({:d} positive)'.format(len(Y), int(np.sum(Y))))

        if svm_c is not None and type(svm_c) == float:
            self.svm, self.lr = self.train_linear_svc(X, Y, svm_c)
        else:
            self.svm, self.lr, self.cv_results = self.train_cv_linear_svc(X, Y, G, c_values=svm_c)

    def train(self, X, Y, G=None, svm_c=None, fvs_path=None):
        ''' Trains a FV model using CV and the supplied features X and labels Y
        :param X: input list of bags, each bag of shape (num_trajectories, num_features)
        '''
        X = self.train_fv_gmm_and_compute_fvs(X, fvs_path)
        assert len(X) == len(Y)

        self.train_svm_and_lr(X, Y, G, svm_c)

    def train_from_files(self,
                         file_paths,
                         Y,
                         bag_idx,
                         feature_idx,
                         svm_c=None,
                         fvs_path=None,
                         sample_rate=0.01):
        ''' Trains the FV classifier from files with features.
        :param file_paths: list of filenames, each files is taken as a different group
        '''
        X, G = self.train_fv_gmm_and_compute_fvs_from_files(file_paths, bag_idx, feature_idx, fvs_path, sample_rate)
        assert len(X) == len(Y)

        self.train_svm_and_lr(X, Y, G, svm_c)

    def predict_proba_from_fvs(self, X):
        return self.lr.predict_proba(
            self.svm.decision_function(self.ss2.transform(X)).reshape(-1, 1))

    def predict_from_fvs(self, X):
        return self.lr.predict(
            self.svm.decision_function(self.ss2.transform(X)).reshape(-1, 1))

    def predict_proba(self, X):
        print('  calculating FVs..')
        X = self.compute_fvs_from_bags(X)
        print('  done!')

        return self.predict_proba_from_fvs(X)

    def save(self, exp_path):
        joblib.dump(self.ss1, os.path.join(exp_path, 'ss1.pkl'))
        joblib.dump(self.pca, os.path.join(exp_path, 'pca.pkl'))
        joblib.dump(self.fv_gmm, os.path.join(exp_path, 'gmm.pkl'))

        joblib.dump(self.ss2, os.path.join(exp_path, 'ss2.pkl'))
        joblib.dump(self.lr, os.path.join(exp_path, 'lr.pkl'))
        joblib.dump(self.svm, os.path.join(exp_path, 'svm.pkl'))

    def load(self, exp_path):
        self.ss1 = joblib.load(os.path.join(exp_path, 'ss1.pkl'))
        self.pca = joblib.load(os.path.join(exp_path, 'pca.pkl'))
        self.fv_gmm = joblib.load(os.path.join(exp_path, 'gmm.pkl'))

        self.ss2 = joblib.load(os.path.join(exp_path, 'ss2.pkl'))
        self.lr = joblib.load(os.path.join(exp_path, 'lr.pkl'))
        self.svm = joblib.load(os.path.join(exp_path, 'svm.pkl'))


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
