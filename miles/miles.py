import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise as sk_pairwise
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise

class ContinueI(Exception):
    pass

class Miles:
    ''' Multiple Instance Learning via embedded instance selection classifier. '''
    def __init__(self, kernel_fn=pairwise.rbf_kernel, kernel_param=None, C=None, normalize=True):
        self.kernel_fn = kernel_fn
        self.kernel_param = kernel_param
        self.C = C
        self.normalize = normalize
        self.clf = None
        self.P = None
        self.num_bags = None
        self.num_instances_per_bag = None
        self.num_features = None
        self.ss = None
        
        self.M = None

    # Trains the MILES classifier
    # A: dataset, with shape [num_bags, num_instances_per_bag, num_features]
    # y: bag labels, with shape [num_bags]
    # C: for logistic regression
    # kernel_fn: kernel fn following sklearn.metrics.pairwise
    # kernel_param : kernel parameter (default = 5)
    # output w: MILES classifier
    def train(self, A, Y, C=0.01, P=None, train_lr=False):
        self.num_bags, self.num_instances_per_bag, self.num_features = A.shape
        self.C = C

        if P is None:
            self.P = np.reshape(A,[-1,self.num_features])
            self.num_instances = self.num_bags * self.num_instances_per_bag
        else:
            self.P = P
            self.num_instances = P.shape[0]

        self.M = self.compute_m(A, normalize=False)

        if self.normalize:
            self.ss = StandardScaler().fit(self.M)
            self.M = self.ss.transform(self.M)

        self.clf = LogisticRegression(penalty='l1', C=self.C, max_iter=1000, solver='liblinear', verbose=2).fit(self.M,Y)

    def compute_m(self, A, normalize=None):
        if normalize is None:
            normalize = self.normalize

        num_bags, num_instances_per_bag, num_features = A.shape

        # Matrix used to train the classifier
        M = np.zeros((num_bags,self.num_instances))

        # calculate the kernel between every bag and self.P to fill M
        for b in range(0,num_bags):
            kernel_out = self.kernel_fn(A[b,:,:], self.P, self.kernel_param)
            M[b,:] = np.max(kernel_out, axis=0)

        if normalize:
            M = self.ss.transform(M)

        return M

    # re-train for a different label
    def train_lr(self,Y):
        assert self.M is not None

        self.clf = LogisticRegression(penalty='l1',C=self.C,max_iter=1000,solver='liblinear',verbose=2).fit(self.M,Y)        

    def get_train_proba(self):
        return self.clf.predict_proba(self.M)


    # Trains the MILES classifier
    # A: dataset, with shape [num_bags, num_instances_per_bag, num_features]
    # y: bag labels, with shape [num_bags]
    # P custom prototypes
    # C: for logistic regression
    # kernel_fn: kernel fn following sklearn.metrics.pairwise
    # kernel_param : kernel parameter (default = 5)
    # output w: MILES classifier
    def train_cv(self, A, Y, P=None, A_groups=None, P_groups=None, group_def=None, c_values=None, cv_log=None):
        if A_groups is not None:
            assert P is not None
            assert P_groups is not None
            assert group_def is not None
            self._train_cv_with_groups(A,Y,P,A_groups,P_groups,group_def,c_values,cv_log)
        else:
            assert P_groups is None
            self._train_cv_no_groups(A,Y,P,c_values,cv_log)


    def _train_cv_with_groups(self, A, Y, P, A_groups, P_groups, group_def, c_values=None, cv_log=None):
        self.num_bags, self.num_instances_per_bag, self.num_features = A.shape

        assert len(A) == len(Y)
        assert len(A) == len(A_groups)
        assert len(P) == len(P_groups)
        
        # prepare the prototypes
        self.P = P
        self.num_instances = P.shape[0]

        # Matrix used to train the classifier
        M = np.zeros((self.num_bags,self.num_instances))

        # calculate the kernel between every bag and self.P to fill M
        print('  caculating M')
        for b in range(0,self.num_bags):
            kernel_out = self.kernel_fn(A[b,:,:], self.P, self.kernel_param)
            M[b,:] = np.max(kernel_out, axis=0)

        # normalize M
        self.ss = StandardScaler().fit(M)
        if self.normalize:
            print('  normalizing M')
            M = self.ss.transform(M)

        # CV loop
        if c_values is None:
            c_values = np.logspace(-20, 20, 40,base=10)

        print('  doing group CV')
        n_folds = 4
        kf = KFold(n_splits=n_folds)
        assert np.all(np.isin(A_groups,group_def))
        assert np.all(np.isin(P_groups,group_def))

        colnames = ['C'] + ['split{:d}_test_score'.format(i) for i in range(n_folds)] + ['mean_test_score'] + [
        'split{:d}_train_score'.format(i) for i in range(n_folds)] + ['mean_train_score']
        results = pd.DataFrame(np.nan, index=range(
            0, len(c_values)), columns=colnames)
        sum_train_auc = np.zeros(len(c_values))
        sum_test_auc = np.zeros(len(c_values))
        max_mean_test_auc = 0
        best_c = None

        for f, (train_groups_idxs, test_groups_idxs) in enumerate(kf.split(group_def)):
            train_groups = group_def[train_groups_idxs]
            test_groups = group_def[test_groups_idxs]
            print(train_groups)
            print(test_groups)
            print(A_groups)

            # separate prototypes and training bags
            # get indices for fold
            prototype_idxs = np.full(len(P_groups),False)
            for idx in train_groups:
                prototype_idxs = np.logical_or(prototype_idxs, (P_groups == idx))

            train_idxs = np.full(len(A_groups), False)
            for idx in train_groups:
                train_idxs = np.logical_or(train_idxs, (A_groups == idx))

            test_idxs = np.logical_not(train_idxs)

            print('  num_prototypes: {:d}'.format(sum(prototype_idxs)))
            print('  num_train_bags: {:d}'.format(sum(train_idxs)))
            print('  num_test_bags: {:d}'.format(sum(test_idxs)))

            # split bags and prototypes
            M_train, M_test = M[train_idxs,:][:,prototype_idxs], M[test_idxs,:][:,prototype_idxs]
            Y_train, Y_test = Y[train_idxs], Y[test_idxs]

            # for each C
            for i, c_value in enumerate(c_values):

                print('Testing C={:f}'.format(c_value))
                results.loc[i, 'C'] = c_value

                lr = LogisticRegression(penalty='l1',C=c_value,max_iter=1000,solver='liblinear',verbose=2).fit(M_train,Y_train)

                # training set results
                proba = lr.predict_proba(M_train)#predict_proba(X_train)
                pred = np.argmax(proba, axis=1)

                fold_auc = roc_auc_score(Y_train, proba[:, 1])
                precision, recall, f1, support = precision_recall_fscore_support(
                    Y_train, pred)

                print('  Training results')
                print('{: >4}'.format('auc: {:f}'.format(fold_auc)))

                results.loc[i, 'split{:d}_train_score'.format(f)] = fold_auc
                sum_train_auc[i] += fold_auc
                
                proba = lr.predict_proba(M_test)
                pred = np.argmax(proba, axis=1)

                fold_auc = roc_auc_score(Y_test, proba[:, 1])
                precision, recall, f1, support = precision_recall_fscore_support(
                    Y_test, pred)

                print('  Test results')
                print('{: >4}'.format('auc: {:f}'.format(fold_auc)))

                results.loc[i, 'split{:d}_test_score'.format(f)] = fold_auc
                sum_test_auc[i] += fold_auc

        results.loc[:,'mean_train_score'] = sum_train_auc / n_folds
        results.loc[:,'mean_test_score'] = sum_test_auc / n_folds

        if cv_log is not None:
            results.to_csv(cv_log, sep=',')

        best_cv_run = results.loc[results['mean_test_score'].idxmax()]
        print(  'best CV run:')
        print(best_cv_run)

        best_c = best_cv_run['C']
        print('  best C: {:f}'.format(best_c))

        # train the classifier
        self.C = best_c
        self.clf = LogisticRegression(penalty='l1',C=self.C,max_iter=1000,solver='liblinear',verbose=2).fit(M,Y)


    def _train_cv_no_groups(self, A, Y, P=None, c_values=None, cv_log=None):
        self.num_bags, self.num_instances_per_bag, self.num_features = A.shape
        
        # flatten the data
        if P is None:
            self.P = np.reshape(A,[-1,self.num_features])
            self.num_instances = self.num_bags * self.num_instances_per_bag
        else:
            self.P = P
            self.num_instances = P.shape[0]

        # Matrix used to train the classifier
        M = np.zeros((self.num_bags,self.num_instances))

        # calculate the kernel between every bag and self.P to fill M
        for b in range(0,self.num_bags):
            kernel_out = self.kernel_fn(A[b,:,:], self.P, self.kernel_param)
            M[b,:] = np.max(kernel_out, axis=0)

        # normalize M
        self.ss = StandardScaler().fit(M)
        if self.normalize:
            M = self.ss.transform(M)

        if c_vect is None:
            c_vect = np.logspace(-20, 20, 40,base=10)
        lr = LogisticRegression(penalty='l1',max_iter=1000,solver='liblinear')

        self.clf = GridSearchCV(estimator=lr, param_grid=dict(C=c_vect),cv=10,scoring='roc_auc',n_jobs=1,return_train_score=True,verbose=2).fit(M,y)

        if cv_log is not None:
            scores_df = pd.DataFrame(self.clf.cv_results_)
            scores_df.to_csv(cv_log,sep=',')


    def predict_proba_from_m(self, M):
        return self.clf.predict_proba(M)

    # D: test dataset with shape [num_bags, num_instances_per_bag, num_features]
    # output: class probabilities
    def predict_proba(self, D):
        assert self.clf is not None, 'classifier not trained'
        assert self.P is not None, 'classifier not trained'

        num_bags, num_instances_per_bag, num_features = D.shape
        assert num_instances_per_bag == self.num_instances_per_bag
        assert num_features == self.num_features

        # first calculate the M
        M = self.compute_m(D)

        return self.clf.predict_proba(M)

    def predict(self, D):
        assert self.clf is not None, 'classifier not trained'
        assert self.P is not None, 'classifier not trained'

        num_bags, num_instances_per_bag, num_features = D.shape
        assert num_instances_per_bag == self.num_instances_per_bag
        assert num_features == self.num_features

        # first calculate the M
        M = self.compute_m(D)

        return self.clf.predict(M)

def main(args):

    b = np.loadtxt('/media/jose/EVO/miles/matlab_comp/bags/traj_00.csv', delimiter=',')
    b = StandardScaler().fit_transform(b)
    b = b.reshape((-1,20,b.shape[-1]))[:5,:,:]
    l = np.loadtxt('/media/jose/EVO/miles/matlab_comp/labels/traj_00.csv', delimiter=',')

    clf = Miles()
    clf.train_cv(b,l)

    # clf = Miles()

    # X = np.array([
    #   [
    #       [1,1,1],
    #       [2,2,2],
    #       [3,3,3]
    #   ],
    #   [
    #       [10,10,10],
    #       [11,11,11],
    #       [12,12,12]
    #   ],
    #   [
    #       [19,19,19],
    #       [20,20,20],
    #       [21,21,21]
    #   ]
    # ])

    # y = [0,1,0]

    # clf.train(X,y,lmbd=0.007,kernel_fn=pairwise.rbf_kernel)
    # print(clf.M)

    # # The shape of coef_ attribute should be: (# of classes, # of features)
    # print(clf.clf.coef_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests the MILES classifier.')

    parser.add_argument('-d', '--debug', help="Print lots of debugging statements",
                        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('-v', '--verbose', help="Be verbose",
                        action="store_const", dest="loglevel", const=logging.INFO)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    main(args)