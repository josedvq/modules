import os

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

from fisher_vectors.fisher_vectors import FisherVectors
from helpers.helper_functions import summarize_kfold, print_binary_classification_results, add_to_dict


def train_svm(fisher_vectors, X, Y, G, svm_c=None, cv_results_path=None):
    ''' Trains the FV model on a set of data
    '''
    results = dict()

    print('  starting cross-validation')
    fisher_vectors.train_svm_and_lr(X, Y, G, svm_c=svm_c)

    cv_results = None
    if svm_c is None or type(svm_c) == np.ndarray:
        assert fisher_vectors.cv_results is not None
        cv_results = fisher_vectors.cv_results
        if cv_results_path:
            cv_results.to_csv(cv_results_path, sep=',')

    print('  classifying training set')
    pred = fisher_vectors.predict_from_fvs(X)
    proba = fisher_vectors.predict_proba_from_fvs(X)

    fold_auc = results['auc'] = roc_auc_score(Y, proba[:, 1])
    precision, recall, f1, support = precision_recall_fscore_support(Y, pred)
    results['precision'], results['recall'], results['f1'], results[
        'support'] = (precision[1], recall[1], f1[1], support[1])

    print('  Training results')
    print_binary_classification_results(results)
    return results, proba[:, 1], cv_results


def test_svm(fisher_vectors, X, Y):
    ''' Tests the FV model on a set of data

    '''
    results = dict()

    print('  classifying test set')
    pred = fisher_vectors.predict_from_fvs(X)
    proba = fisher_vectors.predict_proba_from_fvs(X)

    fold_auc = results['auc'] = roc_auc_score(Y, proba[:, 1])
    precision, recall, f1, support = precision_recall_fscore_support(Y, pred)
    results['precision'], results['recall'], results['f1'], results[
        'support'] = (precision[1], recall[1], f1[1], support[1])

    print('  Test results')
    print_binary_classification_results(results)

    return results, proba[:, 1]


def has_saved_fvs(exp_path, num_folds):
    for f in range(0,num_folds):
        fold_path = os.path.join(exp_path, 'fold{:02d}'.format(f))
        if not os.path.exists(fold_path):
            return False
        train_fvs_path = os.path.join(fold_path, 'train_fvs.pkl')
        test_fvs_path = os.path.join(fold_path, 'test_fvs.pkl')

        if not os.path.isfile(train_fvs_path) or not os.path.isfile(test_fvs_path):
            return False
    return True

def train_grouped(fvs, train_datasets, train_labels, exp_path, svm_c=None, random_state=None, sample_rate=0.01):
    '''

    '''
    train_fvs_path = os.path.join(exp_path, 'train_fvs.pkl')
    if os.path.isfile(train_fvs_path):
        X_train = joblib.load(train_fvs_path)
        # check the lengths
        for X, Y in zip(X_train, train_labels):
            assert len(X) == len(Y), '{:d} != {:d}'.format(len(X), len(Y))
    else:
        assert train_datasets is not None
        fvs.train_fv_gmm_from_datasets(train_datasets, sample_rate=sample_rate)
        joblib.dump(fvs, os.path.join(exp_path, 'fv.pkl'))
        X_train = fvs.compute_fvs_from_datasets(train_datasets)
        joblib.dump(X_train, train_fvs_path)

    X_train = np.vstack(X_train)
    Y_train = np.concatenate(train_labels)
    G_train = np.concatenate([np.full(len(train_labels[i]), i) for i in range(0, len(train_labels))])

    return train_svm(fvs, X_train, Y_train, G_train, svm_c=svm_c)
        

def test_grouped(fvs, test_datasets, test_labels, exp_path):
    '''

    '''
    test_fvs_path = os.path.join(exp_path, 'test_fvs.pkl')

    if os.path.isfile(test_fvs_path):
        X_test = joblib.load(test_fvs_path)
        # check the lengths
        for X, Y in zip(X_test, test_labels):
            assert len(X) == len(Y)
    else:
        X_test = fvs.compute_fvs_from_datasets(test_datasets)
        joblib.dump(X_test, test_fvs_path)

    X_test = np.vstack(X_test)
    Y_test = np.concatenate(test_labels)

    return test_svm(fvs, X_test, Y_test)

def cross_test_grouped(
               datasets,
               labels,
               exp_path,
               pca_components=None,
               num_gmm_samples=None,
               num_gmm_components=None,
               svm_c=None,
               num_folds=4,
               random_state=None,
               sample_rate=0.01):

    cv_split = KFold(n_splits=num_folds,
                         random_state=random_state,
                         shuffle=True).split(datasets)
    scores = list()
    results = list()

    for f, (train_idxs, test_idxs) in enumerate(cv_split):
        print('{:*^10}'.format(' FOLD {:d} '.format(f)))
        fold_results = {'fold': f, 'first_test_elem': test_idxs[0]}

        fold_path = os.path.join(exp_path, 'fold{:02d}'.format(f))

        if not os.path.exists(fold_path):
            os.mkdir(fold_path)

        train_datasets = [datasets[i] for i in train_idxs]
        train_labels = [labels[i] for i in train_idxs]
        test_datasets = [datasets[i] for i in test_idxs]
        test_labels = [labels[i] for i in test_idxs]

        model_path = os.path.join(exp_path, 'fvs.pkl')
        if os.path.isfile(model_path):
            fvs = joblib.load(model_path)
        else:
            fvs = FisherVectors(pca_components=0.99,
                                num_gmm_samples=100000,
                                num_gmm_components=256)

        # train
        train_results, train_scores, fold_cv_results = train_grouped(fvs, train_datasets, train_labels, fold_path, svm_c, random_state, sample_rate)
        add_to_dict(fold_results, train_results, prefix='train')

        # save the model
        joblib.dump(fvs, model_path)

        # test
        test_results, test_scores = test_grouped(fvs, test_datasets, test_labels, fold_path)
        add_to_dict(fold_results, test_results, prefix='test')

        fold_results['C'] = fvs.svm.get_params()['C']
        results.append(fold_results)

    return pd.DataFrame(results), scores


def cross_test(X,
               Y,
               exp_path,
               grouped=False,
               pca_components=None,
               num_gmm_samples=None,
               num_gmm_components=None,
               svm_c=None,
               num_folds=4,
               random_state=None):
    '''
    :param X: list of bags of features, or list of lists of bags
    :param Y: array with label of each bag, or list of arrays of labels
    :param exp_path: folder where to place intermediate files
    '''

    if grouped:
        cv_split = KFold(n_splits=num_folds,
                         random_state=random_state,
                         shuffle=True).split(X)
        scores = list()
        # TODO: update
    else:
        cv_split = StratifiedKFold(n_splits=num_folds,
                                   random_state=random_state,
                                   shuffle=True).split(Y, Y)
        scores = np.empty(len(Y))

    results = list()
    cv_results = list()

    for f, (train_idxs, test_idxs) in enumerate(cv_split):
        print('{:*^10}'.format(' FOLD {:d} '.format(f)))
        fold_results = {'fold': f, 'first_test_elem': test_idxs[0]}

        fold_path = os.path.join(exp_path, 'fold{:02d}'.format(f))

        if not os.path.exists(fold_path):
            os.mkdir(fold_path)

        train_fvs_path = os.path.join(fold_path, 'train_fvs.pkl')
        test_fvs_path = os.path.join(fold_path, 'test_fvs.pkl')

        # fvs already computed
        X_train = X_test = None
        G_train = G_test = None
        if grouped:
            Y_train = np.concatenate([Y[i] for i in train_idxs])
            Y_test = np.concatenate([Y[i] for i in test_idxs])
            G_train = np.concatenate(
                [np.full(len(Y[i]), i) for i in train_idxs])
            G_test = np.concatenate([np.full(len(Y[i]), i) for i in test_idxs])
        else:
            Y_train = Y[train_idxs]
            Y_test = Y[test_idxs]

        # FVs model
        fvs = FisherVectors(pca_components=0.99,
                            num_gmm_samples=100000,
                            num_gmm_components=256)

        if os.path.isfile(train_fvs_path) and os.path.isfile(test_fvs_path):
            X_train = joblib.load(train_fvs_path)
            X_test = joblib.load(test_fvs_path)
            assert len(X_train) == len(Y_train)
            assert len(X_test) == len(Y_test)
        else:
            assert X is not None
            if grouped:
                X_train = [bag for i in train_idxs for bag in X[i]]
                X_test = [bag for i in test_idxs for bag in X[i]]
            else:
                X_train = [X[i] for i in train_idxs]
                X_test = [X[i] for i in test_idxs]

            fvs.train_fv_gmm_and_compute_fvs(X_train, fvs_path=train_fvs_path)
            X_train = joblib.load(train_fvs_path)
            X_test = fvs.compute_fvs_from_bags(X_test)
            joblib.dump(X_test, test_fvs_path)

        train_results, train_scores, fold_cv_results = train_svm(fvs,
                                            X_train,
                                            Y_train,
                                            G_train,
                                            svm_c=svm_c)

        joblib.dump(fvs.svm.coef_, os.path.join(fold_path, 'coef.pkl'))

        add_to_dict(fold_results, train_results, prefix='train')

        test_results, test_scores = test_svm(fvs, X_test, Y_test)
        add_to_dict(fold_results, test_results, prefix='test')

        results.append(fold_results)
        cv_results.append(fold_cv_results)

        if grouped:
            pass
            #raise 'not implemented'
        else:
            scores[test_idxs] = test_scores

    return pd.DataFrame(results), scores, cv_results
