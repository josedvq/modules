from functools import wraps
from time import time
from datetime import timedelta

def summarize_kfold(summary, results, index):
    ''' Takes a set of results and produces a summary
    '''
    for key in summary.keys().tolist():
        if key in results.columns:
            summary.loc[index, key] = results[key].mean()

def summarize_experiment(results):
    '''
    Get an experiment summary from per-fold results

    :param results: pandas dataframe with per-fold results
    :returns: dict with experiment summary
    '''
    summary = dict()
    columns_to_average = ['C', 'train_num_bags', 'train_pos_bags', 'train_auc', 'train_precision', 'train_recall',
        'train_f1', 'test_num_bags', 'test_pos_bags', 'test_auc', 'test_precision', 'test_recall', 'test_f1']
    for column_name in columns_to_average:
        if column_name in results:
            column_mean = results[column_name].mean()
            summary[column_name] = column_mean
        
    return summary

def dict_to_df(summary, results, index, add_prefix=None):
    ''' Takes a set of results and produces a summary
    '''
    for key in summary.keys():
        if key in results.columns:
            summary.loc[index, key] = results[key]

def add_to_dict(target, source, prefix=None):
    ''' Takes a set of results and produces a summary
    '''
    if prefix is not None:
        for key in source.keys():
            target['{:s}_{:s}'.format(prefix,key)] = source[key]
    else:
        for key in source.keys():
            target[key] = source[key]

def print_binary_classification_results(results):
    print(results)
    # print('{: >4}'.format('positive ratio: {:f}'.format(
    #     np.sum(Y) / len(Y))))
    # print('{: >4}'.format('auc: {:f}'.format(fold_auc)))
    # print('{: >4}'.format('precision: {:f}'.format(precision[1])))
    # print('{: >4}'.format('recall:{:f}'.format(recall[1])))
    # print('{: >4}'.format('f1:{:f}'.format(f1[1])))
    # print('{: >4}'.format('supp:{:f}'.format(support[1])))

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:{:s} took: {:s}'.format(f.__name__, str(timedelta(seconds=te-ts))))
        return result
    return wrap

def get_timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        return result, timedelta(seconds=te-ts)
    return wrap
