import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def train_test(train_X, train_Y, test_X, test_Y, model):
    ss = StandardScaler().fit(train_X)
    train_X = ss.transform(train_X)

    ## Find the best regularisation parameter with respect to AUC  and fit on the data
    # args: cv = num_folds
    # cross-entropy loss, L2 regularization
    
    model.fit(train_X,train_Y)
    
    # Print the performance on the training set
    # pass the probability of positive class and label
    proba = model.predict_proba(train_X)
    fold_auc = roc_auc_score(train_Y,proba[:,1])
    print('  Training results')
    print('{: >4}'.format('auc: {:f}'.format(fold_auc)))
    
    proba = model.predict_proba(ss.transform(test_X))
    fold_auc = roc_auc_score(test_Y,proba[:,1])
    print('  Test results')
    print('{: >4}'.format('auc: {:f}'.format(fold_auc)))
    return proba

def dataset_train_test(train_ds, test_ds, model):
    train_data = [train_ds[i] for i in range(len(train_ds))]
    test_data = [test_ds[i] for i in range(len(test_ds))]

    train_X, train_Y = zip(*train_data)
    test_X, test_Y = zip(*test_data)

    train_X = np.vstack(train_X)
    test_X = np.vstack(test_X)

    return train_test(train_X, train_Y, test_X, test_Y, model)
