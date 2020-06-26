import numpy as np
from sklearn.mixture import GaussianMixture
import pickle, os

N_Kernel_Choices = [5, 20, 60, 100, 200, 500]

class FisherVectorGMM:
  def __init__(self, n_kernels=1, covariance_type='diag'):
    assert covariance_type in ['diag', 'full']
    assert n_kernels > 0

    self.n_kernels = n_kernels
    self.covariance_type = covariance_type
    self.fitted = False

  def score(self, X):
    return self.gmm.bic(X.reshape(-1, X.shape[-1]))

  def fit(self, X, model_dump_path=None, verbose=True):
    """
    :param X: 2 dimensions (n_descriptors, n_dim_descriptor)
    :param model_dump_path: (optional) path where the fitted model shall be dumped
    :param verbose - boolean that controls the verbosity
    :return: fitted Fisher vector object
    """

    return self._fit(X, model_dump_path=model_dump_path, verbose=verbose)

  def fit_by_bic(self, X, choices_n_kernels=N_Kernel_Choices, model_dump_path=None, verbose=True):
    """
    Fits the GMM with various n_kernels and selects the model with the lowest BIC
    :param X: 2 dimensions (n_descriptors, n_dim_descriptor)
    :param choices_n_kernels: array of positive integers that specify with how many kernels the GMM shall be trained
                              default: [20, 60, 100, 200, 500]
    :param model_dump_path: (optional) path where the fitted model shall be dumped
    :param verbose - boolean that controls the verbosity
    :return: fitted Fisher vector object
    """

    return self._fit_by_bic(X, choices_n_kernels=choices_n_kernels, model_dump_path=model_dump_path, verbose=verbose)

  def predict(self, X, normalized=True):
    """
    Computes Fisher Vectors of provided X
    :param X: list of arrays with 2 dimensions (n_descriptors, n_dim_descriptor)
    :param normalized: boolean that indicated whether the fisher vectors shall be normalized --> improved fisher vector
              (https://www.robots.ox.ac.uk/~vgg/rg/papers/peronnin_etal_ECCV10.pdf)
    :returns fv: fisher vectors
    """
    return self._predict(X, normalized=normalized)

  def _fit(self, X, model_dump_path=None, verbose=True):
    """
    :param X: shape (n_descriptors, n_dim_descriptor)
    :param model_dump_path: (optional) path where the fitted model shall be dumped
    :param verbose - boolean that controls the verbosity
    :return: fitted Fisher vector object
    """
    assert len(X) > 0
    assert X.ndim == 2

    self.feature_dim = X.shape[-1]

    # fit GMM and store params of fitted model
    self.gmm = gmm = GaussianMixture(n_components=self.n_kernels, covariance_type=self.covariance_type, max_iter=1000).fit(X)
    self.covars = gmm.covariances_
    self.means = gmm.means_
    self.weights = gmm.weights_

    # if cov_type is diagonal - make sure that covars holds a diagonal matrix
    if self.covariance_type == 'diag':
      cov_matrices = np.empty(shape=(self.n_kernels, self.covars.shape[1], self.covars.shape[1]))
      for i in range(self.n_kernels):
        cov_matrices[i, :, :] = np.diag(self.covars[i, :])
      self.covars = cov_matrices

    assert self.covars.ndim == 3
    self.fitted = True
    if verbose:
      print('fitted GMM with %i kernels'%self.n_kernels)

    if model_dump_path:
      with open(model_dump_path, 'wb') as f:
        pickle.dump(self,f, protocol=4)
      if verbose:
        print('Dumped fitted model to', model_dump_path)

    return self

  def _fit_by_bic(self, X, choices_n_kernels=N_Kernel_Choices, model_dump_path=None, verbose=True):
    """
    Fits the GMM with various n_kernels and selects the model with the lowest BIC
    :param X: shape (n_descriptors, n_dim_descriptor)
    :param choices_n_kernels: array of positive integers that specify with how many kernels the GMM shall be trained
                              default: [20, 60, 100, 200, 500]
    :param model_dump_path: (optional) path where the fitted model shall be dumped
    :param verbose - boolean that controls the verbosity
    :return: fitted Fisher vector object
    """

    bic_scores = []
    for n_kernels in choices_n_kernels:
      self.n_kernels = n_kernels
      bic_score = self.fit(X, verbose=False).score(X)
      bic_scores.append(bic_score)

      if verbose:
        print('fitted GMM with %i kernels - BIC = %.4f'%(n_kernels, bic_score))

    best_n_kernels = choices_n_kernels[np.argmin(bic_scores)]

    self.n_kernels = best_n_kernels
    if verbose:
      print('Selected GMM with %i kernels' % best_n_kernels)

    return self.fit(X, model_dump_path=model_dump_path, verbose=True)


  def zero_gmm_weights(self):
    # set equal weights to predict likelihood ratio
    self.gmm.weights_ = np.ones(self.n_kernels) / self.n_kernels

  def _predict_single(self, x, normalized=True):
      n_features, n_feature_dim = x.shape
      
      likelihood_ratio = self.gmm.predict_proba(x)#.reshape(X.shape[0], X.shape[1], self.n_kernels) # (n_features, n_kernels)
      var = np.diagonal(self.covars, axis1=1, axis2=2)

      # norm_dev_from_modes = ((x[:, None, :] - self.means[None, :, :])/ var[None, :, :]) # (n_features, n_kernels, n_feature_dim)
      # Decrease the memory usage of this function by a lot
      norm_dev_from_modes = np.tile(x[:,None,:],(1,self.n_kernels,1))# (n_features, n_kernels, n_featur_dim)
      np.subtract(norm_dev_from_modes, self.means[None, :, :], out=norm_dev_from_modes)
      np.divide(norm_dev_from_modes, var[None, :, :], out=norm_dev_from_modes)

      # mean deviation
      mean_dev = np.multiply(likelihood_ratio[:,:,None], norm_dev_from_modes).mean(axis=0)#(n_kernels, n_feature_dim)
      mean_dev = np.multiply(1 / np.sqrt(self.weights[:,  None]), mean_dev) #(n_kernels, n_feature_dim)

      # covariance deviation
      cov_dev = np.multiply(likelihood_ratio[:,:, None], norm_dev_from_modes**2 - 1).mean(axis=0)
      cov_dev = np.multiply(1 / np.sqrt(2 * self.weights[:,  None]), cov_dev)

      fisher_vector = np.concatenate([mean_dev, cov_dev], axis=0)

      if normalized:
        fisher_vector = np.sqrt(np.abs(fisher_vector)) * np.sign(fisher_vector) # power normalization
        fisher_vector = fisher_vector / np.linalg.norm(fisher_vector, axis=(0,1)) # L2 normalization

      fisher_vector[fisher_vector < 10**-4] = 0 # cause of zero vals

      return fisher_vector

  def _predict(self, X, normalized=True):
    """
    Computes Fisher Vectors of provided X
    :param X: features - list of ndarray of shape (n_features, n_feature_dim)
    :param normalized: boolean that indicates whether the fisher vectors shall be normalized --> improved fisher vector
    :returns fv: fisher vectors - ndarray of shape (2*n_kernels, n_feature_dim)
    """

    assert self.fitted, "Model (GMM) must be fitted"
    assert len(X) > 0
    assert self.feature_dim == X[0].shape[-1], "Features must have same dimensionality as fitted GMM. {:d} != {:d}".format(self.feature_dim, X[0].shape[-1])

    
    fisher_vectors = np.empty((len(X),2*self.n_kernels,self.feature_dim))

    self.zero_gmm_weights()

    for i,x in enumerate(X):
      if x.size == 0:
        # return zeros if the bag is empty
        fv = np.zeros((1, 2*self.n_kernels, self.feature_dim))
      else:
        fv = self._predict_single(x, normalized)
      fisher_vectors[i,:,:] = fv

    return fisher_vectors

  @staticmethod
  def load_from_pickle(pickle_path):
    """
    loads a previously dumped FisherVectorGMM instance
    :param pickle_path: path to the pickle file
    :return: loaded FisherVectorGMM object
    """
    assert os.path.isfile(pickle_path), 'pickle path must be an existing file'
    with open(pickle_path, 'rb') as f:
      fv_gmm = pickle.load(f)
      assert isinstance(fv_gmm, FisherVectorGMM), 'pickled object must be an instance of FisherVectorGMM'
    return fv_gmm