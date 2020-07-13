import numpy as np
import scipy.signal

# Function to extract statistical and spectral features from a single window
# Takes an ndarray containing a signal per column, shape [n_samples,n_axes]
# outputs an array, shape [n_features]
def extract_psd_features(signal, nfft=64, nbins=6):
    n_samples, n_axes = signal.shape
    if signal.ndim != 2:
        raise 'Input signal incorrect. Should have dim 2'
    if n_axes != 3:
        raise 'Input signal incorrect. Should have 3 accel axes'

    # add the magnitude and absolute value signals
    expanded_signal = np.zeros((n_samples, 7))
    expanded_signal[:, 0:3] = signal
    expanded_signal[:, 3:6] = np.absolute(signal)
    expanded_signal[:, 6] = np.sqrt(
        pow(expanded_signal[:, 0], 2) + pow(expanded_signal[:, 1], 2) +
        pow(expanded_signal[:, 2], 2))

    n_features_per_axis = (nbins + 2)
    # the feature vector
    features = np.zeros(n_features_per_axis * 7)

    # For each accel axis
    for ax in range(0, expanded_signal.shape[1]):

        # # legacy feature extraction for comparison
        # f,psd = scipy.signal.periodogram(x=expanded_signal[:,ax],fs=20,nfft=800,return_onesided=True)
        # ids = np.nonzero(np.logical_or.reduce((f==0,f==0.125, f == 0.25,f == 0.5, f== 1, f== 2, f==4, f==8)))
        # binned_psd = psd[ids]

        # Compute psd with periodogram
        f, psd = scipy.signal.periodogram(x=expanded_signal[:, ax],
                                          nfft=nfft,
                                          return_onesided=True)

        # bin the PSD
        binned_psd = np.zeros(nbins)
        # if the bins have power of two bounds
        if 2**(nbins) == nfft:
            binned_psd[0] = psd[1]
            i = 1
            j = 2
            bin_size = 1
            while bin_size <= nfft / 4:
                for k in range(0, bin_size):
                    binned_psd[i] += psd[j]
                    j += 1
                i += 1
                bin_size *= 2

        else:
            raise 'not implemented'
            # bin_bounds = np.logspace(0,np.log10(nfft / 2),nbins-1)
            # binned_psd[0] = psd[0]
            # i = 1
            # for j in range(0,len(bin_bounds)):
            #     while i <= bin_bounds[j]:
            #         print(j)
            #         binned_psd[j+1] += psd[i]
            #         i += 1

        # copy the binned psd
        features[ax * n_features_per_axis + 0:ax * n_features_per_axis +
                 nbins] = binned_psd

        # compute mean and variance
        features[ax * n_features_per_axis + nbins + 0] = np.mean(
            expanded_signal[:, ax], axis=0)
        features[ax * n_features_per_axis + nbins + 1] = np.std(
            expanded_signal[:, ax], axis=0)

    return features