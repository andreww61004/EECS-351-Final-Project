import wfdb
import pywt
import numpy as np
from scipy.signal import hilbert
import load_data

# the wavelet transform is used to filter the signal.
# the transform breaks the signal into time-frequency
# scales (approximation and detail coefficients) which
# then be zeroed to effectively filter the signal
def wavelet_transform(signal, level):
    # specify the wavelet type and decomposition level for the transform
    wavelet = pywt.Wavelet('sym5')

    # perform the stationary wavelet transform
    coeffs = pywt.swt(signal[:,0], wavelet, level)

    # zero the scales we want to filter (cD2, cD1)
    for i in range(0, level - 1):
        coeffs[i][1][:] = 0

    # reconstruct the signal using the inverse wavelet transform
    return pywt.iswt(coeffs, wavelet)


# this function estimates the derivative of the signal
# and returns the differentiated signal
def differentiate(signal, Ts):
    return np.gradient(signal, Ts)


# this function applies a squaring operation
# to the signal values
def square(signal):
    return np.square(signal)


# this is an n (window_size) point moving average filter
def average(signal, window_size):
    weights = np.ones(window_size) / window_size
    ecg_envelope = np.convolve(signal, weights, mode='valid')
    return ecg_envelope