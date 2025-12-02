import wfdb
import pywt
import numpy as np
from scipy.signal import hilbert
import load_data

# The wavelet transform is used to filter the signal.
# The transform breaks the signal into time-frequency
# scales (approximation and detail coefficients) which
# can then be scaled to effectively filter the signal

def dwavelet_transform(signal, level):
    # specify the wavelet type and decomposition level for the transform
    wavelet = pywt.Wavelet('sym5')

    # perform the stationary wavelet transform
    coeffs = pywt.wavedec(signal[:,0], wavelet, level)

    # zero the scales we want to filter (cD2, cD1)
    #coeffs[0] = np.zeros_like(coeffs[0])
    for i in range(1, level - 1):
        coeffs[i] = np.zeros_like(coeffs[i])

    # reconstruct the signal using the inverse wavelet transform
    return pywt.waverec(coeffs, wavelet)


# Estimates the derivative of the values in a signal
def differentiate(signal, Ts):
    return np.gradient(signal, Ts)


# Squares the signal values
def square(signal):
    return np.square(signal)


# N point (window_size) moving average filter
def average(signal, window_size):
    weights = np.ones(window_size) / window_size
    ecg_envelope = np.convolve(signal, weights, mode='valid')
    return ecg_envelope