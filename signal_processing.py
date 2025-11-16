import wfdb
import pywt
import numpy as np
from scipy.signal import hilbert
import load_data

# db5 db10 sym4 sym5

def wavelet_transform(signal, level):
    # specify the wavelet type and decomposition level for the transform
    wavelet = pywt.Wavelet('sym5')

    # perform the stationary wavelet transform
    coeffs = pywt.swt(signal[:,0], wavelet, level)

    # zero the scales we want to filter (cA3, cD2, cD1)
    # coeffs[0][0][:] = 0

    for i in range(1, level - 1):
        coeffs[i][1][:] = 0

    # reconstruct the signal using the inverse wavelet transform
    return pywt.iswt(coeffs, wavelet)



def differentiate(signal, Ts):
    return np.gradient(signal, Ts)



def square(signal):
    return np.square(signal)



def normalize(signal):
    max_value = np.max(signal)
    return np.divide(signal,max_value)



def envelope(signal):
    transform_signal = hilbert(signal)
    return np.abs(transform_signal)