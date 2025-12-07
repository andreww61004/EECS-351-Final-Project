import wfdb
import pywt
import numpy as np
import load_data

class signal_processing_tools:
    def __init__(self, fs, level, window_size):
        # Initializing the signal and sampling frequency
        self.fs = fs
        self.level = level
        self.window_size = window_size

    # dwavelet_transform filters the signal by breaking the 
    # signal into time-frequency scales (approximation and detail
    # coefficients), which can then be scaled to effectively
    # filter the signal
    def dwavelet_transform(self, signal):
        # specify the wavelet type and decomposition level for the transform
        wavelet = pywt.Wavelet('sym4')

        # perform the stationary wavelet transform
        coeffs = pywt.wavedec(signal[:,0], wavelet, self.level)

        # zero the scales we want to filter (cD2, cD1)
        #coeffs[0] = np.zeros_like(coeffs[0])
        for i in range(1, self.level - 1):
            coeffs[i] = np.zeros_like(coeffs[i])

        # reconstruct the signal using the inverse wavelet transform
        return pywt.waverec(coeffs, wavelet)
    
    # differentiate estimates the differential of the signal
    def differentiate(self, signal):
        return np.gradient(signal, 1/self.fs)
    
    # square squares the signal values
    def square(self, signal):
        return np.square(signal)
    
    # average uses an N point (window size) moving average filter to
    # obtain the envelope of the signal
    def average(self, signal):
        weights = np.ones(self.window_size) / self.window_size
        ecg_envelope = np.convolve(signal, weights, mode='valid')
        return ecg_envelope