import load_data
import signal_processing
import help
import numpy as np
import matplotlib.pyplot as plt

def main():
    # the name of the record to retrieve
    record_name = '101'

    # the level of decomposition in the wavelet transform
    level = 3

    # obtain details of the record
    ecg = load_data.ecg_signal(record_name)
    record = load_data.record_info(record_name)

    # use the wavelet transform to filter the ecg data
    filtered_ecg = signal_processing.wavelet_transform(ecg, level)

    # obtain first order differential signal of filtered ecg
    differentiated_ecg = signal_processing.differentiate(filtered_ecg, 1/record[2])

    # obtain squared, first order filtered ecg
    squared_ecg = signal_processing.square(differentiated_ecg)

    # normalize the ecg signal
    normalized_ecg = signal_processing.normalize(squared_ecg)

    # obtain the envelope of squared, first order filtered ecg
    ecg_envelope = signal_processing.envelope(normalized_ecg)





    # - plots for testing -
    plot_signal_filtered = True
    plot_wavelets = False
    upp_lim = 1500
    # ---------------------

    if plot_signal_filtered:
        fig, axes = plt.subplots(6,1)
        
        axes[0].plot(ecg[0:upp_lim,0])
        axes[1].plot(filtered_ecg[0:upp_lim])
        axes[2].plot(differentiated_ecg[0:upp_lim])
        axes[3].plot(squared_ecg[0:upp_lim])
        axes[4].plot(normalized_ecg[0:upp_lim])
        axes[5].plot(ecg_envelope[0:upp_lim])
        plt.show()

    if plot_wavelets:
        help.plot_wavelet_scales(ecg,0,upp_lim, level)
        plt.show()

    return 0

main()