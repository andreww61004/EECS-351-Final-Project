import load_data
import signal_processing
import help
import numpy as np
import matplotlib.pyplot as plt

def main():
    # the name of the record to retrieve
    record_name = '121'

    annotations = load_data.ecg_annotations(record_name)

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

    # obtain the envelope of squared, first order filtered ecg
    ecg_average5 = signal_processing.average(squared_ecg, 5)
    ecg_average15 = signal_processing.average(squared_ecg, 15)





    # - plots for testing -
    plot_filt_progression = False
    plot_filt_ecg = True
    plot_wavelets = False
    plot_filt_on_original = False
    low_lim = 361880
    upp_lim = 365880
    # ---------------------

    if plot_filt_ecg:
        fig, axes = plt.subplots(3,1)
        
        axes[0].plot(ecg[low_lim:upp_lim,0])
        axes[0].set_title(f"ECG Recording")
        axes[0].set_ylabel(f"Amplitude")

        #axes[1].plot(filtered_ecg[low_lim:upp_lim])
        #axes[2].plot(differentiated_ecg[low_lim:upp_lim])
        #axes[3].plot(squared_ecg[low_lim:upp_lim])

        axes[1].plot(ecg_average5[low_lim:upp_lim])
        axes[1].set_title(f"Processed ECG + 5 Point Moving Average")
        axes[1].set_ylabel(f"Amplitude")

        axes[2].plot(ecg_average15[low_lim:upp_lim])
        axes[2].set_title(f"Processed ECG + 15 Point Moving Average")
        axes[2].set_ylabel(f"Amplitude")
        axes[2].set_xlabel(f"Samples : [{low_lim},{upp_lim}]")

        for i in range(0,len(annotations.symbol)):
            if annotations.sample[i] < low_lim:
                continue
            elif annotations.sample[i] >= low_lim and annotations.sample[i] <= upp_lim:
                print(f"{annotations.symbol[i]} at sample {annotations.sample[i]}")
            else:
                break

        plt.show()

    if plot_wavelets:
        help.plot_wavelet_scales(ecg,low_lim,upp_lim, level)
        plt.show()

    if plot_filt_on_original:
        plt.plot(filtered_ecg[low_lim:upp_lim])
        plt.plot(squared_ecg[low_lim:upp_lim])
        plt.plot(ecg_average5[low_lim:upp_lim])
        plt.show()

    return 0

main()