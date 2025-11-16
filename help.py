import load_data
import pywt
import matplotlib.pyplot as plt

def plot_wavelet_scales(ecg_signal, low_limit, upp_limit, level):
    # perform the wavelet transform to obtain
    # the wavelet scales
    wavelet = pywt.Wavelet('sym5')

    coeffs = pywt.swt(ecg_signal[low_limit:upp_limit,0], wavelet, level)

    # create plot
    fig, axes = plt.subplots(2*level+1,1)

    axes[0].plot(ecg_signal[low_limit:upp_limit,0])

    # individually plot the wavelet scales
    for i in range(level):
        # plot approximation coefficients
        axes[i+1].plot(coeffs[i][0])
        axes[i+1].set_ylabel(f'cA_{level-i}')

        # plot detail coefficients
        axes[i+level+1].plot(coeffs[i][1])
        axes[i+level+1].set_ylabel(f'cD_{level-i}')
    
    plt.show()