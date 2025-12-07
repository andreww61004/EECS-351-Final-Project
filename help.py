import load_data
import pywt
import matplotlib.pyplot as plt
import numpy as np

def plot_wavelet_scales(ecg_signal, low_limit, upp_limit, level):
    # Removes the dc offset
    ecg_signal[low_limit:upp_limit, 0] = ecg_signal[low_limit:upp_limit, 0] - np.mean(ecg_signal[low_limit:upp_limit, 0])

    # Basis function
    wavelet = pywt.Wavelet('sym4')
    
    # Obtain the wavelet scale coefficients
    coeffs = pywt.wavedec(ecg_signal[low_limit:upp_limit, 0], wavelet, level=level, mode='periodization')

    # We reconstruct each coefficient band back into the time domain
    reconstructed_scales = []
    
    # coeffs = [cA_3, cD_3, cD_2, cD_1]
    for i in range(len(coeffs)):
        # Create a blank list of coefficients
        temp_coeffs = [np.zeros_like(c) for c in coeffs]
        
        # Isolate the current scale
        temp_coeffs[i] = coeffs[i]
        
        # Inverse transform the isolated scale
        rec = pywt.waverec(temp_coeffs, wavelet, mode='periodization')
        
        # Length matching
        # Waverec can return length N or N+1 depending on even/odd input length
        if len(rec) > len(ecg_signal[low_limit:upp_limit, 0]):
            rec = rec[:len(ecg_signal[low_limit:upp_limit, 0])]
        elif len(rec) < len(ecg_signal[low_limit:upp_limit, 0]):
            pad_width = len(ecg_signal[low_limit:upp_limit, 0]) - len(rec)
            rec = np.pad(rec, (0, pad_width), 'constant')
            
        reconstructed_scales.append(rec)

    # Plot the scales
    num_plots = 1 + len(reconstructed_scales)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 12), sharex=True)
        
    # Ensure axes is iterable
    if num_plots == 1: axes = [axes]

    # Plot original
    axes[0].plot(ecg_signal[low_limit:upp_limit, 0], color='black', label='Original')
    axes[0].set_title(f'Original Signal (Samples {low_limit} - {upp_limit})')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Plot approximation (first coefficient)
    axes[1].plot(reconstructed_scales[0], color='green', label=f'cA{level}')
    axes[1].set_title(f'Approximation (Low Freq / Baseline)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Plot details (remaining coefficients)
    for i in range(1, len(reconstructed_scales)):
        detail_lvl = level - (i - 1)
        ax_idx = i + 1
            
        axes[ax_idx].plot(reconstructed_scales[i], color='red', label=f'cD{detail_lvl}')
        axes[ax_idx].set_title(f'Detail Level {detail_lvl} (High Freq)')
        axes[ax_idx].legend(loc='upper right')
        axes[ax_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=True)