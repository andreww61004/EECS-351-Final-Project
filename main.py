import load_data
import signal_processing
import peak_detection
import help
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Record to retrieve (try '200' or '203' for arrythmia examples)
    record_name = '203'

    annotations = load_data.ecg_annotations(record_name)
    
    ecg = load_data.ecg_signal(record_name)
    record = load_data.record_info(record_name)
    fs = record[2] # Retrieving sampling frequency

    # Filtering
    level = 3
    filtered_ecg = signal_processing.dwavelet_transform(ecg, level)

    # Differentiate
    differentiated_ecg = signal_processing.differentiate(filtered_ecg, 1/fs)

    # Squaring
    squared_ecg = signal_processing.square(differentiated_ecg)

    # Moving average (integration )
    window_size = int(0.05 * fs) 
    integrated_ecg = signal_processing.average(squared_ecg, window_size)

    # Adaptive threshold peak detection
    detector = peak_detection.adaptive_threshold_algorithm(fs)
    detected_peaks_indices = detector.solve(integrated_ecg)

    # Limits of window
    plot_results = True
    low_lim = 108000
    upp_lim = 110000

    # Plot the detected peaks and signal
    if plot_results:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        
        # Plot 1 - Original ECG
        axes[0].plot(ecg[:,0], label='Original ECG', alpha=0.7)
        axes[0].set_title(f"Original ECG (Record {record_name})")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_xlim([low_lim, upp_lim])
        
        # Plot 2 - Integrated signal
        axes[1].plot(integrated_ecg, color='orange', label='Integrated Signal')
        axes[1].set_title(f"Processed Signal (Integration Window: {int(window_size/fs*1000)}ms)")
        axes[1].set_ylabel("Amplitude")

        # Plot 3 - Detected peaks
        axes[2].plot(integrated_ecg, color='green', alpha=0.5)
        
        view_peaks = detected_peaks_indices[
            (detected_peaks_indices >= low_lim) & 
            (detected_peaks_indices <= upp_lim)
        ]
        
        axes[2].plot(view_peaks, integrated_ecg[view_peaks], 'rx', markersize=10, markeredgewidth=2, label='Detected R-Peaks')
        axes[2].set_title("Adaptive Threshold Detection Results")
        axes[2].set_ylabel("Amplitude")
        axes[2].set_xlabel("Samples")
        axes[2].legend()

        print(f"\n--- Detection Stats in window [{low_lim}, {upp_lim}] ---")
        annot_in_window = [x for x in annotations.sample if low_lim <= x <= upp_lim]
        print(f"Annotated Peaks (Truth): {len(annot_in_window)}")
        print(f"Detected Peaks (Algo):  {len(view_peaks)}")

        # Plot the wavelet scales
        help.plot_wavelet_scales(ecg, low_lim, upp_lim, level)

        fig, axes = plt.subplots(5,1, sharex=True, figsize=(10,4))
        
        # Plot the signal progression
        # Plot 1 - Original ECG
        axes[0].plot(ecg[low_lim:upp_lim,0])
        axes[0].set_title('Original ECG')
        axes[0].set_ylabel('Amplitude')

        # Plot 2 - Filtered ECG
        axes[1].plot(filtered_ecg[low_lim:upp_lim])
        axes[1].set_title('Filtered ECG')
        axes[1].set_ylabel('Amplitude')

        # Plot 3 - Differential ECG
        axes[2].plot(differentiated_ecg[low_lim:upp_lim])
        axes[2].set_title('Differential ECG')
        axes[2].set_ylabel('Amplitude')

        # Plot 4 - Squared ECG
        axes[3].plot(squared_ecg[low_lim:upp_lim])
        axes[3].set_title('Squared ECG')
        axes[3].set_ylabel('Amplitude')

        # Plot 5 - Moving average of differential ECG
        axes[4].plot(integrated_ecg[low_lim:upp_lim])
        axes[4].set_title('Integrated ECG')
        axes[4].set_ylabel('Amplitude')
        axes[4].set_xlabel('Samples')

        plt.tight_layout(pad=0)
        plt.show()

    return 0

main()