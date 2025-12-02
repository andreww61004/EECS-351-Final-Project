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

    # Derivative
    differentiated_ecg = signal_processing.differentiate(filtered_ecg, 1/fs)

    # Squaring
    squared_ecg = signal_processing.square(differentiated_ecg)

    # Moving average (integration)
    window_size = int(0.05 * fs) 
    integrated_ecg = signal_processing.average(squared_ecg, window_size)

    # Adaptive threshold peak detection
    detector = peak_detection.adaptive_threshold_algorithm(fs)
    detected_peaks_indices = detector.solve(integrated_ecg)

    #Plots for testing
    plot_results = True
    low_lim = 108000
    upp_lim = 110000

    if plot_results:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        
        # Plot 1: Original ECG
        axes[0].plot(ecg[:,0], label='Original ECG', alpha=0.7)
        axes[0].set_title(f"Original ECG (Record {record_name})")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_xlim([low_lim, upp_lim])
        
        # Plot 2: Integrated Signal
        axes[1].plot(integrated_ecg, color='orange', label='Integrated Signal')
        axes[1].set_title(f"Processed Signal (Integration Window: {int(window_size/fs*1000)}ms)")
        axes[1].set_ylabel("Amplitude")

        # Plot 3: Detected Peaks
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

        plt.tight_layout()
        plt.show()

    return 0

main()