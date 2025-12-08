import load_data
import signal_processing
import peak_detection
import help
import analysis
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load the record data

    # Record to retrieve (try '200' or '203' for arrythmia examples)
    record_name = '203'
    # Viewing window limits
    low_lim = 108000   # valid values between 0 and 650,000
    upp_lim = 110000   #

    # Load the annotations for error analysis
    annotations = load_data.ecg_annotations(record_name)

    ecg = load_data.ecg_signal(record_name)
    record = load_data.record_info(record_name)
    fs = record[2] # retrieving sampling frequency

    # Process the signal

    # Instance the signal processing tools
    level = 3
    window_size = int(0.05 * fs)
    processor = signal_processing.signal_processing_tools(fs, level, window_size)

    # Filtering
    filtered_ecg = processor.dwavelet_transform(ecg)

    # Differentiate
    differentiated_ecg = processor.differentiate(filtered_ecg)

    # Squaring
    squared_ecg = processor.square(differentiated_ecg)

    # Moving average (integration )
    integrated_ecg = processor.average(squared_ecg)



    # Detect the R-peaks

    # R-peak detection uses a modified adaptive thresholding algorithm
    detector = peak_detection.adaptive_threshold_algorithm(fs)
    detected_peaks_indices = detector.solve(integrated_ecg)



    # Calculate the statistics of the record

    print(f"\n--- Analysis Results for Record {record_name} ---")
    
    # RR interval and arrhythmia analysis
    rr_stats = analysis.calculate_rr_statistics(detected_peaks_indices, fs)
    diagnosis, ectopic_count = analysis.detect_arrhythmia(rr_stats)
    
    print(f"Mean Heart Rate: {rr_stats['mean_bpm']:.2f} BPM")
    print(f"SDNN (Variability): {rr_stats['sdnn']:.4f} s")
    print(f"Automated Diagnosis: {diagnosis}")
    print(f"Irregular Beats Detected: {ectopic_count}")

    # Error stats
    # We use annotations.sample for the true peak indices. We filter out
    # non-beat annotations and compare to our detected R-peaks
    valid_beat_symbols = set(['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 
                              'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'])

    true_peaks = [
        samp for samp, symb in zip(annotations.sample, annotations.symbol) 
        if symb in valid_beat_symbols
    ]
    true_peaks = np.array(true_peaks)
    
    perf_metrics = analysis.calculate_error_metrics(detected_peaks_indices, true_peaks, fs)
    
    print(f"\n--- Performance Metrics ---")
    print(f"True Positives (TP): {perf_metrics['TP']}")
    print(f"False Positives (FP): {perf_metrics['FP']}")
    print(f"False Negatives (FN): {perf_metrics['FN']}")
    print(f"Sensitivity: {perf_metrics['Sensitivity']:.2%}")
    print(f"Positive Predictive Value: {perf_metrics['PPV']:.2%}")
    print(f"Average Error Distance (MAE): {perf_metrics['MAE_ms']:.2f} ms")



    # Generate the plots for this record in the window

    # Windows limits for graphs
    plot_results = True

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
        annot_in_window = [x for x in true_peaks if low_lim <= x <= upp_lim]
        print(f"Annotated Peaks (True Peaks): {len(annot_in_window)}")
        print(f"Detected Peaks (Algorithm):  {len(view_peaks)}")

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