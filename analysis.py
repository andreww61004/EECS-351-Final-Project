import numpy as np

# calculate_rr_statistics takes the R-peak indices
# and calculates heart rate metrics
def calculate_rr_statistics(peaks_indices, fs):
    # Calculate RR intervals in seconds
    rr_intervals = np.diff(peaks_indices) / fs
    
    # Heart rate (BPM)
    bpm_instant = 60 / rr_intervals
    mean_bpm = np.mean(bpm_instant)
    
    # Heart rate metrics
    # Standard deviation of NN (RR) intervals
    sdnn = np.std(rr_intervals)
    
    # Root mean square of successive diffferences
    # (common measure for arrhythmia)
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    
    return {
        "rr_intervals": rr_intervals,
        "mean_bpm": mean_bpm,
        "sdnn": sdnn,
        "rmssd": rmssd
    }

# detect_arrhythmia detects arrhythmia in ECG recordings by analyzing the 
# RR interval. Arrhythmia is detected if the SDNN is unusually high or if
# specific beats devate significantly from the mean
def detect_arrhythmia(rr_stats, threshold_factor=1.2):
    rr = rr_stats['rr_intervals']
    mean_rr = np.mean(rr)
    
    # Check against high variability
    # Normal resting SDNN is ~< 100 ms
    # Atrial fibrillation has SDNN ~> 150 ms or higher depending on rate
    is_irregular_rhythm = rr_stats['sdnn'] > 0.15
    
    # Check against local ectopic beats
    # Flag intervals that are +/- 20% (or threshold_factor) of the mean
    lower_bound = mean_rr / threshold_factor
    upper_bound = mean_rr * threshold_factor
    
    irregular_indices = np.where((rr < lower_bound) | (rr > upper_bound))[0]
    has_ectopic_beats = len(irregular_indices) > 0

    diagnosis = "Normal Sinus Rhythm"
    if is_irregular_rhythm:
        diagnosis = "Arrhythmia Detected (Irregular Rhythm)"
    elif has_ectopic_beats:
        diagnosis = "Arrhythmia Detected (Occasional Ectopic Beats)"
        
    return diagnosis, len(irregular_indices)

# calculate_error_metrics matches the detected peaks with the annotated
# R-peaks. We use this matching to then detect error in our algorithm
# (true peaks, false positives/negatives) 
def calculate_error_metrics(detected_peaks, true_peaks, fs, tolerance_ms=100):
    # Convert tolerance from ms to samples
    tolerance_samples = int(tolerance_ms / 1000 * fs)
    
    tp = 0 # true positives
    fp = 0 # false positives
    fn = 0 # false negatives
    
    time_errors = [] # differences in samples between matched peak and true peak
    
    # We copy true_peaks to mark them as "matched" to avoid double counting
    unmatched_true_peaks = set(true_peaks)
    
    # For every detected peak, find the closest true peak
    for d_idx in detected_peaks:
        # Find closest true peak (simple search)
        closest_true_idx = min(true_peaks, key=lambda x: abs(x - d_idx))
        distance = d_idx - closest_true_idx
        
        if abs(distance) <= tolerance_samples:
            tp += 1
            time_errors.append(distance)
            if closest_true_idx in unmatched_true_peaks:
                unmatched_true_peaks.remove(closest_true_idx)
        else:
            # If the closest true peak is too far, this is a false positive
            fp += 1
            
    # Any true peak that wasn't matched is a false negative
    fn = len(unmatched_true_peaks)
    
    # Calculate stats
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0 # Positive Predictive Value
    
    # Error distance stats
    errors_ms = (np.array(time_errors) / fs) * 1000
    mae_error = np.mean(np.abs(errors_ms)) if len(errors_ms) > 0 else 0
    rmse_error = np.sqrt(np.mean(errors_ms**2)) if len(errors_ms) > 0 else 0
    
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "Sensitivity": sensitivity,
        "PPV": ppv,
        "MAE_ms": mae_error,     # mean absolute error
        "RMSE_ms": rmse_error    # root mean square error
    }