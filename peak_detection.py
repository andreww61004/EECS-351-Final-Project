import numpy as np

class adaptive_threshold_algorithm:
    # Initializing all values for the adaptive thresholding algorithm
    def __init__(self, fs):
        # Initalizing the sampling frequency
        self.fs = fs

        # Refractory period = 150 ms
        self.REFRACTORY_PERIOD = int(0.15 * fs)   
        
        # Look-ahead window = 200 ms
        # When a peak candidate is found, we search ahead 200 ms for a 
        # larger peak to ensure the correct R-peak was found
        self.QRS_WINDOW = int(0.20 * fs)          
        
        # T-Wave Window = 360ms
        # Window to discriminate T-waves from PVCs.
        self.T_WAVE_WINDOW = int(0.36 * fs)       
        
        # Searchback triggers if no beat found for 1.66x average RR
        self.SEARCHBACK_LIMIT_FACTOR = 1.66
        
        # Initializing the thresholds and the states
        self.SPKI = 0.0  # running signal peak estimate
        self.NPKI = 0.0  # running noise peak estimate
        self.threshold_i1 = 0.0 # primary threshold
        self.threshold_i2 = 0.0 # secondary (lower) threshold
        self.last_qrs_val = 0.0 

        self.rr_intervals = [] 
        self.peaks_indices = [] 
    
    # The solve function takes the processed signal and searches the signal for candidate peaks.
    # The function then validates the R-peaks using a window to look ahead for other peaks, and
    # also searches the previous samples if an R-peak was missed
    def solve(self, signal):
        # Initialization: Initial values for thresholds using first 2 seconds of ECG data
        init_window = signal[:2*int(self.fs)]
        if len(init_window) > 0:
            self.SPKI = np.max(init_window) * 0.25
            self.NPKI = np.mean(init_window) * 0.5
        else:
            self.SPKI = 0.5
            self.NPKI = 0.1
        self.update_thresholds()

        # Find all local maxima (candidate peaks) to iterate through
        candidate_peaks = self.find_local_maxima(signal)
        
        last_qrs_index = 0
        
        # State variables for the look-ahead window
        potential_peak_idx = None
        potential_peak_val = -np.inf

        for i, peak_idx in enumerate(candidate_peaks):
            peak_val = signal[peak_idx]
            
            # Look ahead from the detected R-peak to search for other possible R-peaks
            if potential_peak_idx is not None:
                # Check if this new peak is within the confirmation window of the pending peak
                if peak_idx - potential_peak_idx < self.QRS_WINDOW:
                    # If this new peak is bigger, it becomes the new candidate.
                    if peak_val > potential_peak_val:
                        potential_peak_idx = peak_idx
                        potential_peak_val = peak_val
                    
                    # Continue to next iteration to keep checking this window
                    continue 
                else:
                    # Now we assume it is the R-peak and run final checks
                    registered_idx = self.finalize_peak(potential_peak_idx, potential_peak_val, last_qrs_index)
                    
                    if registered_idx is not None:
                        last_qrs_index = registered_idx
                    
                    # Reset state
                    potential_peak_idx = None
                    potential_peak_val = -np.inf

            # Evaluate current peak as new candidate
            # Refractory check (relative to the last confirmed QRS)
            if peak_idx - last_qrs_index < self.REFRACTORY_PERIOD:
                continue

            # Threshold check
            if peak_val >= self.threshold_i1:
                # Instead of saving new candidate immediately, start the look-ahead window.
                potential_peak_idx = peak_idx
                potential_peak_val = peak_val
                
            else:
                # If the peak is too small it is ignored
                self.NPKI = 0.125 * peak_val + 0.875 * self.NPKI
                self.update_thresholds()
                
                # If we haven't found a peak in a long time then we look at the most recent samples
                # for a peak
                avg_rr = np.mean(self.rr_intervals) if len(self.rr_intervals) > 0 else self.fs
                if (peak_idx - last_qrs_index) > (self.SEARCHBACK_LIMIT_FACTOR * avg_rr):
                    found_idx = self.perform_searchback(signal, last_qrs_index, peak_idx)
                    if found_idx is not None:
                        last_qrs_index = found_idx

                        # If searchback found a beat, clear any potential peaks
                        potential_peak_idx = None 

        # Final cleanup if a peak was pending at the very end of the signal
        if potential_peak_idx is not None:
             self.finalize_peak(potential_peak_idx, potential_peak_val, last_qrs_index)

        return np.array(self.peaks_indices)

    # finalize_peak takes the peak indices and validates them
    # against possible PVC like patterns
    def finalize_peak(self, idx, val, last_qrs_idx):
        dt = idx - last_qrs_idx
        
        # Looking for T-waves that look like QRS complexes
        # (classified as less than 0.5 of the last QRS complex value)
        is_t_wave = False

        if dt < self.T_WAVE_WINDOW:
             if val < 0.5 * self.last_qrs_val:
                 is_t_wave = True
        
        # Checking if a T-wave is detected
        if is_t_wave:
            # Updating our thresholds
            self.NPKI = 0.125 * val + 0.875 * self.NPKI
            self.update_thresholds()
            return None
        else:
            # Valid QRS found
            self.peaks_indices.append(idx)
            self.SPKI = 0.125 * val + 0.875 * self.SPKI
            self.last_qrs_val = val

            # Updating our values for the next iteration
            self.update_rr_history(idx)
            self.update_thresholds()
            return idx

    # Returns the indices of the local maxima
    def find_local_maxima(self, signal):
        diff_sig = np.diff(signal)
        
        maxima_mask = (diff_sig[:-1] > 0) & (diff_sig[1:] < 0)
        return np.where(maxima_mask)[0] + 1

    # Updates our threshold states for the running SPKI and NPKI
    def update_thresholds(self):
        self.threshold_i1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
        self.threshold_i2 = 0.5 * self.threshold_i1

    # Updates the rr history for the following iterations of the algorithm
    def update_rr_history(self, current_peak_idx):
        if len(self.peaks_indices) > 1:
            rr = current_peak_idx - self.peaks_indices[-2]
            
            if len(self.rr_intervals) == 0 or rr < 2.0 * np.mean(self.rr_intervals):
                self.rr_intervals.append(rr)
                if len(self.rr_intervals) > 8:
                    self.rr_intervals.pop(0)

    # perform_searchback searches the past samples for possible R-peaks
    # if the algorithm goes too long without breaking the threshold
    def perform_searchback(self, signal, start_idx, end_idx):
        search_start = start_idx + self.REFRACTORY_PERIOD
        search_end = end_idx
        
        if search_end <= search_start:
            return None
            
        window = signal[search_start : search_end]
        if len(window) == 0: 
            return None
            
        # Find the highest peak in the missed window
        local_max_idx = np.argmax(window) 
        peak_val = window[local_max_idx]
        real_idx = search_start + local_max_idx

        # Check against secondary lower threshld
        if peak_val > self.threshold_i2:
            self.peaks_indices.append(real_idx)
            self.SPKI = 0.25 * peak_val + 0.75 * self.SPKI
            self.last_qrs_val = peak_val

            # Updating our values for the next iteration
            self.update_rr_history(real_idx)
            self.update_thresholds()
            return real_idx
            
        return None