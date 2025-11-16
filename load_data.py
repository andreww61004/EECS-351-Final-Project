import wfdb

# the load_data functions take a record number as a string input
# and use the wfdb package to access the mit-bih database directly.
# the signal, record info, and annotations are all included in the
# data from these functions

def ecg_signal(record_name):
    # load the signal (.dat file)
    # rdrecord() reads the signal (.dat) file and header (.hea) file
    # pn_dir specifies the physionet database path to download from
    try:
        record = wfdb.rdrecord(record_name, pn_dir='mitdb')
        
        # The signal data is stored in 'record.p_signal'
        # This is a NumPy array where each column is a channel/lead
        signal_data = record.p_signal

        return signal_data

    except Exception as e:
        print(f"Could not load record {record_name}. Error: {e}")

def record_info(record_name):
    # load the signal (.dat file)
    # rdrecord() reads the signal (.dat) file and header (.hea) file
    # pn_dir specifies the physionet database path to download from
    try:
        record = wfdb.rdrecord(record_name, pn_dir='mitdb')
        
        # the signal data is stored in record.p_signal
        signal_data = record.p_signal

        print(f"Successfully loaded signal for record: {record_name}")
        print(f"Signal shape (samples, channels): {signal_data.shape}")
        print(f"Sampling frequency (Fs): {record.fs} Hz")

        return [record_name, signal_data.shape, record.fs]

    except Exception as e:
        print(f"Could not load record {record_name}. Error: {e}")

def ecg_annotations(record_name):
    # load the annotations (.atr file)
    # rdann() reads the .atr file (annotations)
    try:
        annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
        
        print(f"\nSuccessfully loaded annotations for record: {record_name}")

        return annotation

    except Exception as e:
        print(f"Could not load annotations for record {record_name}. Error: {e}")