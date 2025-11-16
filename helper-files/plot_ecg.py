import wfdb
import matplotlib.pyplot as plt

# define the record name (e.g., record '100' from the database)
record_name = '114'
# record_name = str(input("Enter the record name (e.g., '100'): "))

# load the signal (.dat file)
try:
    record = wfdb.rdrecord(record_name, pn_dir='mitdb')
    
    # The signal data is stored in 'record.p_signal'
    # This is a NumPy array where each column is a channel/lead
    signal_data = record.p_signal

    # plot the first 2500 samples of the first channel
    plt.plot(signal_data[0:2500, 0])
    plt.show()

except Exception as e:
    print(f"Could not load record {record_name}. Error: {e}")


# load the annotations (.atr file)
# rdann() reads the .atr file (annotations)
try:


    annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
    
    print(f"\nSuccessfully loaded annotations for record: {record_name}")
    
    # annotation.sample: the sample number of each annotation
    # annotation.symbol: the annotation symbol (e.g., 'N' for normal, 'V' for PVC)

except Exception as e:
    print(f"Could not load annotations for record {record_name}. Error: {e}")