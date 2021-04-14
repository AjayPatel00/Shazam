from scipy.io import wavfile
import numpy as np

# given wav file path, returns sample rate and samples in mono format
def read_file(path):
    fs,samples = wavfile.read(path)
    if samples.ndim >= 2:
        samples = np.mean(samples,axis=1)#*10e-6
    return fs,samples
