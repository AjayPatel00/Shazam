import numpy as np
from scipy.io import wavfile
from utils import read_file
import os

for song in directory:
    read file
    trim samples
    lowpass samples
    downsample
    write file

for song in os.listdir("songs"):
    f,s = wavfile.read("songs/"+song)
    s = np.mean(s,axis=1)*10e-6*3
    s = s[15*f : 30*f]
    wavfile.write("trimmed_songs/"+song,f,s)