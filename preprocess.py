import numpy as np
from scipy.io import wavfile
import scipy.signal as sps
from utils import read_file
import os

# function downsamples a signal with sampling frequency
# fs by a factor of factor. Used a factor of 4 for this 
# project
def downsample(fs,s,factor=4):
    return int(fs/factor), sps.resample(s,int(len(s)/factor))

# preprocess a directory. Directory must look like:
# directory/song1.mp3, directory/song2.mp3, ...
def preprocess_directory(path):
    for song in os.listdir(path):
        # read_file using utils function
        fs,samples = read_file(path+"/"+song)
        # if in stereo format, convert to mono by computing the average of
        # the two channels
        if samples.ndim >=2: samples = np.mean(samples,axis=1)
        # butterworth filter with low cutoff frequency of 4Hz, and high
        # cut off frequency of 10kHz. Use bandpass filter, butterworth of
        # order 4
        sos = sps.butter(4, [10,10000], 'bandpass', fs=fs, output='sos')
        filtered = sps.sosfilt(sos, samples)
        # downsample the signal by a factor of 4
        fs,samples = downsample(fs,samples,factor=4)
        # trim the song to 60 seconds
        samples = samples[:fs*60]
        # write the preprocessed song to destination directory
        os.makedirs("processed_"+path,exist_ok=True)
        wavfile.write("processed_"+path+"/"+song,fs,samples.astype(np.int16))


preprocess_directory("songs/dset6")
preprocess_directory("songs/dset100/Blues")
preprocess_directory("songs/dset100/Classical")
preprocess_directory("songs/dset100/Country")
preprocess_directory("songs/dset100/Electronic")
preprocess_directory("songs/dset100/Folk")
preprocess_directory("songs/dset100/Hip-Hop")
preprocess_directory("songs/dset100/Jazz")
preprocess_directory("songs/dset100/Pop")
preprocess_directory("songs/dset100/Rock")
preprocess_directory("songs/dset100/Soul-RB")

