import numpy as np
from scipy.io import wavfile
import scipy.signal as sps
from utils import read_file
import os

def trim(fs,samples,n_seconds=30):
    return samples[fs*10:fs*(10+n_seconds)]

def downsample(fs,s,factor=4):
    return int(fs/4), sps.resample(s,int(len(s)/4))


def preprocess_directory(path):
    for song in os.listdir(path):
        fs,samples = read_file(path+"/"+song)
        if samples.ndim >=2: samples = np.mean(samples,axis=1)
        #samples = trim(fs,samples,n_seconds=30)
        samples = samples[:fs*45]
        fs,samples = downsample(fs,samples,factor=4)
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


"""
MP3 -> WAV

from pydub import AudioSegment as AS
s = AS.from_mp3(file.mp3)
s.export(file.wav,format="wav")å
"""