from database import Fingerprint, Database
from utils import read_file
from fingerprint import get_spectrogram,get_peaks,gen_hashes

import matplotlib.pyplot as plt
import numpy as np
import time
from itertools import groupby
from operator import itemgetter as ig
import pdb
import os
import sounddevice as sd
import math

np.seterr(divide='ignore')
fs=11025
myDB = Database()



def fingerprint_song(song):
    song_name = song.split("/")[-1]
    fs,samples = read_file(song)
    spec = get_spectrogram(fs,samples,plot=False)
    peaks = get_peaks(spec,plot=False)
    hashes = gen_hashes(peaks)
    for h in hashes:
        f = Fingerprint(song_name,h[0],h[1])
        myDB.add(f)
    myDB.song_table[song_name] = len(hashes)

def fingerprint_directory(path):
    for s in os.listdir(path): fingerprint_song(path+"/"+s)

def recognize_recording(samples,fs):
    spec = get_spectrogram(fs,samples)
    peaks = get_peaks(spec,amp_min=4,peak_nhood_size=10)
    hashes = gen_hashes(peaks)
    matches,songs = myDB.search(hashes)

    f = lambda x: (x[0],x[1])
    counts = [(*k,len(list(g))) for k,g in groupby(sorted(matches,key=f), key=f)]
    matches = [max(list(g),key=ig(2)) for k,g in groupby(counts, key=ig(0))]
    matches = sorted(matches,key=ig(2),reverse=True)

    result = []
    for song,offset,_ in matches[:5]:
        time_offset = int(float(offset)/fs*4096*0.5) #sampling rate * window size * overlap ratio
        result.append({"song_id": song,
                       "total_hashes": myDB.song_table[song],
                       "input_hashes": len(hashes),
                       "matched_hashes": songs[song]})#,
                       #"input_confidence": round(songs[song]/len(hashes),2),
                       #"fingerprint_confidence": round(songs[song]/myDB.song_table[song],2),
                       #"offset_seconds":time_offset})
    return result

def recognize_directory(path,start,end,noise=False,snr=None,verbose=0):
    #print("Recognizing all songs from ",path,"...\n")
    correct = 0
    for s in os.listdir(path):
        fs,samples = read_file(path+"/"+s)
        if noise: samples = add_noise(snr,samples)
        samples = samples[start*fs:end*fs]
        results = recognize_recording(samples,fs)
        if len(results) != 0:
            matched_song = results[0]["song_id"]
            if matched_song== s: correct += 1
    accuracy = (correct/len(os.listdir(path)))*100
    if verbose: print("Shazam correctly identified",accuracy,"percent of songs in",path)
    return accuracy

def recognize_from_mic(fs=44100,n_seconds=15):
    print("start playing music")
    time.sleep(5)
    print("microphone enabled")
    recording = sd.rec(int(n_seconds*fs),samplerate=fs,channels=1)
    sd.wait()
    print("microphone disabled, searching for matches ...")
    results = recognize_recording(recording.flatten(),fs)
    if len(results) == 0:
        print("Shazam found no matches. Try getting closer to source of music")
    else:
        matched_song = results[0]["song_id"]
        print("Shazam found the followign match: ",matched_song)

# duration of recording vs recognition accuracy
def experiment1(dset_path,plot=1):
    # first fingerprint
    for genre in os.listdir(dset_path):
        if not genre.startswith("."): 
            fingerprint_directory(dset_path+"/"+genre)

    accs = []
    times = [1,2,3,4,5,6,7,8,9,10,12,13,14,15]
    for t in times:
        time_accs = []
        for genre in os.listdir(dset_path):
            if not genre.startswith("."): 
                genre_acc = recognize_directory(dset_path+"/"+genre,start=15,end=15+t)
                time_accs.append(genre_acc)
        accs.append(np.mean(time_accs))

    if plot:
        plt.plot([0]+times,[0]+accs)
        plt.title("Duration of Recording vs Recognition Accuracy")
        plt.xlabel("Duration of recording (s)")
        plt.ylabel("Recognition Accuracy")
        plt.show()

# add additive white gaussian noise AWGN to signal
# so that the signal to noise ratiois as specified
# RMS_noise is ~= std of noise since mean is close to 0
def add_noise(snr,samples):
    signal_rms = math.sqrt(np.mean([s**2 for s in samples]))
    noise_rms = math.sqrt(signal_rms**2/(pow(10,snr/10)))
    std = noise_rms
    noise = np.random.normal(0,std,samples.shape[0])
    noisy_samples = noise+samples
    return noisy_samples

def experiment2(dset_path,duration=10,plot=1):
    # first fingerprint
    for genre in os.listdir(dset_path):
        if not genre.startswith("."): 
            fingerprint_directory(dset_path+"/"+genre)

    # add increasing amount of noise. Signal to noise ratio values.
    # testing accuracy on snrs = [40,37,...,3,0,-3,...,-17,-20]
    snrs = np.linspace(40,-20,21)
    accs = []
    for snr in snrs:
        print("processing noise level snr = ",snr)
        snr_accs = []
        for genre in os.listdir(dset_path):
            if not genre.startswith("."):
                genre_acc = recognize_directory(dset_path+"/"+genre,
                                noise=True,
                                snr=snr,
                                start=15,
                                end=15+duration)
                snr_accs.append(genre_acc)
        accs.append(np.mean(snr_accs))
    
    print(accs)
    if plot:
        plt.plot(snrs,accs)
        plt.title("Recognition Accuracy at Various Signal-Noise-Ratio Levels")
        plt.xlabel("Signal-to-Noise Ratio (dB)")
        plt.ylabel("Recognition Accurac")
        plt.gca().invert_xaxis()
        plt.show()

def main():
    #experiment1("processed_songs/dset100")
    #experiment2("processed_songs/dset100")
    for genre in os.listdir("processed_songs/dset100"):
        if not genre.startswith("."): 
            fingerprint_directory("processed_songs/dset100"+"/"+genre)
    pdb.set_trace()


if __name__=="__main__":
    start = time.time()
    main()
    print(time.time()-start)

# for i in range(-30,100,10):
#     n = get_noise(samples,i)
#     noisy_samples = samples+n
#     plt.plot(noisy_samples[:10000],label="noisy")
#     plt.plot(samples[:10000],label="original samples")
#     plt.title("signal to noise ratio = "+str(i))
#     plt.legend()
#     plt.show()