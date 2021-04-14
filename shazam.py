from database import Fingerprint, Database
from utils import read_file
from fingerprint import get_spectrogram,get_peaks,gen_hashes

import numpy as np
import time
<<<<<<< HEAD
=======
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (binary_erosion,
                                      generate_binary_structure,
                                      iterate_structure)
from hashlib import sha1,sha256
>>>>>>> 730b4365c8cea2fe56b05229e1a975c2453fde56
from itertools import groupby
from operator import itemgetter as ig
import pdb
import os
import sounddevice as sd
<<<<<<< HEAD

=======
np.seterr(divide='ignore')
fs=44100


""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return np.transpose(newspec), freqs

# given wav file path, returns sample rate and samples in mono format
def read_file(path):
    fs,samples = wavfile.read(path)
    if samples.ndim >= 2:
        samples = np.mean(samples,axis=1)#*10e-6
    return fs,samples

def get_spectrogram(fs,samples,w_size=4096,window=mlab.window_hanning,overlap_ratio=0.5,plot=False):
    spec = mlab.specgram(
        samples,
        NFFT=w_size,
        Fs=fs,
        window=window,
        noverlap=int(w_size*overlap_ratio))[0]
    spec, freq = logscale_spec(np.transpose(spec), factor=1.0, sr=fs)
    spec = 20.*np.log10(np.abs(spec)/10e-6) # amplitude to decibel

    if plot:
        freqbins, timebins = np.shape(spec)
        plt.imshow(spec, origin="lower", aspect="auto", cmap="jet", interpolation="none")
        plt.colorbar()
        plt.xlabel("time (s)")
        plt.ylabel("frequency (hz)")
        plt.xlim([0, timebins-1])
        plt.ylim([0, freqbins])
        xlocs = np.float32(np.linspace(0, timebins-1, 5))
        plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(overlap_ratio*w_size))/fs])
        ylocs = np.int16(np.round(np.linspace(0,freqbins-1,10)))
        plt.yticks(ylocs, [int(freq[i]) for i in ylocs])
        plt.show()

    return spec

# high peak_nhood_size means less fingerprints, low means more fingerprints
# less fingerprints means faster matching and lower accuracy
# more fingerprints means slower matching and higher accuracy
def get_peaks(spec,peak_nhood_size=15,amp_min=10,plot=False):
    struct = generate_binary_structure(rank=2, connectivity=2) # square mask
    neighborhood = iterate_structure(struct, iterations=peak_nhood_size)
    local_max = maximum_filter(spec, footprint=neighborhood) == spec
    background = (spec == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max != eroded_background
    amps = spec[detected_peaks]
    freqs, times = np.where(detected_peaks)
    amps = amps.flatten()
    filter_idxs = np.where(amps > amp_min)
    freqs_filter = freqs[filter_idxs]
    times_filter = times[filter_idxs]
>>>>>>> 730b4365c8cea2fe56b05229e1a975c2453fde56

np.seterr(divide='ignore')
fs=44100
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

<<<<<<< HEAD
def fingerprint_directory(path):
    for s in os.listdir(path): fingerprint_song(path+"/"+s)

def recognize_recording(samples,fs):
    spec = get_spectrogram(fs,samples)
    peaks = get_peaks(spec,amp_min=4,peak_nhood_size=10)
=======
def recognize_recording(samples,fs):
    spec = get_spectrogram(fs,samples)
    peaks = get_peaks(spec,amp_min=4,peak_nhood_size=10,plot=True)
>>>>>>> 730b4365c8cea2fe56b05229e1a975c2453fde56
    hashes = gen_hashes(peaks)
    matches,songs = myDB.search(hashes)

    f = lambda x: (x[0],x[1])
    counts = [(*k,len(list(g))) for k,g in groupby(sorted(matches,key=f), key=f)]
    matches = [max(list(g),key=ig(2)) for k,g in groupby(counts, key=ig(0))]
    matches = sorted(matches,key=ig(2),reverse=True)

    result = []
    for song,offset,_ in matches[:5]:
        time_offset = int(float(offset)/44100*4096*0.5)
        result.append({"song_id": song,
                       "total_hashes": myDB.song_table[song],
                       "input_hashes": len(hashes),
                       "matched_hashes": songs[song]})#,
                       #"input_confidence": round(songs[song]/len(hashes),2),
                       #"fingerprint_confidence": round(songs[song]/myDB.song_table[song],2),
                       #"offset_seconds":time_offset})
    return result

<<<<<<< HEAD
def recognize_directory(path,limit=None):
    print("Recognizing all songs from ",path,"...\n")
    correct = 0
    for s in os.listdir(path):
        fs,samples = read_file(path+"/"+s)
        results = recognize_recording(samples,fs)
        matched_song = results[0]["song_id"]
        print("Recognizing song: ",s)
        print("Matched song: ",matched_song,"\n")
        if matched_song== s: correct += 1
    accuracy = (correct/len(os.listdir(path)))*100
    print("Shazam correctly identified",accuracy,"percent of songs in",path)

def recognize_from_mic(fs=44100,n_seconds=15):
=======
def fingerprint_songs():
    for s in os.listdir("songs"): fingerprint_song("songs/"+s)

def recognize_directory(path):
    for s in os.listdir(path):
        fs,samples = read_file(path"+s)
        results = recognize_recording(samples,fs)
        print("Recognizing song: ",s)
        print(results[0],"\n\n")

def recognize_from_mic(n_seconds=15):
>>>>>>> 730b4365c8cea2fe56b05229e1a975c2453fde56
    print("start playing music")
    time.sleep(5)
    print("microphone enabled")
    recording = sd.rec(int(n_seconds*fs),samplerate=fs,channels=1)
    sd.wait()
    print("microphone disabled, searching for matches ...")
<<<<<<< HEAD
    results = recognize_recording(recording.flatten(),fs)
    if len(results) == 0:
        print("Shazam found no matches. Try getting closer to source of music")
    else:
        matched_song = results[0]["song_id"]
        print("Shazam found the followign match: ",matched_song)

def main():
    fingerprint_directory("songs/dset6/songs")
    recognize_directory("songs/dset6/trimmed_songs")
    recognize_from_mic(fs=44100)

if __name__=="__main__":
    main()
=======
    results = recognize_recording(recording,fs)

fingerprint_songs()
recognize_directory("trimmed_songs")
recognize_from_mic()


# p = pyaudio.PyAudio()

# stream = p.open(format=pyaudio.paInt16,
#                 channels=1,
#                 rate=44100,
#                 input=True,
#                 frames_per_buffer=4096)
>>>>>>> 730b4365c8cea2fe56b05229e1a975c2453fde56
