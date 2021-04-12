from scipy import signal
from scipy.io import wavfile
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (binary_erosion,
                                      generate_binary_structure,
                                      iterate_structure)
from hashlib import sha1,sha256
from itertools import groupby
from operator import itemgetter as ig
from collections import Counter
import pdb

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
        samples = np.mean(samples,axis=1)
    #samples = samples[44100:20*44100]
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

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(spec,origin="lower", aspect="auto", cmap="jet", interpolation="none")
        ax.scatter(times_filter, freqs_filter,s=1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title("Spectrogram")
        plt.show()
    
    return list(zip(freqs_filter, times_filter))

# fan_val how many points you fan out to, higher fan val means more fingerprints
# lower means lower fingerprints
# returns hash,offset pairs
def gen_hashes(peaks,fan_val=5):
    hashes = []
    for i in range(len(peaks)):
        for j in range(1,fan_val):
            if(i+j)<len(peaks):
                f1,t1 = peaks[i][0],peaks[i][1]
                f2,t2 = peaks[i+j][0],peaks[i+j][1]
                t_prime = t2-t1
                if 0 <= t_prime <= 200:
                    s = str(f1)+"|"+str(f2)+"|"+str(t_prime)
                    h = sha256(s.encode('utf-8')).hexdigest()
                    hashes.append((h[0:20],t1))
    return hashes

class Fingerprint:
    def __init__(self,song_id,h,offset):
        self.song_id = song_id
        self.h = h
        self.offset = offset
    
class DB:
    def __init__(self):
        self.db = []
        self.song_table = {}

    def add(self,fingerprint):
        self.db.append(fingerprint)

    def search(self, recording_data):
        mapper = {}
        for h,offset in recording_data:
            if h in mapper.keys():
                mapper[h].append(offset)
            else:
                mapper[h] = [offset]

        recording_hashes = list(mapper.keys())
        songs = {}
        results = []
        for fingerprint in self.db:
            if fingerprint.h in recording_hashes:
                if fingerprint.song_id not in songs.keys():
                    songs[fingerprint.song_id] = 1
                else:
                    songs[fingerprint.song_id] += 1
                for recording_offset in mapper[fingerprint.h]:
                    results.append((fingerprint.song_id,fingerprint.offset - recording_offset))
        return results, songs

myDB = DB()

def fingerprint_song(song):
    fs,samples = read_file(song)
    spec = get_spectrogram(fs,samples,plot=False)
    peaks = get_peaks(spec,plot=False)
    hashes = gen_hashes(peaks)
    for h in hashes:
        f = Fingerprint(song,h[0],h[1])
        myDB.add(f)
    myDB.song_table[song] = len(hashes)

def recognize_recording(song):
    fs,samples = read_file(song)
    spec = get_spectrogram(fs,samples,plot=False)
    peaks = get_peaks(spec,plot=False)
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
                       "matched_hashes": songs[song],
                       "input_confidence": round(songs[song]/len(hashes),2),
                       "fingerprint_confidence": round(songs[song]/myDB.song_table[song],2),
                       "offset_seconds":time_offset})
    return result


songs = ["song.wav","song2.wav"]
for s in songs: fingerprint_song(s)
fs = 44100
results = recognize_recording("recording.wav")
pdb.set_trace()