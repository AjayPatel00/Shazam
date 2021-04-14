from database import Fingerprint, Database
from utils import read_file
from fingerprint import get_spectrogram,get_peaks,gen_hashes

import numpy as np
import time
from itertools import groupby
from operator import itemgetter as ig
import pdb
import os
import sounddevice as sd

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
        time_offset = int(float(offset)/44100*4096*0.5)
        result.append({"song_id": song,
                       "total_hashes": myDB.song_table[song],
                       "input_hashes": len(hashes),
                       "matched_hashes": songs[song]})#,
                       #"input_confidence": round(songs[song]/len(hashes),2),
                       #"fingerprint_confidence": round(songs[song]/myDB.song_table[song],2),
                       #"offset_seconds":time_offset})
    return result

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

def main():
    fingerprint_directory("songs/dset6/songs")
    recognize_directory("songs/dset6/trimmed_songs")
    recognize_from_mic(fs=44100)

if __name__=="__main__":
    main()
