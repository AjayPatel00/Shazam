import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (binary_erosion,generate_binary_structure,iterate_structure)
from hashlib import sha1,sha256
np.seterr(divide='ignore')


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


# given sample frequency and samples, generate a spectrogram
# if plot is True, will plot a spectrogram
def get_spectrogram(fs,samples,w_size=4096,window=mlab.window_hanning,overlap_ratio=0.5,plot=False):
    # use matlab specgram to with hanning window of size 4096, and overlap ratio 0.5 
    spec = mlab.specgram(samples, NFFT=w_size, Fs=fs, window=window, noverlap=int(w_size*overlap_ratio))[0]
    # scale frequency axis logarithmically
    spec, freq = logscale_spec(np.transpose(spec), factor=1.0, sr=fs)
    # convert amplitude to decibal
    spec = 20.*np.log10(np.abs(spec)/10e-6)

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
        plt.yticks(ylocs, [int(freq[i]*2) for i in ylocs])
        plt.show()

    return spec

# this function returns peak points in the spectrogram
# high peak_nhood_size means less fingerprints, low means more fingerprints
# less fingerprints means faster matching and lower accuracy
# more fingerprints means slower matching and higher accuracy
# see report for more info on peak neighbourhood size
def get_peaks(spec,peak_nhood_size=15,amp_min=10,plot=False):
    # generate a square morphology mask will consider points in all directions around 
    # the origin point
    struct = generate_binary_structure(rank=2, connectivity=2) 
    # create an iterate structure object parametrized by peak neighbourhood size
    neighborhood = iterate_structure(struct, iterations=peak_nhood_size)
    # use scipy maximum filter to find the local maxima with specified neighbourhood size
    local_maxima = maximum_filter(spec, footprint=neighborhood) == spec
    bckgnd = binary_erosion((spec==0), structure=neighborhood, border_value=1)
    peaks = local_maxima != bckgnd
    amps = spec[peaks]
    # get locations of where the local maxima were found
    freqs, times = np.where(peaks)
    amps = amps.flatten()
    # make sure that the peaks we found are above min amplitude
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


# Returns hash,offset pairs given peak points and fan_val
# fan_val how many points you fan out to, higher fan val means more fingerprints
# lower means lower fingerprints
def gen_hashes(peaks,fan_val=5):
    hashes = []
    for i in range(len(peaks)):
        # generate fan_val pairs
        for j in range(1,fan_val):
            if(i+j)<len(peaks):
                # compute the time offset between two peak points
                t_prime = peaks[i+j][1]-peaks[i][1]
                # if distance between two peak points is positive and 
                # less than 200
                if 0 <= t_prime <= 200:
                    # concatenate t1|t2|t2-t1
                    s = str(peaks[i][0])+"|"+str(peaks[i+j][0])+"|"+str(t_prime)
                    # generate sha256 hash
                    h = sha256(s.encode('utf-8')).hexdigest()
                    hashes.append((h[0:20],peaks[i][1]))
    return hashes
