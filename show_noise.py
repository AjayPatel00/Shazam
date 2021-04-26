from shazam import add_noise
from utils import read_file
import numpy as np
import matplotlib.pyplot as plt

p = "processed_songs/dset100/Country/Derek Clegg - Covid Blues.wav"

fs,samples = read_file(p)
samples = samples[fs*5:fs*6]
snrs = [40,25,10,-5]
noisy_samples = []
for s in snrs:
    noisy_sample = add_noise(s,samples)
    noisy_samples.append(noisy_sample)

fig,axs = plt.subplots(2,2)
fig.suptitle("Adding varying levels of noise to Covid Blues.wav ")
axs[0,0].plot(noisy_samples[0],label="SNR="+str(snrs[0]))

axs[0,1].plot(noisy_samples[1],label="SNR="+str(snrs[1]))
axs[1,0].plot(noisy_samples[2],label="SNR="+str(snrs[2]))
axs[1,1].plot(noisy_samples[3],label="SNR="+str(snrs[3]))
axs[0,0].legend(loc="upper right")
axs[0,1].legend(loc="upper right")
axs[1,0].legend(loc="upper right")
axs[1,1].legend(loc="upper right")
plt.show()