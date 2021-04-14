import sounddevice as sd
import pdb

print("listening to microphone...")
recording = sd.rec(int(10*44100),samplerate=44100,channels=1)
sd.wait()
recording = recording.flatten()
print("done listening")
pdb.set_trace()


# p = pyaudio.PyAudio()

# stream = p.open(format=pyaudio.paInt16,
#                 channels=1,
#                 rate=44100,
#                 input=True,
#                 frames_per_buffer=4096)
