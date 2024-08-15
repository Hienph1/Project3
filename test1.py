#%%
import librosa

#%%
audio, sr = librosa.load('aug_data/test\down\down_802.wav',
                         sr =16000)

#%%
import matplotlib.pyplot as plt

#%%
import soundfile as sf

file_path = 'noise/noise14.wav'
signal, sr = sf.read(file_path)
signal, sr = sf.read(file_path, dtype='int16')

#%%
plt.plot(signal)
plt.grid()
plt.show()
# %%
