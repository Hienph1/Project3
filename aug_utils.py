#%%
import os
import librosa
import numpy as np
import soundfile as sf
#%%
def load_data(data_path):
   data = []
   label = []

   for folder_name in os.listdir(data_path):
      file_path = os.path.join(data_path, folder_name)
      for file_name in os.listdir(file_path):
         if file_name.endswith(('.wav')):
            path = os.path.join(file_path, file_name)
            signal, _ = librosa.load(path)
            data.append(signal)
            label.append(folder_name)
   return data, label

#%%
def find_spoken_word_indices(y, threshold=0.01):
   rms = librosa.feature.rms(y=y)[0]
   frames = np.nonzero(rms > threshold)[0]
   if len(frames) > 0:
      start = librosa.frames_to_samples(frames[0])
      end = librosa.frames_to_samples(frames[-1])
      return start, end
   else:
      return 0, len(y)

# %%
def adjust_length(signal, target_length):
   if len(signal) >= target_length:
      return signal[:target_length]
   else:
      repeats = target_length // len(signal) + 1
      extended_signal = np.tile(signal, repeats)
      return extended_signal[:target_length]
# %%
def mix_signals(signal1, signal2, noise_factor=0.5):
    return signal1 + noise_factor * signal2

#%%
def shift_data(signal, shift_max=0.3):
   
   start, end = find_spoken_word_indices(signal, threshold=0.01)
   right_range = int(shift_max * int(end - start) + len(signal[end:]))
   right_shift = np.roll(signal, right_range)
   right_shift[:right_range] = 0

   left_range = int(-shift_max * int(end - start) + len(signal[:start]))
   left_shift = np.roll(signal, left_range)
   left_shift[left_range:] = 0

   return right_shift, left_shift

def pad_or_truncate_centered(y, target_length=16000):
   start, end = find_spoken_word_indices(y)
   word_length = end - start
   
   if word_length > target_length:
      # If the spoken word itself is longer than the target length, truncate it
      start = (start + end - target_length) // 2
      end = start + target_length
      return y[start:end]
   
   # Center the spoken word in the target length frame
   center = (start + end) // 2
   half_target = target_length // 2
   start = max(0, center - half_target)
   end = start + target_length
   
   # Ensure we don't go out of bounds
   if end > len(y):
      start = max(0, len(y) - target_length)
      end = len(y)
   
   truncated = y[start:end]
   
   # Pad if necessary
   if len(truncated) < target_length:
      pad_length = target_length - len(truncated)
      truncated = np.pad(truncated, (0, pad_length), 'constant')
   
   return truncated

def load_noise_files(noise_folder, sr_speech):
   
   noise = []
   for path in os.listdir(noise_folder):
      noise_path = os.path.join(noise_folder, path)
      noise_signal, sr_noise = librosa.load(noise_path, sr=None)
      if sr_speech != sr_noise:
         noise_signal = librosa.resample(noise_signal, orig_sr=sr_noise, target_sr=sr_speech)
         sr_noise = sr_speech
      noise.append(noise_signal)
   noise = np.array(noise, dtype = 'object')
   return noise