#%%
import os
import random
import librosa
import numpy as np
import soundfile as sf
from aug_utils import *
#%%
# Find start and end time of spoken word
# Load the audio file
# file_path = 'C:\VSCode\Project2\data\mini_speech_commands\down/0a9f9af7_nohash_0.wav'
# y, sr = librosa.load(file_path)
#%%
# Compute the Short-Time Fourier Transform (STFT)
# hop_length = 512
# win_length = 2048
# S = np.abs(librosa.stft(y, n_fft=win_length, hop_length=hop_length))

# Compute the root-mean-square (RMS) energy for each frame
# rms = librosa.feature.rms(S=S)[0]

# Threshold the RMS values to find significant peaks
# threshold = 0.01  # This value might need tuning based on your audio
# frames = np.nonzero(rms > threshold)
# indices = librosa.core.frames_to_samples(frames)[0]

# Get the start and end of the spoken word
# start_index = indices[0]
# end_index = indices[-1]

# Convert to time
# start_time = librosa.samples_to_time(start_index, sr=sr)
# end_time = librosa.samples_to_time(end_index, sr=sr)

# print(f"Start time: {start_time:.2f} seconds")
# print(f"End time: {end_time:.2f} seconds")
#%%
# Optional: Plot the waveform with the detected word boundaries
# plt.figure(figsize=(14, 5))
# librosa.display.waveshow(y, sr=sr)
# plt.axvline(x=start_time, color='r', linestyle='--')
# plt.axvline(x=end_time, color='r', linestyle='--')
# plt.title('Word Boundaries')
# plt.show()

#%%
# Load noise to mix with speech audio files
sr_speech = 16000
noise_folder = 'noise'
noise = load_noise_files(noise_folder, sr_speech)
# %%
# Function to augment data
def data_augmentation(data_path, noise):

   for folder_name in os.listdir(data_path):
      index = 0
      noise_idx = 0
      folder_dir = os.path.join(data_path, folder_name)
      num_files = len(os.listdir(folder_dir))
      for file_name in os.listdir(folder_dir):
         if file_name.endswith(('.wav')):
            path = os.path.join(folder_dir, file_name)
         
         if index <= int(0.8*num_files):
            saved_path = 'data/augmented_data/train/'
         else:
            saved_path = 'data/augmented_data/test/'

         if index % 100 == 0:
            _noise = noise[noise_idx]
            _noise = adjust_length(_noise, 16000)
            noise_idx = (noise_idx + 1) % len(noise)

         signal, sr = librosa.load(path)
         signal = pad_or_truncate_centered(signal, 16000)

         mixed_signal = mix_signals(signal, _noise, noise_factor=0.5)
         mixed_signal = np.clip(mixed_signal, -1.0, 1.0)
         signal = np.array(signal)
         right_shifted, left_shifted = shift_data(signal, shift_max=0.3)

         name = saved_path + folder_name + '/audio_' + str(index)
         sf.write(name + '_mix_' + str(noise_idx) + '.wav', mixed_signal, sr)
         sf.write(name + '_shift_right.wav', right_shifted, sr)
         sf.write(name + '_shift_left.wav', left_shifted,sr)
         index += 1

   print(f"Finish augmenting class: {folder_name}")

# %%
# Do augment data
data_path = 'data/speech_commands'
data_augmentation(data_path, noise)

#%%
# from zipfile import ZipFile

# with ZipFile('Zipped.zip', 'r') as zipObj:
#    zipObj.extractall('data/speech_commands')

#%%
folder_path = 'speech_commands/up'

all_files = os.listdir(folder_path)

random.shuffle(all_files)

for file_name in all_files[200:]:
   file_path = os.path.join(folder_path, file_name)
   os.remove(file_path)
# %%
# folder_r = 'data/speech_commands'
# folder_s = 'data/augmented_data/test'

# for folder_name in os.listdir(folder_r):
#    new_folder = os.path.join(folder_s, folder_name)
#    try:
#       os.mkdir(new_folder)
#       print(f"New folder created: {new_folder}")
#    except:
#       print(f"The folder '{new_folder}' already exists.")
#%%
# import zipfile

# def zip_folder(folder_path, zip_file_path):
#     with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
#         for root, dirs, files in os.walk(folder_path):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 zip_file.write(file_path)

# zip_folder('mfcc_data', 'mfcc_data.zip')

#%%
import os
import random
import shutil

source_dir = 'aug_data/train'
target_dir = 'ei_data'

for folder_name in os.listdir(source_dir):
    source_folder = os.path.join(source_dir, folder_name)
    target_folder = os.path.join(target_dir, folder_name)
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    all_files = os.listdir(source_folder)
    random.shuffle(all_files)
    
    num_files_to_copy = 2000 if folder_name == 'unknown' else 900
    
    for file_name in all_files[:num_files_to_copy]:
        source_file = os.path.join(source_folder, file_name)
        target_file = os.path.join(target_folder, file_name)
        shutil.copy2(source_file, target_file)
        
    print(f"Copied {num_files_to_copy} files from {source_folder} to {target_folder}")

print("File copying completed.")

#%%
source_dir = 'aug_data/test'

for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                old_path = os.path.join(folder_path, file_name)
                new_file_name = file_name.replace('audio_', f'{folder_name}_')
                new_path = os.path.join(folder_path, new_file_name)
                
                os.rename(old_path, new_path)
        
        print(f"Renamed files in {folder_name} folder")

print("File renaming completed.")

# %%
import tensorflow as tf

def find_mfcc(input_audio, sample_rate=16000, window_size_ms=30, 
                 dct_coefficient_count=10, window_stride_ms=20, 
                 clip_duration_ms=1000):
    
   lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40
   desired_samples = int(sample_rate * clip_duration_ms / 1000)
   window_size_samples = int(sample_rate * window_size_ms / 1000)
   window_stride_samples = int(sample_rate * window_stride_ms / 1000)
   length_minus_window = (desired_samples - window_size_samples)
   if length_minus_window < 0:
      spectrogram_length = 0
   else:
      spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
      fingerprint_size = dct_coefficient_count * spectrogram_length
   
   stfts = tf.signal.stft(input_audio, frame_length=window_size_samples, 
                        frame_step=window_stride_samples, fft_length=None,
                        window_fn=tf.signal.hann_window)
   spectrograms = tf.abs(stfts)
   spectrograms = tf.cast(spectrograms, tf.float32)
   num_spectrogram_bins = stfts.shape[-1]
   linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, 
                                                                     num_spectrogram_bins,
                                                                     sample_rate,
                                                                     lower_edge_hertz, 
                                                                     upper_edge_hertz)
   mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
   mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
   log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
   mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :dct_coefficient_count]
   mfccs = tf.reshape(mfccs,[spectrogram_length, dct_coefficient_count])
   return mfccs

#%%
import os
import numpy as np
import librosa
import h5py

source_folder = 'speech_commands'
target_folder = 'mfcc_data'

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

for file_name in os.listdir(source_folder):
    source_file = os.path.join(source_folder, file_name)
    audio, sample_rate = librosa.load(source_file, sr=16000)
    audio = pad_or_truncate_centered(audio, 16000)
    mfcc_data = find_mfcc(audio)
    target_file = os.path.join(target_folder, os.path.splitext(file_name)[0] + '.hdf5')
    with h5py.File(target_file, 'w') as f:
        f.create_dataset('mfcc', data=mfcc_data)

print(f"Computed and saved MFCC data for files in {source_folder} to {target_folder}")
# %%
import h5py
import librosa
import matplotlib.pyplot as plt

# Load an HDF5 file from the mfcc_data folder
file_path = 'mfcc_data\\0a2b400e_nohash_1.hdf5'
with h5py.File(file_path, 'r') as f:
    mfcc_data = f['mfcc'][:]

# Plot the MFCC data using librosa.display.specshow
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc_data, sr=16000)
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.tight_layout()
plt.show()

# %%
plt.imshow(mfcc_data, aspect='auto')
plt.show()
# %%
