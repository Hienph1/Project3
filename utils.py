import os
import numpy as np
import tensorflow as tf
import librosa

def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)

def get_label(file_path):
  parts = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-2]

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

# Slice into frames
def get_frames(amplitudes, window_length, hop_length):
  return librosa.util.frame(
      np.pad(amplitudes, int(window_length // 2), mode="reflect"),
      frame_length=window_length, hop_length=hop_length
  ) 

# STFT
def get_stft(amplitudes, window_length, hop_length):
  frames = get_frames(amplitudes, window_length, hop_length)
  fft_weights = librosa.filters.get_window(
      "hann", window_length, fftbins=True
  )
  stft = np.fft.rfft(frames * fft_weights[:, None], axis=0)
  return stft

# To mel-spectrum
def get_melspectrogram(amplitudes, sr=22050, 
                       n_mels=128, window_length=2048, 
                       hop_length=512, fmin=1, fmax=8192):

  stft = get_stft(amplitudes, window_length, hop_length)
  spectrogram = np.abs(stft ** 2)

  mel_basis = librosa.filters.mel(
      sr=sr, n_fft=window_length,
      n_mels=n_mels, fmin=fmin, fmax=fmax
  )

  mel_spectrogram = np.dot(mel_basis, spectrogram)
  return mel_spectrogram