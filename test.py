class Time2MFCC(layers.Layer):
   def __init__(self, sample_rate=16000, window_size_ms=30, 
               dct_coefficient_count=10, window_stride_ms=20, 
               clip_duration_ms=1000):
      super().__init__()
      self.sample_rate = sample_rate
      self.window_size_ms = window_size_ms
      self.dct_coefficient_count = dct_coefficient_count
      self.window_stride_ms = window_stride_ms
      self.clip_duration_ms = clip_duration_ms
      
      self.desired_samples = int(sample_rate * clip_duration_ms / 1000)
      self.lower_edge_hertz, self.upper_edge_hertz, self.num_mel_bins = 20.0, 4000.0, 40 

      self.window_size_samples = int(self.sample_rate * self.window_size_ms / 1000)
      self.window_stride_samples = int(self.sample_rate * self.window_stride_ms / 1000)
      self.length_minus_window = (self.desired_samples - self.window_size_samples)
      if self.length_minus_window < 0:
         self.spectrogram_length = 0
      else:
         self.spectrogram_length = 1 + int(self.length_minus_window / self.window_stride_samples)
         self.fingerprint_size = dct_coefficient_count * self.spectrogram_length
      
   def call(self, input_audio):
      stfts = tf.signal.stft(input_audio, frame_length=self.window_size_samples, 
                              frame_step=self.window_stride_samples, fft_length=None,
                              window_fn=tf.signal.hann_window)
      spectrograms = tf.abs(stfts)
      spectrograms = tf.cast(spectrograms, tf.float32)
      num_spectrogram_bins = stfts.shape[-1]
      linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(self.num_mel_bins, 
                                                                           num_spectrogram_bins,
                                                                           self.sample_rate,
                                                                           self.lower_edge_hertz, 
                                                                           self.upper_edge_hertz)
      mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
      mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
      log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
      mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :self.dct_coefficient_count]
      mfccs = tf.reshape(mfccs,[self.spectrogram_length, self.dct_coefficient_count, 1])
      return mfccs