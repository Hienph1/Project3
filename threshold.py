# Import independent libraries
#%%
import numpy as np
import librosa 
import matplotlib.pyplot as plt

# Fixed Thresholding
#%%
def fixed_threshold(signal, threshold):
    detections = signal > threshold
    return detections

# Example usage
signal, sr = librosa.load('test.wav', sr=16000)
threshold = 0.01 # Threshold value
detections = fixed_threshold(signal, threshold)
print(detections)

plt.figure(figsize=(14, 6))
plt.plot(signal, label='Audio Signal')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(np.where(detections)[0], signal[detections], color='r', marker='o', label='Detections')
plt.title('Audio Signal with Fixed Threshold')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Adaptive Thresholding
#%%
def adaptive_threshold(signal, window_size, factor):
    thresholds = np.zeros_like(signal)
    for i in range(len(signal)):
        start = max(0, i - window_size)
        end = min(len(signal), i + window_size)
        local_mean = np.mean(signal[start:end])
        thresholds[i] = local_mean * factor
    detections = signal > thresholds
    return thresholds, detections

# Parameters for adaptive thresholding
window_size = 1000  # Adjust this value as needed
factor = 1.2  # Adjust this value as needed

# Apply adaptive thresholding
thresholds, detections = adaptive_threshold(signal, window_size, factor)

# Plot the entire signal with the adaptive threshold
plt.figure(figsize=(14, 6))
plt.plot(signal, label='Audio Signal')
plt.plot(thresholds, color='r', linestyle='--', label='Adaptive Threshold')
plt.scatter(np.where(detections)[0], signal[detections], color='r', marker='o', label='Detections')
plt.title('Audio Signal with Adaptive Threshold')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Energy-Based Detection
#%%
def energy_based_detection(signal, window_size, energy_threshold):
    num_windows = int(np.ceil(len(signal) / window_size))
    energies = np.zeros(num_windows)
    for i in range(num_windows):
        start = i * window_size
        end = min(start + window_size, len(signal))
        window = signal[start:end]
        energy = np.sum(window ** 2)
        energies[i] = energy
    detections = energies > energy_threshold
    return energies, detections

# Parameters for energy-based detection
window_size = 1000  # Adjust this value as needed
energy_threshold = 0.7  # Adjust this value as needed

# Apply energy-based detection
energies, detections = energy_based_detection(signal, window_size, energy_threshold)

# Create a time array for plotting
time = np.arange(len(signal)) / sr

# Plot the entire signal with the energy-based threshold
plt.figure(figsize=(14, 6))
plt.plot(time, signal, label='Audio Signal')
for i, detection in enumerate(detections):
    if detection:
        plt.axvspan(i * window_size / sr, (i + 1) * window_size / sr, color='r', alpha=0.3, label='Detection' if i == 0 else "")
plt.title('Audio Signal with Energy-Based Detection')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Statistical Methods
#%%
# def statistical_detection(signal, window_size, z_threshold):
#     num_windows = int(np.ceil(len(signal) / window_size))
#     z_scores = np.zeros(num_windows)
#     for i in range(num_windows):
#         start = i * window_size
#         end = min(start + window_size, len(signal))
#         window = signal[start:end]
#         mean = np.mean(window)
#         std_dev = np.std(window)
#         z_scores[i] = (np.mean(window ** 2) - mean) / std_dev if std_dev != 0 else 0
#     detections = np.abs(z_scores) > z_threshold
#     return z_scores, detections

# # Parameters for statistical detection
# window_size = 1000  # Adjust this value as needed
# z_threshold = 0.15  # Z-score threshold for detection

# # Apply statistical detection
# z_scores, detections = statistical_detection(signal, window_size, z_threshold)

# # Create a time array for plotting
# time = np.arange(len(signal)) / sr
# z_time = np.arange(len(z_scores)) * window_size / sr

# # Plot the entire signal with the statistical detection
# plt.figure(figsize=(14, 6))

# # Plot the audio signal
# plt.subplot(2, 1, 1)
# plt.plot(time, signal, label='Audio Signal')
# plt.title('Audio Signal with Statistical Detection (Z-score)')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Amplitude')
# plt.legend()

# # Plot the Z-scores and threshold
# plt.subplot(2, 1, 2)
# plt.plot(z_time, z_scores, label='Z-scores')
# plt.axhline(y=z_threshold, color='r', linestyle='--', label='Threshold')
# plt.axhline(y=-z_threshold, color='r', linestyle='--')
# for i, detection in enumerate(detections):
#     if detection:
#         plt.axvspan(i * window_size / sr, (i + 1) * window_size / sr, color='r', alpha=0.3, label='Detection' if i == 0 else "")
# plt.title('Z-scores and Detection Threshold')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Z-score')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Machine Learning
# #%%
# from sklearn.tree import DecisionTreeClassifier

# def machine_learning_detection(signal, labels):
#     classifier = DecisionTreeClassifier()
#     classifier.fit(signal.reshape(-1, 1), labels)
#     predictions = classifier.predict(signal.reshape(-1, 1))
#     return predictions

# # Example usage
# signal = np.array([0.1, 0.5, 0.3, 0.7, 0.2])
# labels = np.array([0, 1, 0, 1, 0])  # Example labels: 1 for signal, 0 for noise
# detections = machine_learning_detection(signal, labels)
# print(detections)
# %%
