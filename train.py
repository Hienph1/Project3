#%%
import os 
import numpy as np
from modules.aug_utils import pad_or_truncate_centered
import librosa
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Layer, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import Input

#%%
# Load Dataset
signals = []
labels = []
data_path = 'data/speech_commands'
for idx, folder_name in enumerate(os.listdir(data_path)):
   folder_dir = os.path.join(data_path, folder_name)

   for file_name in os.listdir(folder_dir):
      if file_name.endswith(('.wav')):
         path = os.path.join(folder_dir, file_name)

      signal, sr = librosa.load(path)
      if 16000 != sr:
         signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
         sr = 16000
      signal = pad_or_truncate_centered(signal, 16000)
      signals.append(signal)
      labels.append(idx)

signals = np.array(signals)
labels = np.array(labels)

# %%
def compute_spectrogram(waveform):
   spectrogram = tf.signal.stft(waveform, frame_length=16, frame_step=330,
                                fft_length=None, window_fn=tf.signal.hann_window)
   spectrogram = tf.abs(spectrogram)
   return spectrogram

class SpectrogramLayer(Layer):
    def __init__(self):
        super(SpectrogramLayer, self).__init__()

    def call(self, inputs):
        output = tf.map_fn(compute_spectrogram, inputs, dtype=tf.float32)
        return tf.expand_dims(output, axis=-1)

#%%
def create_model():
   input_layer = Input(shape=(16000,))
   spec_layer = SpectrogramLayer()(input_layer)
   spec_layer = Reshape((49, 9, 1))(spec_layer)  # Reshape to add channel dimension

   x = Conv2D(16, (3, 3), padding='same', activation='relu')(spec_layer)
   x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
   x = MaxPooling2D()(x)
   x = Conv2D(48, (3, 3), padding='same', activation='relu')(x)
   x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
   x = MaxPooling2D()(x)

   flatten_layer = Flatten()(x)
   dense_layer = Dense(12, activation='softmax')(flatten_layer)
   model = Model(inputs=input_layer, outputs=dense_layer)
   return model

#%%
model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

#%%
# Prepare the dataset
from sklearn.model_selection import train_test_split

# %%
x_train, x_test, y_train, y_test = train_test_split(signals, labels, test_size=0.2)
#%%
# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.2)

# %%
# Save the model
# model.save('my_model_mfcc')

#%%
model.evaluate(x_test, y_test)

# %%
# Load the saved model
model = tf.keras.models.load_model('my_model_mfcc')

# %%
# Convert the model to a TensorFlow Lite model with TF Select enabled
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops.
]
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# %%
# Apply dynamic range quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the quantized model
with open('model_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)

# %%
# Define a representative dataset function
def representative_data_gen():
   for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
      yield [input_value]

# %%
# Apply full integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops for quantization
]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()

#%%
# Save the quantized model
with open('model_quant_int8.tflite', 'wb') as f:
   f.write(tflite_quant_model)

#%%
# Load the quantized model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# %%
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# %%
# Prepare the test data
x_test = x_test.astype('float32')

# %%
# Function to run inference on the test data
def run_inference(interpreter, input_data):
   input_index = input_details[0]['index']
   output_index = output_details[0]['index']
   
   interpreter.set_tensor(input_index, input_data)
   interpreter.invoke()
   output_data = interpreter.get_tensor(output_index)
   return output_data

# %%
# Run inference on the test set
predictions = []
for i in range(len(x_test)):
   input_data = np.expand_dims(x_test[i], axis=0)  # Add batch dimension
   output_data = run_inference(interpreter, input_data)
   predictions.append(np.argmax(output_data))

predictions = np.array(predictions)

# %%
from sklearn.metrics import accuracy_score
# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
# %%
