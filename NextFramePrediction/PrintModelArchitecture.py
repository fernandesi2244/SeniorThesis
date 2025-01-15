import os
import numpy as np
import tensorflow as tf


loaded_model = tf.keras.models.load_model('next_frame_prediction_2.keras')

# Print the model architecture
print(loaded_model.summary())

# Iterate through the model's layers
for layer in loaded_model.layers:
    if 'conv_lstm' in layer.name.lower():  # Check if it is a ConvLSTM layer
        print(f"Layer Name: {layer.name}")
        print(f"Kernel Size: {layer.kernel_size}")
        print(f"Filters: {layer.filters}")
        print(f"Strides: {layer.strides}")