import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization
from sklearn.model_selection import train_test_split

import keras

import time

GLOBAL_MIN = -2500
GLOBAL_MAX = 2500

# Perform next-frame video prediction with Convolutional LSTM model
# NPY files are stored in the /share/development/data/drms/MagPy_Shared_Data/LOSFullDiskMagnetogramNPYFiles folder.
# File names are in the format of "hmi.m_720s.YYYYmmdd_HHMMSS_TAI.<digit>.npy" as npy files.
# There exists a file for each hour of the day. The goal is to take the previous n frames from n consecutive hours and
# predict the next frame at the next hour.

def correct_nans(image):
        image[np.isnan(image)] = 0
        return image

def normalize(image):
    return (image - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)

class ImageSequenceGenerator(tf.keras.utils.Sequence):
    def __init__(self, filepaths, sequence_length, batch_size, target_size):
        self.sequence_length = sequence_length # using sequence_length images to predict the next one
        self.batch_size = batch_size
        self.target_size = target_size
        self.filepaths = filepaths
        self.filepaths.sort()

    def __len__(self):
        # Number of batches we can make from the data
        return (len(self.filepaths) - self.sequence_length) // self.batch_size
    
    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = batch_start + self.batch_size + self.sequence_length # need 1 more for the label
        batch_filepaths = self.filepaths[batch_start:batch_end]

        batch_data = []
        batch_labels = []
        for i in range(0, self.batch_size):
            sequence_files = batch_filepaths[i:i+self.sequence_length]
            label_file = batch_filepaths[i+self.sequence_length]
            sequence_data = [np.expand_dims(normalize(correct_nans(np.load(file))), axis=-1) for file in sequence_files]
            label_data = np.expand_dims(normalize(correct_nans(np.load(label_file))), axis=-1)
            batch_data.append(sequence_data)
            batch_labels.append(label_data)
        return np.array(batch_data), np.array(batch_labels)

    def on_epoch_end(self):
        pass

def build_model(sequence_length, width, height, channels):
    # Build a ConvLSTM model that has a few ConvLSTM layers that take a sequence of images
    # and generate a single image as output. Remember that the output should not be a
    # sequence of images, but a single image.
    model = Sequential([
        tf.keras.layers.Input(shape=(sequence_length, width, height, channels)),
        ConvLSTM2D(filters=64, kernel_size=(5, 5), padding='same', activation='tanh', return_sequences=True),
        BatchNormalization(),
        ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', activation='tanh', return_sequences=True),
        BatchNormalization(),
        ConvLSTM2D(filters=64, kernel_size=(1, 1), padding='same', activation='tanh', return_sequences=False),
        # BatchNormalization(), # TODO: Test with and without this.
        tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

directory = 'npy_files_compressed/LOSFullDiskMagnetogramNPYFiles'
sequence_length = 10  # Number of frames in each sequence
batch_size = 5
target_size = (256, 256)  # Adjust based on your dataset

filepaths = os.listdir(directory)
filepaths = [os.path.join(directory, filepath) for filepath in filepaths]
filepaths.sort()
filepaths = filepaths[::4]

generator = ImageSequenceGenerator(filepaths, sequence_length, batch_size, target_size)

print('Number of batches overall:', len(generator))
print('Length of each batch:', batch_size)

# Split the filepaths into training and test sets, making sure to keep data within each set contiguous
train_filepaths, test_filepaths = train_test_split(filepaths, test_size=0.2, shuffle=False)

# Further split the training filepaths into training and validation sets, making sure to keep data within each set contiguous
train_filepaths, val_filepaths = train_test_split(train_filepaths, test_size=0.25, shuffle=False)

# Create separate generators for the training, validation, and test sets
train_generator = ImageSequenceGenerator(train_filepaths, sequence_length, batch_size, target_size)
val_generator = ImageSequenceGenerator(val_filepaths, sequence_length, batch_size, target_size)
test_generator = ImageSequenceGenerator(test_filepaths, sequence_length, batch_size, target_size)

# No channels in image, so assume greyscale.
model = build_model(sequence_length, target_size[0], target_size[1], 1)

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# Start timer
start = time.time()

model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[early_stopping, reduce_lr])

# Print the time that has elapsed during training
print('Training time:', time.time() - start)

# Evaluate the model using the test data
test_loss = model.evaluate(test_generator)
print('Test loss:', test_loss)

# Save the model
model.save('next_frame_prediction_3.keras')
