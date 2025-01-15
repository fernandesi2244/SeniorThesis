import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

from skimage.transform import resize
import tensorflow_datasets as tfds

import keras
from keras import layers

import io
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox

import sunpy.map


GLOBAL_MIN = -1500
GLOBAL_MAX = 1500

# Perform next-frame video prediction with Convolutional LSTM model
# TIFF files are stored in the /share/development/data/drms/MagPy_Shared_Data/LOSFullDiskMagnetogramTIFFFiles folder.
# File names are in the format of "hmi.m_720s.YYYYmmdd_HHMMSS_TAI.<digit>.tiff" as TIFF files.
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
        self.filepaths.sort() # Are we getting this in a contiguous sequence?

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
            sequence_data = [normalize(correct_nans(np.load(file))) for file in sequence_files]
            label_data = normalize(correct_nans(np.load(label_file)))
            batch_data.append(sequence_data)
            batch_labels.append(label_data)
        return np.array(batch_data), np.array(batch_labels)

    def on_epoch_end(self):
        pass

def build_model(sequence_length, width, height, channels):
    model = Sequential([
        ConvLSTM2D(filters=40, kernel_size=(3, 3), return_sequences=False, input_shape=(sequence_length, width, height, channels), padding='same', activation='tanh'),
        BatchNormalization(),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

directory = '/share/development/data/drms/MagPy_Shared_Data/LOSFullDiskMagnetogramNPYFiles'
sequence_length = 10  # Number of frames in each sequence
batch_size = 5
target_size = (256, 256)  # Adjust based on your dataset

filepaths = os.listdir(directory)
filepaths = [os.path.join(directory, filepath) for filepath in filepaths]
filepaths.sort()

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

import time

print('Beginning batch retrieval...')
start = time.time()
batch, labels = train_generator[0]
end = time.time()
print('Time taken:', end - start)
print('Batch shape:', batch.shape)
print('Labels shape:', labels.shape)

print('First sequence in batch:')
# Plot all frames in the first sequence in the batch
fig, axes = plt.subplots(1, sequence_length, figsize=(20, 20))
for i in range(sequence_length):
    axes[i].imshow(batch[0][i], cmap='gray')
    axes[i].axis('off')
plt.savefig('first_sequence_1st_batch.png')
plt.clf()

print('First label in batch:')
# Plot the label
plt.imshow(labels[0], cmap='gray')
plt.axis('off')
plt.savefig('first_label_1st_batch.png')
plt.clf()

print('Fifth sequence in batch:')
# Plot all frames in the fifth sequence in the batch
fig, axes = plt.subplots(1, sequence_length, figsize=(20, 20))
for i in range(sequence_length):
    axes[i].imshow(batch[4][i], cmap='gray')
    axes[i].axis('off')
plt.savefig('fifth_sequence_1st_batch.png')
plt.clf()

print('Fifth label in batch:')
# Plot the label
plt.imshow(labels[4], cmap='gray')
plt.axis('off')
plt.savefig('fifth_label_1st_batch.png')
plt.clf()


# Now get 5th batch
print('Beginning batch retrieval...')
start = time.time()
batch, labels = train_generator[4]
end = time.time()
print('Time taken:', end - start)

print('First sequence in batch:')

# Plot all frames in the first sequence in the batch
fig, axes = plt.subplots(1, sequence_length, figsize=(20, 20))
for i in range(sequence_length):
    axes[i].imshow(batch[0][i], cmap='gray')
    axes[i].axis('off')
plt.savefig('first_sequence_5th_batch.png')
plt.clf()

print('First label in batch:')
# Plot the label
plt.imshow(labels[0], cmap='gray')
plt.axis('off')
plt.savefig('first_label_5th_batch.png')
plt.clf()
