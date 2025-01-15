import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization
from sklearn.model_selection import train_test_split

from skimage.transform import resize

import keras
import sys
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


directory = 'npy_files_compressed_/LOSFullDiskMagnetogramNPYFiles'
sequence_length = 10  # Number of frames in each sequence
batch_size = 5
target_size = (256, 256)  # Adjust based on your dataset

filepaths = os.listdir(directory)
filepaths = [os.path.join(directory, filepath) for filepath in filepaths]
filepaths.sort()
filepaths = filepaths[::8]

# Split the filepaths into training and test sets, making sure to keep data within each set contiguous
_, test_filepaths = train_test_split(filepaths, test_size=0.2, shuffle=False)

test_generator = ImageSequenceGenerator(test_filepaths, sequence_length, batch_size, target_size)

# Load the next_frame_prediction_.h5 model
loaded_model = tf.keras.models.load_model('next_frame_prediction_64_32_16_every_8.keras')

# Mask out the values that are not the solar disk
width = 4096
height = 4096
center_x = width // 2
center_y = height // 2

angular_radius_of_disk = 955.382568 # arcsec
radius_in_pixels = angular_radius_of_disk / 0.504039

Y, X = np.ogrid[:height, :width]
dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

mask = dist_from_center <= radius_in_pixels

# Resize the mask to the target size
mask = resize(mask, target_size, anti_aliasing=False)

# Take the batch, labels, predictions, and sequence number as input and plot the sequence, predicted frame, and difference.
def plot_sequence_and_predicted_frame(batch, labels, predicted_frames, batch_num, sequence_num):
    fig, axes = plt.subplots(2, 6, figsize=(20, 20))
    axes = np.ravel(axes)

    for i in range(sequence_length):
        frame_denormalized = batch[sequence_num][i].squeeze() * (GLOBAL_MAX - GLOBAL_MIN) + GLOBAL_MIN
        frame_denormalized[~mask] = np.nan
        # Horizontally flip the image so it is displayed correctly
        frame_denormalized = np.fliplr(frame_denormalized)
        axes[i].imshow(frame_denormalized, cmap='gray', vmin=-300, vmax=300)
        # Draw dotted gridlines over the plot so we can see the spatial resolution
        axes[i].set_xticks(np.arange(0, 256, 25), minor=True)
        axes[i].set_yticks(np.arange(0, 256, 25), minor=True)
        axes[i].grid(which='both', color='w', linestyle=':', linewidth=0.2)
        axes[i].set_title(f'Frame {i + 1}')

    predicted_frame_denormalized = predicted_frames[sequence_num].squeeze() * (GLOBAL_MAX - GLOBAL_MIN) + GLOBAL_MIN
    predicted_frame_denormalized[~mask] = np.nan
    predicted_frame_denormalized = np.fliplr(predicted_frame_denormalized)
    axes[sequence_length].imshow(predicted_frame_denormalized, cmap='gray', vmin=-300, vmax=300)
    axes[sequence_length].set_xticks(np.arange(0, 256, 25), minor=True)
    axes[sequence_length].set_yticks(np.arange(0, 256, 25), minor=True)
    axes[sequence_length].grid(which='both', color='w', linestyle=':', linewidth=0.2)
    axes[sequence_length].set_title('Predicted Frame')

    difference = (predicted_frames[sequence_num].squeeze() - labels[sequence_num].squeeze()) * (GLOBAL_MAX - GLOBAL_MIN)
    difference[~mask] = np.nan
    difference = np.fliplr(difference)
    im = axes[sequence_length + 1].imshow(difference, cmap='RdBu')
    axes[sequence_length + 1].set_xticks(np.arange(0, 256, 25), minor=True)
    axes[sequence_length + 1].set_yticks(np.arange(0, 256, 25), minor=True)
    axes[sequence_length + 1].grid(which='both', color='w', linestyle=':', linewidth=0.2)
    axes[sequence_length + 1].set_title('Predicted - Actual')

    cbar = fig.colorbar(im, ax=axes[sequence_length + 1], fraction=0.046, pad=0.04)

    axes = np.reshape(axes, (2, 6))

    plt.tight_layout()

    plt.savefig(f'Next Frame Results/ConvLSTM 3 (64-32-16 every 8)/batch_{batch_num}_sequence_{sequence_num}_and_predicted_frame_and_difference.png')

    plt.clf()
    plt.close()

# Predict the next frame for the first sequence in the first batch of the test data
first_batch, first_labels = test_generator[0]
predicted_frames = loaded_model.predict(first_batch)

# Plot the first sequence of the batch and its next predicted frame, 5 in first row and 6 in second row
plot_sequence_and_predicted_frame(first_batch, first_labels, predicted_frames, 0, 0)


third_batch, third_labels = test_generator[2]
predicted_frames = loaded_model.predict(third_batch)
plot_sequence_and_predicted_frame(third_batch, third_labels, predicted_frames, 2, 0)

fifth_batch, fifth_labels = test_generator[4]
predicted_frames = loaded_model.predict(fifth_batch)
plot_sequence_and_predicted_frame(fifth_batch, fifth_labels, predicted_frames, 4, 0)


