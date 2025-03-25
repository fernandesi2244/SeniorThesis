import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D, LeakyReLU
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
import multiprocessing
from sklearn.model_selection import train_test_split
import keras
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

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

def apply_gaussian_filter(image):
    bitmap = gaussian_filter(abs(image), 48, order=0) > 32
    return image * bitmap

class ImageSequenceGenerator(tf.keras.utils.Sequence):
    def __init__(self, filepaths, sequence_length, batch_size, target_size, **kwargs):
        super().__init__(**kwargs)  # For multiprocessing parameters

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
            sequence_data = [np.expand_dims(normalize(apply_gaussian_filter(correct_nans(np.load(file)))), axis=-1) for file in sequence_files]
            label_data = np.expand_dims(normalize(apply_gaussian_filter(correct_nans(np.load(label_file)))), axis=-1)
            batch_data.append(sequence_data)
            batch_labels.append(label_data)
        return np.array(batch_data), np.array(batch_labels)

    def on_epoch_end(self):
        pass


directory = '/share/development/data/drms/MagPy_Shared_Data/LOSFullDiskMagnetogramNPYFiles512'
sequence_length = 10  # Number of frames in each sequence
batch_size = 5
target_size = (512, 512)  # Adjust based on your dataset

filepaths = os.listdir(directory)
filepaths = [os.path.join(directory, filepath) for filepath in filepaths]
filepaths.sort()
filepaths = filepaths[::4] # Was this used to train the most basic model? TODO: Increase the distance between the files to allow for lower-shot extrapolation.

cpus_to_use = max(int(multiprocessing.cpu_count() * 0.9), 1)
print('Using', cpus_to_use, 'CPUs.')

# Split the filepaths into training and test sets, making sure to keep data within each set contiguous
_, test_filepaths = train_test_split(filepaths, test_size=0.2, shuffle=False)

test_generator = ImageSequenceGenerator(test_filepaths, sequence_length, batch_size, target_size, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)

# Load the next_frame_prediction_.h5 model
loaded_model = tf.keras.models.load_model('next_frame_prediction_final.keras')

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

# Validate the model using the SSIM and PSNR metrics
ssims = []
psnrs = []

for i in range(len(test_generator)):
    print('Batch', i+1, 'of', len(test_generator))
    batch_data, batch_labels = test_generator[i]
    predictions = loaded_model.predict(batch_data)
    for j in range(len(predictions)):
        prediction = predictions[j, :, :, 0]
        label = batch_labels[j, :, :, 0]

        # Denoise the images, then use gaussian mask to only consider blobs, then mask out the values that are not the solar disk
        prediction = prediction.squeeze() * (GLOBAL_MAX - GLOBAL_MIN) + GLOBAL_MIN
        label = label.squeeze() * (GLOBAL_MAX - GLOBAL_MIN) + GLOBAL_MIN

        prediction = apply_gaussian_filter(prediction)
        label = apply_gaussian_filter(label)

        # NOTE: I think this should be redundant now that we are using the gaussian mask before this for blob detection
        prediction = prediction * mask
        label = label * mask

        ssim_val = ssim(prediction, label, data_range=1)
        psnr_val = psnr(prediction, label, data_range=1)

        ssims.append(ssim_val)
        psnrs.append(psnr_val)
    
print('Average SSIM:', np.mean(ssims))
print('Average PSNR:', np.mean(psnrs))
print('Standard deviation of SSIM:', np.std(ssims))
print('Standard deviation of PSNR:', np.std(psnrs))
print('Max SSIM:', np.max(ssims))
print('Max PSNR:', np.max(psnrs))
print('Min SSIM:', np.min(ssims))
print('Min PSNR:', np.min(psnrs))
print('Median SSIM:', np.median(ssims))
print('Median PSNR:', np.median(psnrs))
print('Number of comparisons', len(ssims))

# Plot the SSIM and PSNR values
plt.figure()
plt.hist(ssims, bins=50, alpha=0.5, label='SSIM')
plt.legend(loc='upper right')
plt.title('SSIM values for the test set')
plt.show()

# Save figure
if not os.path.exists('Next Frame Results/ConvLSTM Final'):
    os.makedirs('Next Frame Results/ConvLSTM Final')

plt.savefig('Next Frame Results/ConvLSTM Final/ssim_values.png')

plt.figure()
plt.hist(psnrs, bins=50, alpha=0.5, label='PSNR')
plt.legend(loc='upper right')
plt.title('PSNR values for the test set')
plt.show()

# Save figure
plt.savefig('Next Frame Results/ConvLSTM Final/psnr_values.png')

# Save the SSIM and PSNR values to numpy files
np.save('Next Frame Results/ConvLSTM Final/ssim_values.npy', ssims)
np.save('Next Frame Results/ConvLSTM Final/psnr_values.npy', psnrs)
