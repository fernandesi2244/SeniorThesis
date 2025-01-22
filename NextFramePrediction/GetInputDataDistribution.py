"""
Based on the model in NextFramePrediction/NextFramePrediction_FinalModel.py,
get all the magnetogram images from the training data and, after NaN correction
and the gaussian filter, compute the mean and standard deviation of the pixel values
across the entire training dataset. That is, compute only one mean and std across
the aggregate of all pixels. This will help inform how to normalize the data
before feeding it into the model.
"""

import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter

# Pulled from NextFramePrediction/NextFramePrediction_FinalModel.py
def correct_nans(image):
    image[np.isnan(image)] = 0
    return image

def apply_gaussian_filter(image):
    bitmap = gaussian_filter(abs(image), 48, order=0) > 32
    return image * bitmap


directory = '/share/development/data/drms/MagPy_Shared_Data/LOSFullDiskMagnetogramNPYFiles512'
filepaths = os.listdir(directory)
filepaths = [os.path.join(directory, filepath) for filepath in filepaths]
filepaths.sort()
filepaths = filepaths[::4] # TODO: Play with later

train_filepaths, test_filepaths = train_test_split(filepaths, test_size=0.2, shuffle=False)
train_filepaths, val_filepaths = train_test_split(train_filepaths, test_size=0.25, shuffle=False)

mean = 0
std = 0
num_pixels = 0

for filepath in tqdm(train_filepaths):
    data = np.load(filepath)
    data = apply_gaussian_filter(correct_nans(data))
    mean += np.sum(data)
    std += np.sum(data ** 2)
    num_pixels += data.size

mean /= num_pixels
biased_var = std / num_pixels - mean ** 2
unbiased_var = biased_var * num_pixels / (num_pixels - 1)
std = np.sqrt(unbiased_var)

print('Stats for full-disk magnetogram data:')
print('-' * 40)
print('Mean:', mean)
print('Sample standard deviation:', std)
print('Number of pixels:', num_pixels)
print('Number of magnetograms:', len(train_filepaths))
