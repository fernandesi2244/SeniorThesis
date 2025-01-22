"""
Based on the model in NextFramePrediction/GetInputDataDistribution.py,
get all the magnetic field volumes from the training data and compute the
mean and standard deviation of the pixel values across the entire training dataset.
That is, compute only one mean and std across the aggregate of all pixels.
This will help inform how to normalize the data before feeding it into the model.
"""

import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sunpy.map
from skimage.transform import resize
import pathlib

rootDir = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(1, os.path.join(rootDir))

from Utils import load_volume_components

GENERATED_VOLUMES_PATH_SINGLE_BLOB = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumes'
GENERATED_VOLUMES_PATH_MULTI_BLOB = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumesMultiblob'
REGULAR_SHARED_DATA_DIR = os.path.join(os.sep + 'share', 'development', 'data', 'drms', 'MagPy_Shared_Data')
DEFINITIVE_SHARP_DATA_DIR = os.path.join(REGULAR_SHARED_DATA_DIR, 'TrainingData' + os.sep)

single_blob_volume_paths = [os.path.join(GENERATED_VOLUMES_PATH_SINGLE_BLOB, f) for f in os.listdir(GENERATED_VOLUMES_PATH_SINGLE_BLOB)]
multi_blob_volume_paths = [os.path.join(GENERATED_VOLUMES_PATH_MULTI_BLOB, f) for f in os.listdir(GENERATED_VOLUMES_PATH_MULTI_BLOB)]
volumes = single_blob_volume_paths + multi_blob_volume_paths
volumes.sort()

# TODO: Change the train set here once we fix it later in the actual data loading/training script.

# Use a train test split with a classic 80/20 split with a fixed seed for shuffling for reproducibility
train_volumes, test_volumes = train_test_split(volumes, test_size=0.2, random_state=42)

mean_x, mean_y, mean_z = 0, 0, 0
std_x, std_y, std_z = 0, 0, 0
num_pixels = 0

for volume_path in tqdm(train_volumes):
    # Load the volume, broadcast the bitmap mask across the height dimension,
    # and apply the mask to the volume. Then compute the mean and std from the
    # masked volume.
    target_ar_gen = os.path.basename(volume_path)[5:-4]

    bx_3D, by_3D, bz_3D = load_volume_components(volume_path)

    associated_bitmap_path = os.path.join(DEFINITIVE_SHARP_DATA_DIR, target_ar_gen + '.bitmap.fits')
    bitmap = sunpy.map.Map(associated_bitmap_path)
    
    bitmap_resized = resize(bitmap.data, (200, 400), anti_aliasing=True, preserve_range=True)
    mask_resized = bitmap_resized > 30
    blob_mask_resized = bitmap_resized*mask_resized.astype(int)*1.

    bx_3D_blob = np.transpose(bx_3D, (2, 0, 1))
    by_3D_blob = np.transpose(by_3D, (2, 0, 1))
    bz_3D_blob = np.transpose(bz_3D, (2, 0, 1))

    bx_3D_blob_masked = bx_3D_blob * mask_resized
    by_3D_blob_masked = by_3D_blob * mask_resized
    bz_3D_blob_masked = bz_3D_blob * mask_resized

    mean_x += np.sum(bx_3D_blob_masked)
    mean_y += np.sum(by_3D_blob_masked)
    mean_z += np.sum(bz_3D_blob_masked)

    std_x += np.sum(bx_3D_blob_masked ** 2)
    std_y += np.sum(by_3D_blob_masked ** 2)
    std_z += np.sum(bz_3D_blob_masked ** 2)

    num_pixels += np.sum(mask_resized)

mean_x /= num_pixels
mean_y /= num_pixels
mean_z /= num_pixels

biased_var_x = std_x / num_pixels - mean_x ** 2
biased_var_y = std_y / num_pixels - mean_y ** 2
biased_var_z = std_z / num_pixels - mean_z ** 2

unbiased_var_x = biased_var_x * num_pixels / (num_pixels - 1)
unbiased_var_y = biased_var_y * num_pixels / (num_pixels - 1)
unbiased_var_z = biased_var_z * num_pixels / (num_pixels - 1)

std_x = np.sqrt(unbiased_var_x)
std_y = np.sqrt(unbiased_var_y)
std_z = np.sqrt(unbiased_var_z)

print('Stats for magnetic field volume data:')
print('-' * 40)
print('Mean x:', mean_x)
print('Mean y:', mean_y)
print('Mean z:', mean_z)
print('Sample standard deviation x:', std_x)
print('Sample standard deviation y:', std_y)
print('Sample standard deviation z:', std_z)
print('Number of pixels:', num_pixels)
print('Number of volumes:', len(train_volumes))
