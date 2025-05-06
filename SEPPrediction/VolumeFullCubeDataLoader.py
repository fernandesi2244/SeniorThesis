import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import sunpy.map
import pathlib
import tensorflow as tf
from scipy.ndimage import label, generate_binary_structure
from sklearn.preprocessing import StandardScaler
import time
import multiprocessing
import datetime
from VolumeSlicesAndCubeDataLoader import PrimarySEPInputDataGenerator

rootDir = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(1, os.path.join(rootDir))

from Utils import load_volume_components

GENERATED_VOLUMES_PATH_SINGLE_BLOB = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumes'
GENERATED_VOLUMES_PATH_MULTI_BLOB = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumesMultiblob'
REGULAR_SHARED_DATA_DIR = os.path.join(os.sep + 'share', 'development', 'data', 'drms', 'MagPy_Shared_Data')
DEFINITIVE_SHARP_DATA_DIR = os.path.join(REGULAR_SHARED_DATA_DIR, 'TrainingData' + os.sep)

# The only point of this generator is to add the 2D/3D volume data to the records produced by the above generator.
class SecondarySEPInputDataGenerator(tf.keras.utils.Sequence):
    BLOB_VECTOR_COLUMNS_GENERAL = ['Latitude', 'Carrington Longitude', 'Is Plage', 'Stonyhurst Longitude']
    BLOB_ONE_TIME_INFO = ['Number of Recent Flares', 'Max Class Type of Recent Flares', 'Number of Recent CMEs', 'Max Product of Half Angle and Speed of Recent CMEs', 'Number of Sunspots', 'Max Flare Peak of Recent Flares', 'Min Temperature of Recent Flares', 'Median Emission Measure of Recent Flares', 'Median Duration of Recent Flares', 'Number of Recent SEPs', 'Number of Recent Subthreshold SEPs']

    TOP_N_BLOBS = 5

    def __init__(self, primary_arr, batch_size, shuffle, granularity, **kwargs):
        super().__init__(**kwargs)  # For multiprocessing parameters

        self.primary_arr = primary_arr
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.granularity = granularity

        self.indexes = np.arange(len(self.primary_arr))

        # Set random seed for comparison purposes since some of the models we're training
        # will take an extremely long time to train. This will allow us to compare results without
        # having to average over multiple runs. This is not ideal, but it's a temporary solution until
        # we are possibly able to distill the models or use a more efficient training mechanism.
        np.random.seed(42)

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.indexes) // self.batch_size

    # NOTE: NOT THE CASE ANYMORE: No volume loading so client using this class can probably use large batch size.
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        if self.granularity == 'per-blob':
            batch_rows = self.primary_arr[batch_indexes]

            # Iterate over the selected rows and load their corresponding data
            x_data = []
            y_data = []

            for row in batch_rows:
                overall_data_vector = []

                features = row[:-3]
                filename_general = row[-3]
                blob_index = row[-2]

                overall_data_vector.extend(features)

                # Bout_hmi.sharp_cea_720s.10960.20240314_170000_TAI.bin
                volume_path_single_blob = os.path.join(GENERATED_VOLUMES_PATH_SINGLE_BLOB, f'Bout_{filename_general}.bin')
                volume_path_multi_blob = os.path.join(GENERATED_VOLUMES_PATH_MULTI_BLOB, f'Bout_{filename_general}.bin')

                if os.path.exists(volume_path_single_blob):
                    bx_3D, by_3D, bz_3D = load_volume_components(volume_path_single_blob)
                else:
                    bx_3D, by_3D, bz_3D = load_volume_components(volume_path_multi_blob)
                
                # Fetch the corresponding bitmap so we an mask the bx, by, and bz fields
                associated_bitmap_path = os.path.join(DEFINITIVE_SHARP_DATA_DIR, filename_general + '.bitmap.fits')
                bitmap = sunpy.map.Map(associated_bitmap_path)

                # Resize bitmap data to 200x400
                bitmap_resized = resize(bitmap.data, (200, 400), anti_aliasing=True, preserve_range=True)

                # weak and strong field pixels within the HARP = (33, 34). A little bit of error allowed here due to
                # interpolation from resizing function. The next lowest values in the bitmap are low enough that we
                # can easily look from 30 and up.
                mask_resized = bitmap_resized > 30
                blob_mask_resized = bitmap_resized*mask_resized.astype(int)*1.

                # Separate out blobs
                s = generate_binary_structure(2,2)  # Allows diagonal pixels to be considered part of the same blob

                labeled_resized, nblobs_resized = label(blob_mask_resized, structure=s)

                # Sort blobs in order from greatest area to least area (relevant to AR # identification in MagPy, so needed for cross-referencing)
                # The assumption is that resizing the bitmap won't change the relative sizes of the blobs to each other.
                blobs_resized = [i for i in range(1, nblobs_resized+1)]
                blobs_resized = sorted(blobs_resized, key=lambda x: np.count_nonzero(labeled_resized == x), reverse=True)

                curr_blob = blobs_resized[blob_index - 1]

                # Create a new volume for the blob. That is, mask out ~blob pixels at every height of the volume, which is in the resized scale.
                mask = labeled_resized == curr_blob

                # In order to multiply each 3D component by the mask, we need to make the first dimension of the volume the height.
                # This is because the mask has the same shape as the volume at a given height, and broadcasting only works on the
                # first dimension.
                bx_3D_blob = np.transpose(bx_3D, (2, 0, 1)) # height dimension, then number of rows, then number of cols
                by_3D_blob = np.transpose(by_3D, (2, 0, 1))
                bz_3D_blob = np.transpose(bz_3D, (2, 0, 1))

                bx_3D_blob = bx_3D_blob * mask
                by_3D_blob = by_3D_blob * mask
                bz_3D_blob = bz_3D_blob * mask

                # Transpose back to original shape
                bx_3D_blob = np.transpose(bx_3D_blob, (1, 2, 0))
                by_3D_blob = np.transpose(by_3D_blob, (1, 2, 0))
                bz_3D_blob = np.transpose(bz_3D_blob, (1, 2, 0))

                # Reshape the bx_3D, by_3D, and bz_3D arrays to one array with shape (200, 400, 100, 3),
                # corresponding to the length, width, height and number of channels. Then flatten the cube.
                cube = np.zeros((200, 400, 100, 3))
                cube[:, :, :, 0] = bx_3D_blob
                cube[:, :, :, 1] = by_3D_blob
                cube[:, :, :, 2] = bz_3D_blob

                # Flatten and add to the overall data vector.
                overall_data_vector.extend(cube.flatten())
                
                # overall_data_vector = np.array(overall_data_vector)
                overall_data_vector = np.array(overall_data_vector, dtype=float)

                x_data.append(overall_data_vector)

                # Class label for the blob, which is just the SEP event rate for the blob at the current timestep
                produced_SEP = int(row[-1])
                y_data.append(produced_SEP)

            return np.array(x_data), np.array(y_data)
        
        # At this point, we're assuming that the granularity is per-disk-4hr or per-disk-1d.

        # Get filename generals of all blobs for the chosen batch indexes.
        batch_rows = self.primary_arr[batch_indexes]
        
        x_data = []
        y_data = []

        for row in batch_rows:

            overall_data_vector = []

            features = row[:-3]
            filename_generals_str = row[-3]
            blob_indices_str = row[-2]

            overall_data_vector.extend(features)

            filename_generals = filename_generals_str.split(',')
            blob_indices = [int(i) for i in blob_indices_str.split(',')]

            volume_data = []

            for filename_general, blob_index in zip(filename_generals, blob_indices):
                # Bout_hmi.sharp_cea_720s.10960.20240314_170000_TAI.bin
                volume_path_single_blob = os.path.join(GENERATED_VOLUMES_PATH_SINGLE_BLOB, f'Bout_{filename_general}.bin')
                volume_path_multi_blob = os.path.join(GENERATED_VOLUMES_PATH_MULTI_BLOB, f'Bout_{filename_general}.bin')

                if os.path.exists(volume_path_single_blob):
                    bx_3D, by_3D, bz_3D = load_volume_components(volume_path_single_blob)
                else:
                    bx_3D, by_3D, bz_3D = load_volume_components(volume_path_multi_blob)
                
                # Fetch the corresponding bitmap so we an mask the bx, by, and bz fields
                associated_bitmap_path = os.path.join(DEFINITIVE_SHARP_DATA_DIR, filename_general + '.bitmap.fits')
                bitmap = sunpy.map.Map(associated_bitmap_path)

                # Resize bitmap data to 200x400
                bitmap_resized = resize(bitmap.data, (200, 400), anti_aliasing=True, preserve_range=True)

                # weak and strong field pixels within the HARP = (33, 34). A little bit of error allowed here due to
                # interpolation from resizing function. The next lowest values in the bitmap are low enough that we
                # can easily look from 30 and up.
                mask_resized = bitmap_resized > 30
                blob_mask_resized = bitmap_resized*mask_resized.astype(int)*1.

                # Separate out blobs
                s = generate_binary_structure(2,2)  # Allows diagonal pixels to be considered part of the same blob

                labeled_resized, nblobs_resized = label(blob_mask_resized, structure=s)

                # Sort blobs in order from greatest area to least area (relevant to AR # identification in MagPy, so needed for cross-referencing)
                # The assumption is that resizing the bitmap won't change the relative sizes of the blobs to each other.
                blobs_resized = [i for i in range(1, nblobs_resized+1)]
                blobs_resized = sorted(blobs_resized, key=lambda x: np.count_nonzero(labeled_resized == x), reverse=True)

                curr_blob = blobs_resized[blob_index - 1]

                # Create a new volume for the blob. That is, mask out ~blob pixels at every height of the volume, which is in the resized scale.
                mask = labeled_resized == curr_blob

                # In order to multiply each 3D component by the mask, we need to make the first dimension of the volume the height.
                # This is because the mask has the same shape as the volume at a given height, and broadcasting only works on the
                # first dimension.
                bx_3D_blob = np.transpose(bx_3D, (2, 0, 1)) # height dimension, then number of rows, then number of cols
                by_3D_blob = np.transpose(by_3D, (2, 0, 1))
                bz_3D_blob = np.transpose(bz_3D, (2, 0, 1))

                bx_3D_blob = bx_3D_blob * mask
                by_3D_blob = by_3D_blob * mask
                bz_3D_blob = bz_3D_blob * mask

                # Transpose back to original shape
                bx_3D_blob = np.transpose(bx_3D_blob, (1, 2, 0))
                by_3D_blob = np.transpose(by_3D_blob, (1, 2, 0))
                bz_3D_blob = np.transpose(bz_3D_blob, (1, 2, 0))

                # Reshape the bx_3D, by_3D, and bz_3D arrays to one array with shape (200, 400, 100, 3),
                # corresponding to the length, width, height and number of channels. Then flatten the cube.
                cube = np.zeros((200, 400, 100, 3))
                cube[:, :, :, 0] = bx_3D_blob
                cube[:, :, :, 1] = by_3D_blob
                cube[:, :, :, 2] = bz_3D_blob

                # Flatten each one and add to the overall data vector.
                volume_data.extend(cube.flatten())

            nx, ny, nz, channels = 200, 400, 100, 3
            
            # If there were less than 5 blobs, fill in the rest with 0s
            while len(volume_data) < PrimarySEPInputDataGenerator.TOP_N_BLOBS*(nx*ny*nz*channels):
                volume_data.extend(np.zeros(nx*ny*nz*channels))

            overall_data_vector.extend(volume_data)
            overall_data_vector = np.array(overall_data_vector, dtype=float)

            x_data.append(overall_data_vector)

            produced_SEP = int(row[-1])
            y_data.append(produced_SEP)

        return np.array(x_data), np.array(y_data)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def toDatetime(self, date):
        return datetime.datetime.strptime(date.strip(), '%Y-%m-%dT%H:%MZ')
    
    # Active region numbers in JSON files automatically get converted to floats.
    def toIntString(self, floatNum):
        return str(int(floatNum))
