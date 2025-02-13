"""
In a style similar to the data sequence mechanism used in NextFramePrediction/NextFramePrediction_FinalModel.py,
create a data mechanism for the SEPPrediction model. Each data sample should consist of:
- The x, y, and z components of the magnetic field volume for a given blob.
- The corresponding vector of data for the blob from the unified 2D/3D dataset.
- The corresponding vectors of data for the same blob and dataset but for the previous 5 time steps (t-4, t-8, t-12, t-16, t-20).
- The class label for the blob, which is whether or not an SEP event occurred within 24 hours of the current time step.
"""

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

rootDir = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(1, os.path.join(rootDir))

from Utils import load_volume_components

GENERATED_VOLUMES_PATH_SINGLE_BLOB = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumes'
GENERATED_VOLUMES_PATH_MULTI_BLOB = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumesMultiblob'
REGULAR_SHARED_DATA_DIR = os.path.join(os.sep + 'share', 'development', 'data', 'drms', 'MagPy_Shared_Data')
DEFINITIVE_SHARP_DATA_DIR = os.path.join(REGULAR_SHARED_DATA_DIR, 'TrainingData' + os.sep)
UNIFIED_DATA_DIR = os.path.join(rootDir, 'OutputData', 'UnifiedActiveRegionData.csv')

class SEPInputDataGenerator(tf.keras.utils.Sequence):
    BLOB_VECTOR_COLUMNS_GENERAL = ['Latitude', 'Carrington Longitude', 'Volume Total Magnetic Energy', 'Volume Total Unsigned Current Helicity', 'Volume Total Absolute Net Current Helicity', 'Volume Mean Shear Angle', 'Volume Total Unsigned Volume Vertical Current', 'Volume Twist Parameter Alpha', 'Volume Mean Gradient of Vertical Magnetic Field', 'Volume Mean Gradient of Total Magnetic Field', 'Volume Total Magnitude of Lorentz Force', 'Volume Total Unsigned Magnetic Flux', 'Gradient_00', 'Gradient_10', 'Gradient_30', 'Gradient_50', 'Shear_00', 'Shear_10', 'Shear_30', 'Shear_50', 'Phi', 'Total Unsigned Current Helicity', 'Total Photospheric Magnetic Free Energy Density', 'Total Unsigned Vertical Current', 'Abs of Net Current helicity', 'Is Plage', 'Stonyhurst Longitude']
    BLOB_ONE_TIME_INFO = ['Number of Recent Flares', 'Max Class Type of Recent Flares', 'Number of Recent CMEs', 'Max Product of Half Angle and Speed of Recent CMEs', 'Number of Sunspots', 'Max Flare Peak of Recent Flares', 'Min Temperature of Recent Flares', 'Median Emission Measure of Recent Flares', 'Median Duration of Recent Flares']
    TIMESERIES_STEPS = 6

    def __init__(self, blob_df, batch_size, shuffle):
        self.blob_df = blob_df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.blob_df))

        # just for consistency if not shuffling
        self.blob_df['datetime'] = pd.to_datetime(self.blob_df['Filename General'].apply(lambda x: x.split('.')[3]), format='%Y%m%d_%H%M%S_TAI')
        self.blob_df = self.blob_df.sort_values(by='datetime')

        # Set random seed for comparison purposes since some of the models we're training
        # will take an extremely long time to train. This will allow us to compare results without
        # having to average over multiple runs. This is not ideal, but it's a temporary solution until
        # we are possibly able to distill the models or use a more efficient training mechanism.
        np.random.seed(42)

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.blob_df) // self.batch_size

    # TODO: Need to check performance of this function. Map loading takes a long time, so make sure batch size
    # is large enough and that parallelization is taking place.
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df_rows = self.blob_df.iloc[batch_indexes]

        # Iterate over the selected rows and load their corresponding data
        x_data = []
        y_data = []

        for _, row in batch_df_rows.iterrows():
            overall_data_vector = []
            curr_blob_one_time_info = row[SEPInputDataGenerator.BLOB_ONE_TIME_INFO].values
            curr_blob_vector = row[SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL].values

            overall_data_vector.extend(curr_blob_one_time_info)

            blob_timeseries_vector = []
            blob_timeseries_vector.extend(curr_blob_vector)
            # overall_data_vector.append(curr_blob_vector)

            filename_general = row['Filename General']
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

            curr_blob_index = row['Blob Index']
            curr_blob = blobs_resized[curr_blob_index - 1]

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

            # We now have both the masked 3D magnetic field components
            # for this blob as well as its corresponding vector from the
            # unified dataset.

            # Get the corresponding blob vectors from the unified dataset
            # for the previous 5 time steps. As a fast lookup approximation,
            # look at the previous 5 time steps by 'Filename General' and 'Blob Index'.
            # Note that this is not perfect because it assumes the relative sizes of
            # blobs of the same SHARP region are the same across time steps, which is not
            # necessarily true. However, this is a reasonable approximation that will prevent
            # us from having to do manual inequality checks on the latitude and longitude columns.
            # TODO: Check how well this approximation works in practice.
            # prev_blob_vectors = []
            for i in range(1, SEPInputDataGenerator.TIMESERIES_STEPS):
                prev_time = row['datetime'] - pd.Timedelta(hours=4*i)
                prev_time_str = prev_time.strftime('%Y%m%d_%H%M%S_TAI')
                prev_filename_part = '.'.join(filename_general.split('.')[:-1])
                prev_filename_general = f'{prev_filename_part}.{prev_time_str}'
                prev_blob_df = self.blob_df[
                    (self.blob_df['Filename General'] == prev_filename_general) &
                    (self.blob_df['Blob Index'] == curr_blob_index)
                ]

                if prev_blob_df.empty:
                    prev_blob_vector = np.zeros(len(SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL))
                else:
                    prev_blob_vector = prev_blob_df.iloc[0][SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL].values
                    
                # prev_blob_vectors.append(prev_blob_vector)
                blob_timeseries_vector.extend(prev_blob_vector)


            # overall_data_vector.extend(prev_blob_vectors)
            overall_data_vector.extend(blob_timeseries_vector)

            # overall_data_vector.append(np.stack([bx_3D_blob, by_3D_blob, bz_3D_blob], axis=-1))
            # Instead of stacking, flatten out the 3D magnetic field components and extend the overall data vector
            overall_data_vector.extend(bx_3D_blob.flatten())
            overall_data_vector.extend(by_3D_blob.flatten())
            overall_data_vector.extend(bz_3D_blob.flatten())

            # overall_data_vector = np.array(overall_data_vector)
            overall_data_vector = np.array(overall_data_vector)

            x_data.append(overall_data_vector)

            # Class label for the blob, which is just the SEP event rate for the blob at the current timestep
            produced_SEP = row['Produced an SEP']
            y_data.append(produced_SEP)

        return np.array(x_data), np.array(y_data)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
