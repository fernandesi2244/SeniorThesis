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
from Utils import get_region_of_interest_planes_and_cube

GENERATED_VOLUMES_PATH_SINGLE_BLOB = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumes'
GENERATED_VOLUMES_PATH_MULTI_BLOB = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumesMultiblob'
REGULAR_SHARED_DATA_DIR = os.path.join(os.sep + 'share', 'development', 'data', 'drms', 'MagPy_Shared_Data')
DEFINITIVE_SHARP_DATA_DIR = os.path.join(REGULAR_SHARED_DATA_DIR, 'TrainingData' + os.sep)
UNIFIED_DATA_DIR = os.path.join(rootDir, 'OutputData', 'UnifiedActiveRegionData.csv')

class SEPInputDataGenerator(tf.keras.utils.Sequence):
    BLOB_VECTOR_COLUMNS_GENERAL = ['Latitude', 'Carrington Longitude', 'Is Plage', 'Stonyhurst Longitude']
    BLOB_ONE_TIME_INFO = ['Number of Recent Flares', 'Max Class Type of Recent Flares', 'Number of Recent CMEs', 'Max Product of Half Angle and Speed of Recent CMEs', 'Number of Sunspots', 'Max Flare Peak of Recent Flares', 'Min Temperature of Recent Flares', 'Median Emission Measure of Recent Flares', 'Median Duration of Recent Flares', 'Number of Recent SEPs', 'Number of Recent Subthreshold SEPs']

    TOP_N_BLOBS = 5

    def __init__(self, blob_df, batch_size, shuffle, granularity='per-blob', **kwargs):
        super().__init__(**kwargs)  # For multiprocessing parameters

        self.blob_df = blob_df
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.blob_df['datetime'] = pd.to_datetime(blob_df['Filename General'].apply(lambda x: x.split('.')[3]), format='%Y%m%d_%H%M%S_TAI')
        self.blob_df['date'] = self.blob_df['datetime'].dt.date

        self.granularity = granularity # either 'per-blob', 'per-disk-4hr', or 'per-disk-1d'
        if self.granularity == 'per-blob':
            self.indexes = np.arange(len(self.blob_df))
        elif self.granularity == 'per-disk-4hr':
            # Get unique number of 'datetime's (since we're already assuming 4-hr granularity of the blob_df)
            self.unique_datetimes = self.blob_df['datetime'].unique()
            num_unique_datetimes = len(self.unique_datetimes)
            self.indexes = np.arange(num_unique_datetimes)
            print('Number of unique datetimes:', num_unique_datetimes)
            print('Number of blobs:', len(self.blob_df))
        elif self.granularity == 'per-disk-1d':
            self.unique_dates = self.blob_df['date'].unique()
            num_unique_dates = len(self.unique_dates)
            self.indexes = np.arange(num_unique_dates)
            print('Number of unique days:', num_unique_dates)
            print('Number of blobs:', len(self.blob_df))

        # just for consistency if not shuffling
        self.blob_df = self.blob_df.sort_values(by='datetime')

        # Set random seed for comparison purposes since some of the models we're training
        # will take an extremely long time to train. This will allow us to compare results without
        # having to average over multiple runs. This is not ideal, but it's a temporary solution until
        # we are possibly able to distill the models or use a more efficient training mechanism.
        np.random.seed(42)

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.indexes) // self.batch_size

    # No volume loading so client using this class can probably use large batch size.
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        if self.granularity == 'per-blob':
            batch_df_rows = self.blob_df.iloc[batch_indexes]

            # Iterate over the selected rows and load their corresponding data
            x_data = []
            y_data = []

            for _, row in batch_df_rows.iterrows():
                overall_data_vector = []
                curr_blob_one_time_info = row[SEPInputDataGenerator.BLOB_ONE_TIME_INFO].values
                curr_blob_vector = row[SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL].values

                overall_data_vector.extend(curr_blob_one_time_info)
                overall_data_vector.extend(curr_blob_vector)

                filename_general = row['Filename General']

                # Add selected planes and intersection cube from associated coronal volume.
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

                planes_xy, planes_xz, planes_yz, cube = get_region_of_interest_planes_and_cube(bx_3D_blob, by_3D_blob, bz_3D_blob)

                # Each of planes_{xy, xz, yz} contains 5 2D planes.
                # The cube contains a 5x5x5 cube.
                # Flatten each one and add to the overall data vector.
                for plane in planes_xy:
                    overall_data_vector.extend(plane.flatten())
                for plane in planes_xz:
                    overall_data_vector.extend(plane.flatten())
                for plane in planes_yz:
                    overall_data_vector.extend(plane.flatten())

                overall_data_vector.extend(cube.flatten())
                
                # overall_data_vector = np.array(overall_data_vector)
                overall_data_vector = np.array(overall_data_vector)

                x_data.append(overall_data_vector)

                # Class label for the blob, which is just the SEP event rate for the blob at the current timestep
                produced_SEP = row['Produced an SEP']
                y_data.append(produced_SEP)

            return np.array(x_data), np.array(y_data)
        
        elif self.granularity == 'per-disk-4hr' or self.granularity == 'per-disk-1d':
            # Get filename generals of all blobs for the chosen batch indexes.
            if self.granularity == 'per-disk-4hr':
                chosen_datetimes = self.unique_datetimes[batch_indexes]
                # chosen_blob_df = self.blob_df[self.blob_df['datetime'].isin(chosen_datetimes)]
            elif self.granularity == 'per-disk-1d':
                chosen_datetimes = self.unique_dates[batch_indexes]
                # chosen_blob_df = self.blob_df[self.blob_df['date'].isin(chosen_dates)]
            
            x_data = []
            y_data = []

            for dt in chosen_datetimes:
                if self.granularity == 'per-disk-4hr':
                    chosen_blob_df = self.blob_df[self.blob_df['datetime'] == dt]
                elif self.granularity == 'per-disk-1d':
                    chosen_blob_df = self.blob_df[self.blob_df['date'] == dt]

            
                """
                For all chosen blobs of a given batch, first sort them in descending order of Phi.
                Then, take the top 5 blobs and get the one-time info and blob vector for each of them.
                For each of these top 5 blobs, also get the corresponding vectors for the previous 5 time steps.

                Let the y label be 1 if any of the top 5 blobs produced an SEP event, and 0 otherwise.
                """

                # Sort the blobs in descending order of Phi
                chosen_blob_df = chosen_blob_df.sort_values(by='Phi', ascending=False)
                chosen_blob_df_all_blobs = chosen_blob_df.copy()
                chosen_blob_df = chosen_blob_df.head(SEPInputDataGenerator.TOP_N_BLOBS)

                # take sum of recent flares
                full_disk_num_recent_flares = chosen_blob_df['Number of Recent Flares'].sum()
                full_disk_max_class_type = chosen_blob_df['Max Class Type of Recent Flares'].max()
                full_disk_num_recent_cmes = chosen_blob_df['Number of Recent CMEs'].sum()
                full_disk_max_product_half_angle_speed = chosen_blob_df['Max Product of Half Angle and Speed of Recent CMEs'].max()
                full_disk_num_sunspots = chosen_blob_df.iloc[0]['Number of Sunspots']


                # note that these full-disk quantities are much more imperfect since the relationships between
                # these measures and SEP production is noticed in pairs, i.e., Max Flare Peak and Min Temperature
                # being used together to predict SEPs and Median Emission and Median Duration being used together
                # to predict SEPs. Therefore, maxs, mins, and medians may come from different points, but the overall
                # assumption here is that given the apparent decision boundaries, these quantities are still useful.
                # See https://sun.njit.edu/SEP3/datasets.html for the relevant plots.
                full_disk_max_flare_peak = chosen_blob_df['Max Flare Peak of Recent Flares'].max()
                full_disk_min_temp = chosen_blob_df['Min Temperature of Recent Flares'].min()
                full_disk_median_emission_measure = chosen_blob_df['Median Emission Measure of Recent Flares'].median()
                full_disk_median_duration = chosen_blob_df['Median Duration of Recent Flares'].median()

                full_disk_num_recent_SEPs = chosen_blob_df['Number of Recent SEPs'].sum()
                full_disk_num_recent_subthreshold_SEPs = chosen_blob_df['Number of Recent Subthreshold SEPs'].sum()

                full_disk_one_time_info = [
                    full_disk_num_recent_flares, full_disk_max_class_type, full_disk_num_recent_cmes,
                    full_disk_max_product_half_angle_speed, full_disk_num_sunspots, full_disk_max_flare_peak,
                    full_disk_min_temp, full_disk_median_emission_measure, full_disk_median_duration,
                    full_disk_num_recent_SEPs, full_disk_num_recent_subthreshold_SEPs
                ]

                overall_blob_data = []
                for _, row in chosen_blob_df.iterrows():
                    curr_blob_vector = row[SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL].values

                    overall_blob_data.extend(curr_blob_vector)

                    filename_general = row['Filename General']

                    # Add selected planes and intersection cube from associated coronal volume.
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

                    planes_xy, planes_xz, planes_yz, cube = get_region_of_interest_planes_and_cube(bx_3D_blob, by_3D_blob, bz_3D_blob)

                    # Each of planes_{xy, xz, yz} contains 5 2D planes.
                    # The cube contains a 5x5x5 cube.
                    # Flatten each one and add to the overall data vector.
                    for plane in planes_xy:
                        overall_blob_data.extend(plane.flatten())
                    for plane in planes_xz:
                        overall_blob_data.extend(plane.flatten())
                    for plane in planes_yz:
                        overall_blob_data.extend(plane.flatten())

                    overall_blob_data.extend(cube.flatten())
                
                nx, ny, nz, channels = 200, 400, 100, 3

                # If there were less than 5 blobs, fill in the rest with 0s
                while len(overall_blob_data) < SEPInputDataGenerator.TOP_N_BLOBS * (len(SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL) + (5*nx*ny*channels + 5*nx*nz*channels + 5*ny*nz*channels + 5*5*5*channels)):
                    overall_blob_data.extend(np.zeros(len(SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL) + 5*nx*ny*channels + 5*nx*nz*channels + 5*ny*nz*channels + 5*5*5*channels))
                
                complete_data_vector = full_disk_one_time_info + overall_blob_data
                complete_data_vector = np.array(complete_data_vector)

                x_data.append(complete_data_vector)

                # Class label for disk, which is 1 if any of the top 5 blobs produced an SEP event, and 0 otherwise
                produced_SEP = chosen_blob_df_all_blobs['Produced an SEP'].max() # need to look at all blobs since top 5 are just a proxy heuristic
                y_data.append(produced_SEP)

            return np.array(x_data), np.array(y_data)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
