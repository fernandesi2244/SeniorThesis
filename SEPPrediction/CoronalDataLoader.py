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
    BLOB_VECTOR_COLUMNS_GENERAL = ['Latitude', 'Carrington Longitude', 'Volume Total Magnetic Energy', 'Volume Total Unsigned Current Helicity', 'Volume Total Absolute Net Current Helicity', 'Volume Mean Shear Angle', 'Volume Total Unsigned Volume Vertical Current', 'Volume Twist Parameter Alpha', 'Volume Mean Gradient of Vertical Magnetic Field', 'Volume Mean Gradient of Total Magnetic Field', 'Volume Total Magnitude of Lorentz Force', 'Volume Total Unsigned Magnetic Flux', 'Is Plage', 'Stonyhurst Longitude']
    BLOB_ONE_TIME_INFO = ['Number of Recent Flares', 'Max Class Type of Recent Flares', 'Number of Recent CMEs', 'Max Product of Half Angle and Speed of Recent CMEs', 'Number of Sunspots', 'Max Flare Peak of Recent Flares', 'Min Temperature of Recent Flares', 'Median Emission Measure of Recent Flares', 'Median Duration of Recent Flares', 'Number of Recent SEPs', 'Number of Recent Subthreshold SEPs']
    TIMESERIES_STEPS = 6
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

                blob_timeseries_vector = []
                blob_timeseries_vector.extend(curr_blob_vector)
                # overall_data_vector.append(curr_blob_vector)

                filename_general = row['Filename General']
                
                curr_blob_index = row['Blob Index']

                # We now have this blob's corresponding vector from the
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

                    blob_timeseries_vector = []
                    blob_timeseries_vector.extend(curr_blob_vector)

                    filename_general = row['Filename General']
                    
                    curr_blob_index = row['Blob Index']

                    # We now have this blob's corresponding vector from the
                    # unified dataset.

                    # Get the corresponding blob vectors from the unified dataset
                    # for the previous 5 time steps. As a fast lookup approximation,
                    # look at the previous 5 time steps by 'Filename General' and 'Blob Index'.
                    # Note that this is not perfect because it assumes the relative sizes of
                    # blobs of the same SHARP region are the same across time steps, which is not
                    # necessarily true. However, this is a reasonable approximation that will prevent
                    # us from having to do manual inequality checks on the latitude and longitude columns.
                    for i in range(1, SEPInputDataGenerator.TIMESERIES_STEPS):
                        if self.granularity == 'per-disk-4hr':
                            prev_time = row['datetime'] - pd.Timedelta(hours=4*i)
                        elif self.granularity == 'per-disk-1d':
                            prev_time = row['datetime'] - pd.Timedelta(days=i)
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
                            
                        blob_timeseries_vector.extend(prev_blob_vector)

                    overall_blob_data.extend(blob_timeseries_vector)
                
                # If there were less than 5 blobs, fill in the rest with 0s
                while len(overall_blob_data) < SEPInputDataGenerator.TOP_N_BLOBS * SEPInputDataGenerator.TIMESERIES_STEPS * len(SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL):
                    overall_blob_data.extend(np.zeros(SEPInputDataGenerator.TIMESERIES_STEPS * len(SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL))) # add blob at a time
                
                complete_data_vector = full_disk_one_time_info + overall_blob_data
                complete_data_vector = np.array(complete_data_vector)

                x_data.append(complete_data_vector)

                # Class label for disk, which is 1 if any of the top 5 blobs produced an SEP event, and 0 otherwise
                produced_SEP = chosen_blob_df_all_blobs['Produced an SEP'].max()
                y_data.append(produced_SEP)

            return np.array(x_data), np.array(y_data)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
