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

rootDir = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(1, os.path.join(rootDir))

GENERATED_VOLUME_SLICES_AND_CUBE_PATH = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/VolumeSlicesAndCubes'
REGULAR_SHARED_DATA_DIR = os.path.join(os.sep + 'share', 'development', 'data', 'drms', 'MagPy_Shared_Data')
DEFINITIVE_SHARP_DATA_DIR = os.path.join(REGULAR_SHARED_DATA_DIR, 'TrainingData' + os.sep)

# Primary is shared by all of [slices and cubes, slices, cubes] since the actual physical volume data is not attached yet (only attached in secondary data loader).
class PrimarySEPInputDataGenerator(tf.keras.utils.Sequence):
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
        
        # Load SEP (CSV), flare (JSON) and CME (JSON) data
        self.SEPs = pd.read_csv('../InputData/SPEs_with_types.csv')

        # Ignore any SEPs rows that don't have an 'endTime'
        self.SEPs = self.SEPs.dropna(subset=['endTime'])

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
                curr_blob_one_time_info = row[PrimarySEPInputDataGenerator.BLOB_ONE_TIME_INFO].values
                curr_blob_vector = row[PrimarySEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL].values

                overall_data_vector.extend(curr_blob_one_time_info)
                overall_data_vector.extend(curr_blob_vector)

                filename_general = row['Filename General']
                overall_data_vector.append(filename_general)
                overall_data_vector.append(row['Blob Index'])
                
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
                chosen_blob_df = chosen_blob_df.head(PrimarySEPInputDataGenerator.TOP_N_BLOBS)

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
                filename_generals = []
                blob_indices = []
                for _, row in chosen_blob_df.iterrows():
                    curr_blob_vector = row[PrimarySEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL].values

                    overall_blob_data.extend(curr_blob_vector)

                    filename_general = row['Filename General']
                    filename_generals.append(filename_general)
                    blob_indices.append(row['Blob Index'])

                # If there were less than 5 blobs, fill in the rest with 0s
                while len(overall_blob_data) < PrimarySEPInputDataGenerator.TOP_N_BLOBS * (len(PrimarySEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL)):
                    overall_blob_data.extend(np.zeros(len(PrimarySEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL)))
                
                filename_generals_str = ','.join(filename_generals)
                blob_indices_str = ','.join([str(i) for i in blob_indices])
                complete_data_vector = full_disk_one_time_info + overall_blob_data + [filename_generals_str] + [blob_indices_str]

                complete_data_vector = np.array(complete_data_vector)

                x_data.append(complete_data_vector)

                # Class label for disk, which is 1 if any of the blobs at that time produced an SEP event, and 0 otherwise.
                # However, we actually need to first get the NOAA AR numbers of the blobs on the disk at the current time
                # and then check if any SEPs are associated with that time (not the time of the blobs).
                
                # Although in the per-disk-4hr case, all blobs are at same time so old method still works.
                if self.granularity == 'per-disk-4hr':
                    produced_SEP = chosen_blob_df_all_blobs['Produced an SEP'].max()
                    y_data.append(produced_SEP)
                    continue

                # Get the NOAA AR numbers of the blobs on the disk at the current time
                associated_ARs = chosen_blob_df_all_blobs['Most Probable AR Num'].values
                # add 1 day to the current date and then set the time to 00:00:00
                date_plus_one_day = dt + datetime.timedelta(days=1)
                datetime_plus_one_day = datetime.datetime.combine(date_plus_one_day, datetime.datetime.min.time())

                SEPs_in_range = self.SEPs[
                    (self.SEPs['endTime'].apply(self.toDatetime) >= datetime_plus_one_day) &
                    (self.SEPs['beginTime'].apply(self.toDatetime) <= datetime_plus_one_day + datetime.timedelta(days=1))
                ]
                relevant_SEPs = SEPs_in_range[
                    (SEPs_in_range['activeRegionNum'].apply(self.toIntString).isin(associated_ARs)) &
                    (SEPs_in_range['P10OnsetMax'] >= 10)
                ]
                produced_SEP = int(not relevant_SEPs.empty)
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

    # No volume loading so client using this class can probably use large batch size.
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

                xy_slices_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_planes_xy.npy')
                xz_slices_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_planes_xz.npy')
                yz_slices_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_planes_yz.npy')
                cube_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_cube.npy')

                # Load the slices and cube
                planes_xy = np.load(xy_slices_path)
                planes_xz = np.load(xz_slices_path)
                planes_yz = np.load(yz_slices_path)
                cube = np.load(cube_path)

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
                produced_SEP = row[-1]
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
                xy_slices_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_planes_xy.npy')
                xz_slices_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_planes_xz.npy')
                yz_slices_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_planes_yz.npy')
                cube_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_cube.npy')

                # Load the slices and cube
                planes_xy = np.load(xy_slices_path)
                planes_xz = np.load(xz_slices_path)
                planes_yz = np.load(yz_slices_path)
                cube = np.load(cube_path)

                # Each of planes_{xy, xz, yz} contains 5 2D planes.
                # The cube contains a 5x5x5 cube.
                # Flatten each one and add to the overall data vector.
                for plane in planes_xy:
                    volume_data.extend(plane.flatten())
                for plane in planes_xz:
                    volume_data.extend(plane.flatten())
                for plane in planes_yz:
                    volume_data.extend(plane.flatten())

                volume_data.extend(cube.flatten())
            

            nx, ny, nz, channels = 200, 400, 100, 3
            
            # If there were less than 5 blobs, fill in the rest with 0s
            while len(volume_data) < PrimarySEPInputDataGenerator.TOP_N_BLOBS*(5*nx*ny*channels + 5*nx*nz*channels + 5*ny*nz*channels + 5*5*5*channels):
                volume_data.extend(np.zeros(5*nx*ny*channels + 5*nx*nz*channels + 5*ny*nz*channels + 5*5*5*channels))

            overall_data_vector.extend(volume_data)
            overall_data_vector = np.array(overall_data_vector)

            x_data.append(overall_data_vector)

            produced_SEP = row[-1]
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
