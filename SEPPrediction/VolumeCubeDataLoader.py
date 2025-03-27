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

GENERATED_VOLUME_SLICES_AND_CUBE_PATH = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/VolumeSlicesAndCubes'
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

                cube_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_cube.npy')

                # Load the cube
                cube = np.load(cube_path)

                # The cube contains a 5x5x5 cube.
                # Flatten each one and add to the overall data vector.
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
                cube_path = os.path.join(GENERATED_VOLUME_SLICES_AND_CUBE_PATH, f'{filename_general}_blob{blob_index}_cube.npy')

                # Load the cube
                cube = np.load(cube_path)

                # The cube contains a 5x5x5 cube.
                # Flatten each one and add to the overall data vector.
                volume_data.extend(cube.flatten())

            nx, ny, nz, channels = 200, 400, 100, 3
            
            # If there were less than 5 blobs, fill in the rest with 0s
            while len(volume_data) < PrimarySEPInputDataGenerator.TOP_N_BLOBS*(5*5*5*channels):
                volume_data.extend(np.zeros(5*5*5*channels))

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
