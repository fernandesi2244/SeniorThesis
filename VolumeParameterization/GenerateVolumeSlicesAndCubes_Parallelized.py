import os
import re
import pathlib
import pandas as pd
import subprocess
import time
import pickle
import sys

rootDir = pathlib.Path(__file__).resolve().parent.parent.absolute()

sys.path.insert(1, os.path.join(rootDir))
from Logger import Logger

GENERATED_VOLUMES_PATH_SINGLE_BLOB = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumes'
GENERATED_VOLUMES_PATH_MULTI_BLOB = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumesMultiblob'

REGULAR_SHARED_DATA_DIR = os.path.join(os.sep + 'share', 'development', 'data', 'drms', 'MagPy_Shared_Data')
DEFINITIVE_SHARP_DATA_DIR = os.path.join(REGULAR_SHARED_DATA_DIR, 'TrainingData' + os.sep)

OUTPUT_PATH = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/VolumeSlicesAndCubes'

TEMP_DIR = os.path.join(rootDir, 'OutputData', 'Temp')

def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    # Generated volume example file: Bout_hmi.sharp_cea_720s.10000.20230828_090000_TAI.bin
    single_blob_volume_paths = [os.path.join(GENERATED_VOLUMES_PATH_SINGLE_BLOB, f) for f in os.listdir(GENERATED_VOLUMES_PATH_SINGLE_BLOB)]
    multi_blob_volume_paths = [os.path.join(GENERATED_VOLUMES_PATH_MULTI_BLOB, f) for f in os.listdir(GENERATED_VOLUMES_PATH_MULTI_BLOB)]
    volumes = single_blob_volume_paths + multi_blob_volume_paths
    volumes.sort()

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # Store list of volume files in txt file so that parameterizationWorkerscript.sh can access them later
    with open(os.path.join(TEMP_DIR, 'volumes_for_slices_and_cubes.txt'), 'w') as volumesFile:
        volumesFile.writelines(f'{volume}\n' for volume in volumes)

    maxIndex = len(volumes) - 1
    workerscriptLocation = os.path.join(rootDir, 'VolumeParameterization', 'slicesAndCubeGenerationWorkerscript.sh')

    # Schedule job array to process each targetARGen. Allow only 150 jobs to run at a time to avoid overuse of cluster resources.
    os.system(f'sbatch --partition=full --mem-per-cpu=10G --array=0-{maxIndex}%150 "{workerscriptLocation}"') # should be roughly 250 jobs at a time

    logger.log('Volume parameterization job scheduling complete.', 'LOW')

if __name__ == '__main__':
    logger = Logger('VolumeParameterizationLog.txt')
    logger.clearLog()
    logger.log('Starting volume parameterization job scheduling...', 'LOW')

    main()
