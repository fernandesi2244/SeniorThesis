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
PARAMETERIZATION_CSV_PATH = os.path.join(rootDir, 'OutputData', 'volume_parameterizations.csv')
TEMP_DIR = os.path.join(rootDir, 'OutputData', 'Temp')

def updateCSVWithIntermediateOutput():
    """
    Check the Temp directory for new pickled results from each job. When a new result is found, append it to
    the parameterization CSV file.
    """

    # Get list of files in Temp directory
    files = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR)]

    # Load the parameterization CSV file
    df = pd.read_csv(PARAMETERIZATION_CSV_PATH)

    # Iterate over each file in Temp directory
    for file in files:
        if not file.endswith('.pkl'):
            continue

        # Load the pickled data
        with open(file, 'rb') as f:
            intermediate_data = pickle.load(f)

        # Append the data to the parameterization CSV file
        df = pd.concat([df, intermediate_data], ignore_index=True)

        # Remove the pickled file
        os.remove(file)

    # Save the updated CSV file
    df.to_csv(PARAMETERIZATION_CSV_PATH, index=False)

def processResultsAsTheyComeIn():
    """
    While there are still jobs running in SLURM, continuously check OutputData\Temp
    for new pickled results from each job. When a new result is found, append it to
    the parameterization CSV file.
    """

    cmdCommand = 'squeue | grep MagPy | wc -l'
    while(True):
        process = subprocess.Popen(cmdCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = process.communicate()[0]
        try:
            numJobs = int(out)
        except:
            print('numJobs could not be parsed when running the CMD command!')
            time.sleep(300) # Try again in 300 seconds (5 minutes)
            continue
        
        print('Number of jobs running:', numJobs)

        if numJobs == 0:
            break
            
        updateCSVWithIntermediateOutput()

        time.sleep(300)

    updateCSVWithIntermediateOutput()

def main():
    if not os.path.exists(PARAMETERIZATION_CSV_PATH):
        # Create empty df with required columns and save as CSV.
        df = pd.DataFrame(columns=['Filename General', 'Blob Index', 'Latitude', 'Carrington Longitude', 'Volume Total Magnetic Energy', 'Volume Total Unsigned Current Helicity', 'Volume Total Absolute Net Current Helicity', 'Volume Mean Shear Angle', 'Volume Total Unsigned Volume Vertical Current', 'Volume Twist Parameter Alpha', 'Volume Mean Gradient of Vertical Magnetic Field', 'Volume Mean Gradient of Total Magnetic Field', 'Volume Total Magnitude of Lorentz Force', 'Volume Total Unsigned Magnetic Flux', 'Number of Field Lines Leaving Top'])
        df.to_csv(PARAMETERIZATION_CSV_PATH, index=False)
    
    # Generated volume example file: Bout_hmi.sharp_cea_720s.10000.20230828_090000_TAI.bin
    single_blob_volume_paths = [os.path.join(GENERATED_VOLUMES_PATH_SINGLE_BLOB, f) for f in os.listdir(GENERATED_VOLUMES_PATH_SINGLE_BLOB)]
    multi_blob_volume_paths = [os.path.join(GENERATED_VOLUMES_PATH_MULTI_BLOB, f) for f in os.listdir(GENERATED_VOLUMES_PATH_MULTI_BLOB)]
    volumes = single_blob_volume_paths + multi_blob_volume_paths
    volumes.sort()

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # Store list of volume files in txt file so that parameterizationWorkerscript.sh can access them later
    with open(os.path.join(TEMP_DIR, 'volumes.txt'), 'w') as volumesFile:
        volumesFile.writelines(f'{volume}\n' for volume in volumes)

    maxIndex = len(volumes) - 1
    workerscriptLocation = os.path.join(rootDir, 'VolumeParameterization', 'parameterizationWorkerscript.sh')

    # Schedule job array to process each targetARGen. Allow only 150 jobs to run at a time to avoid overuse of cluster resources.
    os.system(f'sbatch --partition=full --mem-per-cpu=10G --array=0-{maxIndex}%150 "{workerscriptLocation}"') # should be roughly 250 jobs at a time

    logger.log('Volume parameterization job scheduling complete.', 'LOW')

    time.sleep(600) # Just wait 10 minutes in case scheduling takes a while

    processResultsAsTheyComeIn()

    logger.log('Volume parameterization job processing complete.', 'LOW')

if __name__ == '__main__':
    logger = Logger('VolumeParameterizationLog.txt')
    logger.clearLog()
    logger.log('Starting volume parameterization job scheduling...', 'LOW')

    main()
