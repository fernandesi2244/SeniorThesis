from Logger import Logger
import os
import re
import pathlib
import pandas as pd
import subprocess
import time
import pickle

rootDir = pathlib.Path(__file__).resolve().parent.absolute()

GENERATED_VOLUMES_PATH = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumes'
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
        df = df.append(intermediate_data, ignore_index=True)

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

        if numJobs == 0:
            break
            
        updateCSVWithIntermediateOutput()

        time.sleep(300)

    updateCSVWithIntermediateOutput()

def main():
    if not os.path.exists(PARAMETERIZATION_CSV_PATH):
        # Create empty df with required columns and save as CSV.
        # Columns: Filename General, Total Magnetic Energy
        df = pd.DataFrame(columns=['Filename General', 'Total Magnetic Energy'])
        df.to_csv(PARAMETERIZATION_CSV_PATH, index=False)
    
    # Generated volume example file: Bout_hmi.sharp_cea_720s.10000.20230828_090000_TAI.bin
    volumes = [os.path.join(GENERATED_VOLUMES_PATH, f) for f in os.listdir(GENERATED_VOLUMES_PATH)]
    volumes.sort()

    # Store list of volume files in txt file so that parameterizationWorkerscript.sh can access them later
    with open(os.path.join(TEMP_DIR, 'volumes.txt'), 'w') as volumesFile:
        volumesFile.writelines(f'{volume}\n' for volume in volumes)

    maxIndex = len(volumes) - 1
    workerscriptLocation = os.path.join(rootDir, 'parameterizationWorkerscript.sh')

    # Schedule job array to process each targetARGen. Allow only 250 jobs to run at a time to avoid overuse of cluster resources.
    os.system(f'sbatch --partition=full --mem-per-cpu=10G --array=0-{maxIndex}%250 "{workerscriptLocation}"') # should be roughly 250 jobs at a time

    logger.log('Volume parameterization job scheduling complete.', 'LOW')

    processResultsAsTheyComeIn()

    logger.log('Volume parameterization job processing complete.', 'LOW')

if __name__ == '__main__':
    logger = Logger('VolumeParameterizationLog.txt')
    logger.clearLog()
    logger.log('Starting volume parameterization job scheduling...', 'LOW')

    main()