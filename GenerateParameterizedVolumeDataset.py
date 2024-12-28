from Logger import Logger
import os
import re
import pathlib
import pandas as pd

rootDir = pathlib.Path(__file__).resolve().parent.absolute()

GENERATED_VOLUMES_PATH = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumes'
PARAMETERIZATION_CSV_PATH = os.path.join(rootDir, 'OutputData', 'volume_parameterizations.csv')
TEMP_DIR = os.path.join(rootDir, 'OutputData', 'Temp')

def main():
    if not os.path.exists(PARAMETERIZATION_CSV_PATH):
        # Create empty df with required columns and save as CSV.
        # Columns: Filename General, Total Magnetic Energy
        df = pd.DataFrame(columns=['Filename General', 'Total Magnetic Energy'])
        df.to_csv(PARAMETERIZATION_CSV_PATH, index=False)
    
    # Generated volume example file: Bout_hmi.sharp_cea_720s.10000.20230828_090000_TAI.bin
    volumes = [os.path.join(GENERATED_VOLUMES_PATH, f) for f in os.listdir(GENERATED_VOLUMES_PATH)]
    volumes.sort()

    # Store list of targetARGens in txt file so that parameterizationWorkerscript.sh can access them later
    with open(os.path.join(TEMP_DIR, 'volumes.txt'), 'w') as volumesFile:
        volumesFile.writelines(f'{volume}\n' for volume in volumes)

    max_index = len(volumes) - 1
    workerscriptLocation = os.path.join(rootDir, 'parameterizationWorkerscript.sh')

    # Schedule job array to process each targetARGen. Allow only 250 jobs to run at a time to avoid overuse of cluster resources.
    os.system(f'sbatch --partition=full --mem-per-cpu=10G --array=0-{max_index}%250 "{workerscriptLocation}"') # should be roughly 250 jobs at a time

if __name__ == '__main__':
    main()