#!/bin/bash
#SBATCH --job-name=MagPy_volume_parameterization_sbatch
#SBATCH --output=MagPy_volume_parameterization_sbatch.out
#SBATCH --error=MagPy_volume_parameterization_sbatch.err
export OMP_NUM_THREADS=1

echo "This job in the array has:"
echo "- SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "- SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

VOLUME_FILES_FILE="../OutputData/Temp/volumes.txt"

mapfile -t FILES < $VOLUME_FILES_FILE
VOLUME_PATH=${FILES[$SLURM_ARRAY_TASK_ID]}
VOLUME_PATH_WITHOUT_DIR=${VOLUME_PATH##*/}

echo "My array index is ${SLURM_ARRAY_TASK_ID} out of an array length of ${#FILES[@]}"
echo "My input file is ${VOLUME_PATH}"

echo "VOLUME PATH WITHOUT DIR: ${VOLUME_PATH_WITHOUT_DIR}"

echo "Starting volume parameterization for ${VOLUME_PATH_WITHOUT_DIR}"

python3.9 ParameterizeVolume.py ${VOLUME_PATH}

# Check if the Python script returned an error
if [ $? -ne 0 ]; then
    echo "Error in parameterization script for ${VOLUME_PATH}"
    exit 1
fi

echo "Completed volume parameterization for ${VOLUME_PATH_WITHOUT_DIR}"
