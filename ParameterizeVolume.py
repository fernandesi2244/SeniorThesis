"""
Given a volume file path, load the volume and calculate various parameters for the volume,
e.g., the total magnetic energy. Save these parameters to a CSV file.
"""

import os
import sys
import numpy as np
import pandas as pd
from Logger import Logger
import sunpy.map
import math
import pathlib

rootDir = pathlib.Path(__file__).resolve().parent.absolute()

REGULAR_SHARED_DATA_DIR = os.path.join(os.sep + 'share', 'development', 'data', 'drms', 'MagPy_Shared_Data')
DEFINITIVE_SHARP_DATA_DIR = os.path.join(REGULAR_SHARED_DATA_DIR, 'TrainingData' + os.sep)

PARAMETERIZATION_CSV_PATH = os.path.join(rootDir, 'OutputData', 'volume_parameterizations.csv')

def load_volume_components(filename):
    """
    Read in 3-D magnetic field components from generated C code output.
    """
    
    # Load binary data from file
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.float64)  # Assuming double precision float
    
    nx, ny, nz = 200, 400, 100
    
    flattened_size = nx * ny * nz

    # Initialize 3D magnetic field arrays
    bx_3D = data[:flattened_size].reshape((nx, ny, nz))
    by_3D = data[flattened_size:2*flattened_size].reshape((nx, ny, nz))
    bz_3D = data[2*flattened_size:3*flattened_size].reshape((nx, ny, nz))

    return bx_3D, by_3D, bz_3D

def get_total_magnetic_energy(bx_3D, by_3D, bz_3D, target_ar_gen):
    """
    Calculate the total magnetic energy in a Cartesian volume.
    =================================================
    Parameters:
    - bx_3D, by_3D, bz_3D : 3D numpy arrays representing magnetic field components in x, y, z directions.
    - target_ar_gen : The targetARGen of the volume file.

    This function computes total volume magnetic energy. The units for magnetic energy in cgs are ergs. The formula B^2/8*PI integrated over all space, dV
    automatically yields erg for an input B in Gauss. Note that the 8*PI can come out of the integral; thus, the integral is over B^2 dV and the 8*PI is divided at the end.

    Total magnetic energy is the magnetic energy density times dV. To convert ergs per centimeter cubed to ergs, simply multiply by its volume per pixel in cm:
      erg/cm^3*(CDELT1^3)*(RSUN_REF/RSUN_OBS ^3)*(100.^3)
    = erg(1/pix^3)
    
    Free magnetic energy in active regions typically ranges between 10^31 and 10^33 erg depending on the region's size, field strength, and complexity.

    Returns:
    - Total magnetic energy (float)
    """
        
    # Read in FITS headers
    associated_Bp_file_path = os.path.join(DEFINITIVE_SHARP_DATA_DIR, target_ar_gen + '.Bp.fits')
    bp_map = sunpy.map.Map(associated_Bp_file_path)
    cdelt1 = (math.atan((bp_map.meta['rsun_ref']*bp_map.meta['cdelt1']*np.pi/180)/(bp_map.meta['dsun_obs'])))*(180/np.pi)*(3600)  
    dx = (cdelt1*(bp_map.meta['rsun_ref']/bp_map.meta['rsun_obs'])*100.0) 

    # Calculate the squared magnetic field magnitude at each point
    mu_0 = 8 * np.pi   # 8π appears due to the CGS formulation of Maxwell's equations.
    B_squared = bx_3D**2 + by_3D**2 + bz_3D**2

    # Calculate magnetic energy density at each point
    energy_density = B_squared / (mu_0)
    
    # Integrate the energy density over the volume
    dy = dx
    dz = dx
    dV = dx * dy * dz  # Volume of each grid cell
    total_energy = np.sum(energy_density) * dV

    return total_energy

def main():
    # Load the volume file
    volume_path = sys.argv[1]
    try:
        bx_3D, by_3D, bz_3D = load_volume_components(volume_path)
    except Exception as e:
        print(f'ERROR: Failed to load volume components from {volume_path}: {repr(e)}')
        logger.log(f'Failed to load volume components from {volume_path}: {repr(e)}', 'HIGH')
        return

    # Bout_hmi.sharp_cea_720s.10960.20240314_170000_TAI.bin
    target_ar_gen = os.path.basename(volume_path)[5:-4]

    tot_mag_energy = get_total_magnetic_energy(bx_3D, by_3D, bz_3D, target_ar_gen)
    # other parameters to calculate here...

    # Make a new DF with the calculated parameters as one row and save it as a pickle file.
    df = pd.DataFrame(columns=['Filename General', 'Total Magnetic Energy'])
    df.loc[0] = [target_ar_gen, tot_mag_energy]
    # save as pickle file to OutputData/Temp with same name as target_ar_gen
    df.to_pickle(os.path.join(rootDir, 'OutputData', 'Temp', f'{target_ar_gen}.pkl'))

if __name__ == '__main__':
    logger = Logger('VolumeParameterizationLog.txt')
    main()