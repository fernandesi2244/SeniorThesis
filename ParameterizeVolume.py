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

import Utils

rootDir = pathlib.Path(__file__).resolve().parent.absolute()

REGULAR_SHARED_DATA_DIR = os.path.join(os.sep + 'share', 'development', 'data', 'drms', 'MagPy_Shared_Data')
DEFINITIVE_SHARP_DATA_DIR = os.path.join(REGULAR_SHARED_DATA_DIR, 'TrainingData' + os.sep)

PARAMETERIZATION_CSV_PATH = os.path.join(rootDir, 'OutputData', 'volume_parameterizations.csv')

"""
- Calculate the total magnetic energy in a Cartesian volume.
- Compute the total unsigned current helicity in a Cartesian box.
- Compute total Absolute value of the net current helicity
- Calculate the mean shear angle between the magnetic field and the current density
- Calculate the total unsigned volume vertical current in a Cartesian box in CGS.
- Calculate the twist parameter alpha in a solar volume.
- Compute the mean gradient of the vertical magnetic field in a Cartesian box.
- Compute the mean gradient of the total magnetic field in a Cartesian box.
- Compute the total magnitude of the Lorentz force in a Cartesian box.
- Compute the total unsigned magnetic flux in a Cartesian box.
"""

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

    global dx, dy, dz, dV
        
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

def get_total_unsigned_current_helicity(bx_3D, by_3D, bz_3D):
    """
    Compute the total unsigned current helicity in a Cartesian box
    """
    global hc, jx, jy, jz

    # Compute the curl of the magnetic field (current density)
    # jx = (dBz/dy - dBy/dz)
    jx = (np.gradient(bz_3D, axis=1) / dy - np.gradient(by_3D, axis=2) / dz)
    # jy = (dBx/dz - dBz/dx)
    jy = (np.gradient(bx_3D, axis=2) / dz - np.gradient(bz_3D, axis=0) / dx)
    # jz = (dBy/dx - dBx/dy)
    jz = (np.gradient(by_3D, axis=0) / dx - np.gradient(bx_3D, axis=1) / dy)

    # Compute the current helicity density: hc = Bx * Jx + By * Jy + Bz * Jz
    hc = bx_3D * jx + by_3D * jy + bz_3D * jz # Dot product of B and J

    # Compute the absolute value of the helicity density
    abs_hc = np.abs(hc)    

    # Integrate over the entire box to get the total unsigned current helicity
    total_unsigned_helicity = np.sum(abs_hc) * dV # Total Unsigned Current Helicity

    return total_unsigned_helicity

def get_total_absolute_net_current_helicity(bx_3D, by_3D, bz_3D):
    abs_value_net_current_helicity = np.abs(np.sum(hc))* dV # Absolute value of the net current helicity    
    return abs_value_net_current_helicity

def get_mean_shear_angle(bx_3D, by_3D, bz_3D):
    # Calculate the magnitudes of B and J
    B_magnitude = np.sqrt(bx_3D**2 + by_3D**2 + bz_3D**2)  # |B| (in Gauss)
    J_magnitude = np.sqrt(jx**2 + jy**2 + jz**2)  # |J| (in statA/cm^2)
    BJ = B_magnitude * J_magnitude
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = np.where(BJ != 0, hc/BJ, 0)
    # Clip values of cos(theta) to avoid numerical errors leading to values slightly greater than 1 or less than -1
    cos_theta = np.clip(cos_theta, -1, 1)
    # Calculate shear angles in radians and convert to degrees
    shear_angles_deg = np.degrees(np.arccos(cos_theta))
    
    # Calculate the mean shear angle
    mean_shear_angle = np.mean(shear_angles_deg)
    
    return mean_shear_angle

def get_total_unsigned_volume_vertical_current(bx_3D, by_3D, bz_3D):
    """
    Calculate the total unsigned volume vertical current
    """

    # Constants
    c = 3e10  # Speed of light in CGS (cm/s)
    factor = c / (4 * np.pi)  # Conversion factor for current density in CGS

    # Compute partial derivatives
    dBy_dx = np.gradient(by_3D, dx, axis=1)  # Partial derivative of By with respect to x
    dBx_dy = np.gradient(bx_3D, dy, axis=0)  # Partial derivative of Bx with respect to y

    # Vertical current density Jz (in statA/cm^2)
    Jz = factor * (dBy_dx - dBx_dy)

    # Calculate total unsigned vertical current
    total_unsigned_current = np.sum(np.abs(Jz)) * dV # Total Unsigned Volume Vertical Current
    
    return total_unsigned_current

def get_twist_parameter_alpha(bx_3D, by_3D, bz_3D):
    """
    Calculate the mean twist parameter alpha in a solar volume.
    """

    # Compute the curl of B
    dBy_dz2 = np.gradient(by_3D, axis=2) / dz
    dBz_dy2 = np.gradient(bz_3D, axis=1) / dy
    dBz_dx2 = np.gradient(bz_3D, axis=0) / dx
    dBx_dz2 = np.gradient(bx_3D, axis=2) / dz
    dBx_dy2 = np.gradient(bx_3D, axis=1) / dy
    dBy_dx2 = np.gradient(by_3D, axis=0) / dx

    curl_Bx = dBz_dy2 - dBy_dz2
    curl_By = dBx_dz2 - dBz_dx2
    curl_Bz = dBy_dx2 - dBx_dy2
   
    # Compute the dot product (curl(B) . B)
    curl_B_dot_B = (curl_Bx * bx_3D) + (curl_By * by_3D) + (curl_Bz * bz_3D)

    # Compute the magnitude of B squared
    B_magnitude_squared = np.sum(bx_3D**2 + by_3D**2 + bz_3D**2)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        twist_parameter_alpha = np.where(B_magnitude_squared > 0, np.sum(curl_B_dot_B) / B_magnitude_squared, 0) # Mean characteristic twist parameter, alpha
    
    return twist_parameter_alpha

def get_mean_gradient_vertical_magnetic_field(bx_3D, by_3D, bz_3D):
    """
    Compute the mean gradient of the vertical magnetic field in a Cartesian box
    """

    # Compute the partial derivatives using central differences
    dBz_dx = (np.roll(bz_3D, -1, axis=0) - np.roll(bz_3D, 1, axis=0)) / (2 * dx)
    dBz_dy = (np.roll(bz_3D, -1, axis=1) - np.roll(bz_3D, 1, axis=1)) / (2 * dy)
    dBz_dz = (np.roll(bz_3D, -1, axis=2) - np.roll(bz_3D, 1, axis=2)) / (2 * dz)

    # Compute the magnitude of the gradient at each point
    gradient_magnitude = np.sqrt(dBz_dx**2 + dBz_dy**2 + dBz_dz**2) 
    mean_gradient_Bz = np.mean(gradient_magnitude)
    
    return mean_gradient_Bz

def get_mean_gradient_total_magnetic_field(bx_3D, by_3D, bz_3D):
    """
    Compute the mean gradient of the total magnetic field in a Cartesian box.
    """

    B_magnitude = np.sqrt(bx_3D**2 + by_3D**2 + bz_3D**2)  # |B| (in Gauss)

    # Compute partial derivatives of |B| using central differences
    dB_dx = (np.roll(B_magnitude, -1, axis=0) - np.roll(B_magnitude, 1, axis=0)) / (2 * dx)
    dB_dy = (np.roll(B_magnitude, -1, axis=1) - np.roll(B_magnitude, 1, axis=1)) / (2 * dy)
    dB_dz = (np.roll(B_magnitude, -1, axis=2) - np.roll(B_magnitude, 1, axis=2)) / (2 * dz)

    # Compute the magnitude of the gradient of |B|
    gradient_magnitude = np.sqrt(dB_dx**2 + dB_dy**2 + dB_dz**2)
    
    # Compute the mean gradient
    mean_total_gradient = np.mean(gradient_magnitude)
    
    return mean_total_gradient

def get_total_magnitude_lorentz_force(bx_3D, by_3D, bz_3D):
    """
    Compute the total magnitude of the Lorentz force in a Cartesian box.
    """

    # Compute the curl of B
    curl_Bx = (np.roll(bz_3D, -1, axis=1) - np.roll(bz_3D, 1, axis=1)) / (2 * dy) - \
              (np.roll(by_3D, -1, axis=2) - np.roll(by_3D, 1, axis=2)) / (2 * dz)
    curl_By = (np.roll(bx_3D, -1, axis=2) - np.roll(bx_3D, 1, axis=2)) / (2 * dz) - \
              (np.roll(bz_3D, -1, axis=0) - np.roll(bz_3D, 1, axis=0)) / (2 * dx)
    curl_Bz = (np.roll(by_3D, -1, axis=0) - np.roll(by_3D, 1, axis=0)) / (2 * dx) - \
              (np.roll(bx_3D, -1, axis=1) - np.roll(bx_3D, 1, axis=1)) / (2 * dy)
    
    # Compute the Lorentz force components
    Fx = (curl_By * bz_3D - curl_Bz * by_3D) / (4 * np.pi)
    Fy = (curl_Bz * bx_3D - curl_Bx * bz_3D) / (4 * np.pi)
    Fz = (curl_Bx * by_3D - curl_By * bx_3D) / (4 * np.pi)

    # Compute the magnitude of the Lorentz force density
    F_magnitude = np.sqrt(Fx**2 + Fy**2 + Fz**2)

    # Compute the total Lorentz force by summing over the volume
    total_Lorentz_force = np.sum(F_magnitude) * dV

    return total_Lorentz_force

def get_total_unsigned_magnetic_flux(bx_3D, by_3D, bz_3D):
    """
    Compute the total unsigned magnetic flux in a Cartesian box.
    """

    # Area elements for faces
    dA_xy = dx * dy
    dA_xz = dx * dz
    dA_yz = dy * dz

    # Top and bottom faces (Bz normal)
    flux_top = np.sum(np.abs(bz_3D[:, :, -1]) * dA_xy)
    flux_bottom = np.sum(np.abs(bz_3D[:, :, 0]) * dA_xy)

    # Left and right faces (By normal)
    flux_left = np.sum(np.abs(by_3D[:, 0, :]) * dA_xz)
    flux_right = np.sum(np.abs(by_3D[:, -1, :]) * dA_xz)

    # Front and back faces (Bx normal)
    flux_front = np.sum(np.abs(bx_3D[0, :, :]) * dA_yz)
    flux_back = np.sum(np.abs(bx_3D[-1, :, :]) * dA_yz)

    # Total unsigned flux
    total_flux_of_AR_box = flux_top + flux_bottom + flux_left + flux_right + flux_front + flux_back
    
    return total_flux_of_AR_box

def main():
    # Load the volume file
    volume_path = sys.argv[1]
    try:
        bx_3D, by_3D, bz_3D = Utils.load_volume_components(volume_path)
    except Exception as e:
        print(f'ERROR: Failed to load volume components from {volume_path}: {repr(e)}')
        logger.log(f'Failed to load volume components from {volume_path}: {repr(e)}', 'HIGH')
        exit(1)

    # Bout_hmi.sharp_cea_720s.10960.20240314_170000_TAI.bin
    target_ar_gen = os.path.basename(volume_path)[5:-4]

    try:
        tot_mag_energy = get_total_magnetic_energy(bx_3D, by_3D, bz_3D, target_ar_gen)
        tot_unsigned_current_helicity = get_total_unsigned_current_helicity(bx_3D, by_3D, bz_3D)
        tot_abs_net_current_helicity = get_total_absolute_net_current_helicity(bx_3D, by_3D, bz_3D)
        mean_shear_angle = get_mean_shear_angle(bx_3D, by_3D, bz_3D)
        tot_unsigned_volume_vertical_current = get_total_unsigned_volume_vertical_current(bx_3D, by_3D, bz_3D)
        twist_param_alpha = get_twist_parameter_alpha(bx_3D, by_3D, bz_3D)
        mean_grad_vert_mag_field = get_mean_gradient_vertical_magnetic_field(bx_3D, by_3D, bz_3D)
        mean_grad_total_mag_field = get_mean_gradient_total_magnetic_field(bx_3D, by_3D, bz_3D)
        tot_mag_lorentz_force = get_total_magnitude_lorentz_force(bx_3D, by_3D, bz_3D)
        tot_unsigned_mag_flux = get_total_unsigned_magnetic_flux(bx_3D, by_3D, bz_3D)
    except Exception as e:
        print(f'ERROR: Failed to calculate volume parameters for targetARGen {target_ar_gen}: {repr(e)}')
        logger.log(f'Failed to calculate volume parameters for targetARGen {target_ar_gen}: {repr(e)}', 'HIGH')
        exit(1)

    # Make a new DF with the calculated parameters as one row and save it as a pickle file.
    df = pd.DataFrame(columns=['Filename General', 'Total Magnetic Energy', 'Total Unsigned Current Helicity', 'Total Absolute Net Current Helicity', 'Mean Shear Angle', 'Total Unsigned Volume Vertical Current', 'Twist Parameter Alpha', 'Mean Gradient of Vertical Magnetic Field', 'Mean Gradient of Total Magnetic Field', 'Total Magnitude of Lorentz Force', 'Total Unsigned Magnetic Flux'])
    df.loc[0] = [target_ar_gen, tot_mag_energy, tot_unsigned_current_helicity, tot_abs_net_current_helicity, mean_shear_angle, tot_unsigned_volume_vertical_current, twist_param_alpha, mean_grad_vert_mag_field, mean_grad_total_mag_field, tot_mag_lorentz_force, tot_unsigned_mag_flux]
    # save as pickle file to OutputData/Temp with same name as target_ar_gen
    df.to_pickle(os.path.join(rootDir, 'OutputData', 'Temp', f'{target_ar_gen}.pkl'))

if __name__ == '__main__':
    logger = Logger('VolumeParameterizationLog.txt')

    # global vars
    dx = 0.0
    dy = 0.0
    dz = 0.0
    dV = 0.0
    hc = 0.0
    jx = 0.0
    jy = 0.0
    jz = 0.0

    main()