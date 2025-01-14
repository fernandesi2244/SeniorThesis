"""
Given a volume file path, load the volume and calculate various parameters for the volume,
e.g., the total magnetic energy. Save these parameters to a CSV file.
"""

import os
import sys
import numpy as np
import pandas as pd
import sunpy.map
import math
import pathlib
from scipy.ndimage import label, generate_binary_structure
from skimage.transform import resize
from scipy import ndimage

rootDir = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(1, os.path.join(rootDir))

import Utils
from Logger import Logger

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
    cdelt1 = (math.atan((bitmap.meta['rsun_ref']*bitmap.meta['cdelt1']*np.pi/180)/(bitmap.meta['dsun_obs'])))*(180/np.pi)*(3600)  
    dx = (cdelt1*(bitmap.meta['rsun_ref']/bitmap.meta['rsun_obs'])*100.0) 

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

def get_blob_loc(labeled, blob_i):
    """
    Gets the lat/lon of the centroid of the blob indicated by blob_i.

    Returns a tuple where the first and second components are the
    latitude and Carrington longitude of the blob centroid, respectively.

    :param labeled: the array of labeled blobs
    :param blob_i: the the label of the current blob being examined
    """
    
    # Creates 2D array that is the same size as the HARP's maps that labels just the current blob being examined.
    # Background value = 0; Blob value = some value x, such that x > 0.
    mask = labeled == blob_i
    currentBlob = labeled*mask

    # Reverses the order of the rows in the currentBlob array. This is necessary because the map data (labeled
    # array) is reversed in the vertical direction.
    currentBlob = np.flip(currentBlob, axis=0)

    # Get pixel/location information about the HARP
    lonReferencePoint = bitmap.meta['CRVAL1']      # This is at the center of the HARP (in deg); in Heliographic Carrington
    latReferencePoint = bitmap.meta['CRVAL2']      # This is at the center of the HARP (in deg); in Heliographic Carrington
    lonDegIncrement = bitmap.meta['CDELT1']        # Degree increment per pixel in longitudinal direction
    latDegIncrement = bitmap.meta['CDELT2']        # Degree increment per pixel in latitudinal direction
    lonReferencePixel = bitmap.meta['CRPIX1'] - 1  # This is the column (longitude) pixel value at HARP center (one-based indexing)
    latReferencePixel = bitmap.meta['CRPIX2'] - 1  # This is the row (latitude) pixel value at HARP center (one-based indexing)

    # For some reason, sometimes Carrington longitudes from FITS headers are outside of normal range
    while(lonReferencePoint >= 360):
        lonReferencePoint -= 360
    while(lonReferencePoint < 0):
        lonReferencePoint += 360

    # Calculate the centroid of the blob (in pixel coordinates)
    rowCenter, colCenter = ndimage.measurements.center_of_mass(currentBlob)
    rowCenter, colCenter = round(rowCenter), round(colCenter)

    # Get the latitude and longitude coordinates of the blob centroid
    blobLat = latReferencePoint - (rowCenter-latReferencePixel) * latDegIncrement
    blobLon = lonReferencePoint + (colCenter-lonReferencePixel) * lonDegIncrement
    while(blobLon >= 360):
        blobLon -= 360
    while(blobLon < 0):
        blobLon += 360

    return (blobLat, blobLon)

def get_segmented_volumes(bx_3D, by_3D, bz_3D):
    # Resize bitmap data to 200x400
    bitmap_resized = resize(bitmap.data, (200, 400), anti_aliasing=True, preserve_range=True)

    # weak and strong field pixels within the HARP = (33, 34). A little bit of error allowed here due to
    # interpolation from resizing function. The next lowest values in the bitmap are low enough that we
    # can easily look from 30 and up.
    mask_resized = bitmap_resized > 30
    blob_mask_resized = bitmap_resized*mask_resized.astype(int)*1.

    mask = bitmap.data > 30
    blob_mask = bitmap.data*mask.astype(int)*1.

    # Separate out blobs
    s = generate_binary_structure(2,2)  # Allows diagonal pixels to be considered part of the same blob

    labeled_resized, nblobs_resized = label(blob_mask_resized, structure=s)
    labeled, nblobs = label(blob_mask, structure=s)

    # Sort blobs in order from greatest area to least area (relevant to AR # identification in MagPy, so needed for cross-referencing)
    # The assumption is that resizing the bitmap won't change the relative sizes of the blobs to each other.
    blobs_resized = [i for i in range(1, nblobs_resized+1)]
    blobs_resized = sorted(blobs_resized, key=lambda x: np.count_nonzero(labeled_resized == x), reverse=True)

    blobs = [i for i in range(1, nblobs+1)]
    blobs = sorted(blobs, key=lambda x: np.count_nonzero(labeled == x), reverse=True)

    # Confirm that the number of blobs is the same in both the resized and original bitmaps and that
    # the relative sizes of the blobs are the same.
    if nblobs_resized != nblobs:
        print('ERROR: Number of blobs in resized bitmap does not match number of blobs in original bitmap.')
        if nblobs_resized < nblobs:
            logger.log(f'Number of blobs in resized bitmap less than number of blobs in original bitmap for SHARP {target_ar_gen}.\
                blobs: {blobs}. blobs_resized: {blobs_resized}', 'MEDIUM')
        else:
            logger.log(f'Unexpected: number of blobs in resized bitmap greater than number of blobs in original bitmap for SHARP {target_ar_gen}.\
                blobs: {blobs}. blobs_resized: {blobs_resized}', 'HIGH')
        
        # Actually an okay heuristic to continue here, since we only expect this to happen by blobs in the resized bitmap getting
        # "smoothed out" or small artificial artifacts being generated during resizing. Since this would only happen for very small
        # blobs, and because both blob lists are sorted by size in decreasing order, we can continue to pairwise-relate blobs at every
        # position in the lists from the beginning until one of the lists becomes exhausted.

    # For now, do exact cutouts of the blobs from the x, y, z components of the volume.
    # This means use the mask to cut out each blob at each z level of the original volume.
    segmented_volumes = []
    for i, volume_blob_num in enumerate(blobs_resized):
        if i >= len(blobs):
            break

        blobLat, blobLon = get_blob_loc(labeled, blobs[i])

        # Create a new volume for the blob. That is, mask out ~blob pixels at every height of the volume, which is in the resized scale.
        mask = labeled_resized == volume_blob_num

        # In order to multiply each 3D component by the mask, we need to make the first dimension of the volume the height.
        # This is because the mask has the same shape as the volume at a given height, and broadcasting only works on the
        # first dimension.
        bx_3D_blob = np.transpose(bx_3D, (2, 0, 1)) # height dimension, then number of rows, then number of cols
        by_3D_blob = np.transpose(by_3D, (2, 0, 1))
        bz_3D_blob = np.transpose(bz_3D, (2, 0, 1))

        bx_3D_blob = bx_3D_blob * mask
        by_3D_blob = by_3D_blob * mask
        bz_3D_blob = bz_3D_blob * mask

        # Transpose back to original shape
        bx_3D_blob = np.transpose(bx_3D_blob, (1, 2, 0))
        by_3D_blob = np.transpose(by_3D_blob, (1, 2, 0))
        bz_3D_blob = np.transpose(bz_3D_blob, (1, 2, 0))

        segmented_volumes.append((bx_3D_blob, by_3D_blob, bz_3D_blob, blobLat, blobLon, i + 1))
    
    return segmented_volumes

def main():
    global target_ar_gen, bitmap

    # Load the volume file
    volume_path = sys.argv[1]

    # Bout_hmi.sharp_cea_720s.10960.20240314_170000_TAI.bin
    target_ar_gen = os.path.basename(volume_path)[5:-4]

    try:
        bx_3D, by_3D, bz_3D = Utils.load_volume_components(volume_path)
    except Exception as e:
        print(f'ERROR: Failed to load volume components from {volume_path}: {repr(e)}')
        logger.log(f'Failed to load volume components from {volume_path}: {repr(e)}', 'HIGH')
        exit(1)
    
    # Load the associated bitmap for any operations that need targetARGen metadata or bitmap data
    associated_bitmap_path = os.path.join(DEFINITIVE_SHARP_DATA_DIR, target_ar_gen + '.bitmap.fits')
    bitmap = sunpy.map.Map(associated_bitmap_path)

    try:
        # Each segmented volume represents a separate blob in the SHARP
        segmented_volumes = get_segmented_volumes(bx_3D, by_3D, bz_3D)
    except Exception as e:
        print(f'ERROR: Failed to segment volumes for targetARGen {target_ar_gen}: {repr(e)}')
        logger.log(f'Failed to segment volumes for targetARGen {target_ar_gen}: {repr(e)}', 'HIGH')
        exit(1)

    for (bx_3D_blob, by_3D_blob, bz_3D_blob, blob_lat, blob_lon, blob_index) in segmented_volumes:
        try:
            tot_mag_energy = get_total_magnetic_energy(bx_3D_blob, by_3D_blob, bz_3D_blob, target_ar_gen)
            tot_unsigned_current_helicity = get_total_unsigned_current_helicity(bx_3D_blob, by_3D_blob, bz_3D_blob)
            tot_abs_net_current_helicity = get_total_absolute_net_current_helicity(bx_3D_blob, by_3D_blob, bz_3D_blob)
            mean_shear_angle = get_mean_shear_angle(bx_3D_blob, by_3D_blob, bz_3D_blob)
            tot_unsigned_volume_vertical_current = get_total_unsigned_volume_vertical_current(bx_3D_blob, by_3D_blob, bz_3D_blob)
            twist_param_alpha = get_twist_parameter_alpha(bx_3D_blob, by_3D_blob, bz_3D_blob)
            mean_grad_vert_mag_field = get_mean_gradient_vertical_magnetic_field(bx_3D_blob, by_3D_blob, bz_3D_blob)
            mean_grad_total_mag_field = get_mean_gradient_total_magnetic_field(bx_3D_blob, by_3D_blob, bz_3D_blob)
            tot_mag_lorentz_force = get_total_magnitude_lorentz_force(bx_3D_blob, by_3D_blob, bz_3D_blob)
            tot_unsigned_mag_flux = get_total_unsigned_magnetic_flux(bx_3D_blob, by_3D_blob, bz_3D_blob)
        except Exception as e:
            print(f'ERROR: Failed to calculate volume parameters for targetARGen {target_ar_gen}: {repr(e)}')
            logger.log(f'Failed to calculate volume parameters for targetARGen {target_ar_gen}: {repr(e)}', 'HIGH')
            exit(1)

        # Make a new DF with the calculated parameters as one row and save it as a pickle file.
        df = pd.DataFrame(columns=['Filename General', 'Latitude', 'Carrington Longitude', 'Volume Total Magnetic Energy', 'Volume Total Unsigned Current Helicity', 'Volume Total Absolute Net Current Helicity', 'Volume Mean Shear Angle', 'Volume Total Unsigned Volume Vertical Current', 'Volume Twist Parameter Alpha', 'Volume Mean Gradient of Vertical Magnetic Field', 'Volume Mean Gradient of Total Magnetic Field', 'Volume Total Magnitude of Lorentz Force', 'Volume Total Unsigned Magnetic Flux'])
        df.loc[0] = [target_ar_gen, blob_lat, blob_lon, tot_mag_energy, tot_unsigned_current_helicity, tot_abs_net_current_helicity, mean_shear_angle, tot_unsigned_volume_vertical_current, twist_param_alpha, mean_grad_vert_mag_field, mean_grad_total_mag_field, tot_mag_lorentz_force, tot_unsigned_mag_flux]
        # save as pickle file to OutputData/Temp with same name as target_ar_gen
        df.to_pickle(os.path.join(rootDir, 'OutputData', 'Temp', f'{target_ar_gen}-{blob_index}.pkl'))

if __name__ == '__main__':
    logger = Logger('VolumeParameterizationLog.txt')

    # global vars
    target_ar_gen = None
    bitmap = None

    dx = 0.0
    dy = 0.0
    dz = 0.0
    dV = 0.0
    hc = 0.0
    jx = 0.0
    jy = 0.0
    jz = 0.0

    takenARs = []

    main()