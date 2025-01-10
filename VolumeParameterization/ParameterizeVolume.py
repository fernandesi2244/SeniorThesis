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
import datetime

rootDir = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(1, rootDir)

import Utils
from Logger import Logger
from SRSHandler import SRSHandler

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

def get_AR_num(labeled, blob_i, arList, blobNum, targetARGen):
    """
    Gets the AR number of the blob indicated by blob_i.

    Returns a tuple where the first element is the list of active regions
    that are associated with the blob. The first element in this list is
    the AR that the algorithm thinks is associated with the blob. The
    elements after that are any other ARs that have coordinates within the
    blob but are not the most prevalent. The second element of the tuple is
    whether or not the blob has plage, as indicated by the NOAA SRS file.
    The third and fourth elements are the Carrington latitude and longtiude
    coordinates of the blob, respectively.

    :param labeled: the array of labeled blobs
    :param blob_i: the the label of the current blob being examined
    :param arList: the list of ARs from the SRS file
    :param blobNum: the blob iteration we are currently on (not equal to the number of the blob—which is sorted by blob area)
    :param targetARGen: the 'general' file name of the SHARP
    """
    
    # Creates 2D array that is the same size as the HARP's maps that labels just the current blob being examined.
    # Background value = 0; Blob value = some value x, such that x > 0.
    mask = labeled == blob_i
    currentBlob = labeled*mask

    # Reverses the order of the rows in the currentBlob array. This is necessary because the map data (labeled
    # array) is reversed in the vertical direction.
    currentBlob = np.flip(currentBlob, axis=0)

    # Get pixel/location information about the HARP
    # Keep in mind that the Br map has the same scale and size as the bitmap.
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

    totalLatPixels = len(brMap.data)
    totalLonPixels = len(brMap.data[0])

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

    ARsInBlob = list()
    ARsOutsideBlob = list()

    if len(arList) == 0:
        logger.log(f'There were no SRS entries in the file for HARP {targetARGen}', 'LOW')

    # For each AR in the SRS file, see if it's in the current blob, and make other relevant calculations
    for srsAR in arList:
        # Search for N or S followed by any number of digits (that may include a decimal point)
        latPattern = re.compile('([NS])([\d.]+)')

        # Locate latitude segment from location string
        latMatch = re.search(latPattern, srsAR.location)
        if latMatch:
            # group(1) extracts the cardinal direction character of the latitude (using 1st set of parentheses in regex pattern)
            latCardinal = latMatch.group(1)

            # group(2) extracts the numerical part of the latitude (using 2nd set of parentheses in regex pattern)
            latAbs = float(latMatch.group(2))

            # Use cardinal direction to determine signed latitude
            latSigned = latAbs if latCardinal == 'N' else -latAbs

            # Retrieve Carrington Heliographic longitude (from SRS file)
            if srs.carrLongitude is not None:
                lonSigned = srsAR.carrLongitude
            else:
                print('Ignoring AR with no Carrington longitude')
                continue
        else:
            logger.log(f"Couldn't identify latitude part of the following location: {srsAR.location}", 'MEDIUM')
            print("Couldn't identify latitude part of location.")
            continue
        
        # At this point we have the signed lat and lon of the current AR from the SRS file
        arLat = latSigned
        arLon = lonSigned
        
        # Retrieve the pixel in the HARP array that corresponds with the AR's latitude and longitude coordinates
        arLatPixel = int(latReferencePixel - (arLat-latReferencePoint)/latDegIncrement) # The higher the latitude, the lower the row index
        arLonPixel = int(lonReferencePixel + (arLon-lonReferencePoint)/lonDegIncrement) # The higher the longitude, the higher the col index

        # Check if corresponding pixel has a valid row and column index in the currentBlob array
        if arLatPixel < 0 or arLatPixel >= totalLatPixels or arLonPixel < 0 or arLonPixel >= totalLonPixels:
            continue

        if currentBlob[arLatPixel][arLonPixel] > 0:
            # The current AR is in the blob!

            # Calculate distance of AR from blob centroid
            distance = math.sqrt(math.pow(arLonPixel - colCenter, 2) + math.pow(arLatPixel - rowCenter, 2))
            srsAR.setDistanceFromBlob(distance)

            ARsInBlob.append(srsAR)
        else:
            # The current AR is within the HARP but outside the current blob

            # Quickly calculate closest distance of AR to blob boundary
            blobIndices = np.argwhere(currentBlob > 0)
            blobRows = blobIndices[:,0]
            blobColumns = blobIndices[:,1]
            distances = np.sqrt((blobRows-arLatPixel)**2 + (blobColumns-arLonPixel)**2)
            closestDistance = np.amin(distances)
            srsAR.setDistanceFromBlob(closestDistance)

            ARsOutsideBlob.append(srsAR)

    print('Printing ARs in blob:', ARsInBlob)

    # Initially assume that there is no NOAA AR associated with the blob and that it doesn't have plage
    arNum = None
    hasPlage = False

    # Add all of the AR #s within the blob to the list of "taken" ARs; we don't want other small blobs nearby trying to take these 
    # numbers from the FITS header for themselves.
    [takenARs.append(AR.arNum) for AR in ARsInBlob]

    if len(ARsInBlob) == 1:
        # Only one AR; assign it
        hasPlage = ARsInBlob[0].hasPlage
        arNum = ARsInBlob[0].arNum
    elif len(ARsInBlob) > 1:
        # If there are multiple active regions within the blob, then we will first look through those with sunspots and choose the 
        # one that has the greatest area. If there are no active regions with sunspots in the blob, only then will we try to associate
        # an active region with plage with the blob. This is because active region assignments with sunspots are preferred.

        # If multiple active regions with sunspots are within the blob, choose the one with the largest area.
        ARsWithSunspots = list(filter(lambda x: not x.hasPlage, ARsInBlob))  # Only keep ARs that don't have plage
        ARsWithSunspots = sorted(ARsWithSunspots, key=lambda x: x.area, reverse=True)   # Sorts ARs in descending order based on area

        if len(ARsWithSunspots) > 0:
            # Return the AR # with the greatest area
            hasPlage = False
            arNum = ARsWithSunspots[0].arNum
        else:
            # In this case, there are only active regions with plage in the blob
            ARsWithPlage = sorted(ARsInBlob, key=lambda x: x.distanceFromBlob)
            closestAR = ARsWithPlage[0]

            logger.log(f'AR {closestAR.arNum} WITH PLAGE inside the blob {blobNum} was assigned to it. Distance = {closestAR.distanceFromBlob:.2f} pixels. Blob within HARP {targetARGen}', 'LOW')
            hasPlage = True
            arNum = closestAR.arNum
    else:
        # Try to find the closest AR to the blob boundary if no ARs are detected within the blob
        ARs = ARsOutsideBlob[:]  # ARs outside blob (if any)

        # Remove ARs that are already taken
        ARs = list(filter(lambda x: x.arNum not in takenARs, ARs))

        # Sort by distance from blob boundary in ascending order so that the first element is the closest one
        ARs = sorted(ARs, key=lambda x: x.distanceFromBlob)

        # If the AR is too far from the blob boundary (greater than 50 pixels), we do not want to associate the two.
        # 33 pixels comes from the 1 degree error in the NOAA SRS locations combined with the fact that there are 0.03 degrees per HARP pixel
        # The other 17 pixels come from the need to account for the changing shape of blobs throughout the day.
        # This adds up to an error of 1.5 degrees.
        if len(ARs) > 0 and ARs[0].distanceFromBlob <= 50:
            takenARs.append(ARs[0].arNum)
            hasPlage = ARs[0].hasPlage
            if hasPlage:
                logger.log(f'AR {ARs[0].arNum} WITH PLAGE outside the blob {blobNum} was assigned to it. Distance = {ARs[0].distanceFromBlob:.2f} pixels. Blob within HARP {targetARGen}', 'LOW')
            else:
                logger.log(f'AR {ARs[0].arNum} WITH SUNSPOT outside the blob {blobNum} was assigned to it. Distance = {ARs[0].distanceFromBlob:.2f} pixels. Blob within HARP {targetARGen}', 'LOW')
            arNum = ARs[0].arNum
        else:
            # As a last resort, refer to the NOAA active regions provided in the FITS header
            print('No ARs matched up with the blob. Attempting lookup in FITS header...')
            relARs = brMap.meta['NOAA_ARS']
            if relARs != 'MISSING' and relARs != '': # MISSING in Lookdata actually means ''. However, take both into account just in case.
                try:
                    relARs = relARs.replace('[', '').replace(']', '').split(',') # Remove brackets and split on commas
                    relARs = [int(i) for i in relARs] # Cast each element as int
                    for ar in relARs:
                        if ar not in takenARs:
                            logger.log(f'FITS header AR {ar} assigned to blob {blobNum} within HARP {targetARGen}', 'LOW')
                            takenARs.append(ar)
                            arNum = ar
                            break
                except:
                    # Something went wrong when parsing the AR list; continue on.
                    pass
    
    # If arNum is still None, assign unique ID
    if arNum is None:
        logger.log(f'Assigning unique ID to blob {blobNum} within HARP {targetARGen}', 'LOW')
        print('No ARs could be associated with the blob. Assigning unique ID...')
        arNum = f"{brMap.meta['HARPNUM']}-{blobNum}"
        # Note that using blobNum usually means that for the same HARP at different times,
        # if all other identification schemes fail, you can depend on the biggest blob being
        # [HARPNUM]-1, with increasing numbers ([HARPNUM]-2) corresponding to smaller blobs.
        # This is because we iterated from the largest to smallest blob in the beginning.

    # First element to return is a list in the format: [most likely AR #, all other AR #s within blob], where
    # each element is a single AR number.
    ARsToReturn = [arNum]
    [ARsToReturn.append(ar.arNum) for ar in ARsInBlob if ar.arNum not in ARsToReturn]   # Ensure no duplicates while preserving order.

    return (ARsToReturn, hasPlage, blobLat, blobLon)

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
    
    # Get the list of ARs from the SRS file to be used by the get_AR_num function
    dtStr = bitmap.meta['T_REC'][:19] #string of datetime without the _TAI
    HARPDate = datetime.datetime.strptime(dtStr, '%Y.%m.%d_%H:%M:%S') #convert string to datetime object
    srsHandler = SRSHandler(HARPDate)
    srsHandler.downloadSRS()
    arList = srsHandler.getARList()

    # For now, do exact cutouts of the blobs from the x, y, z components of the volume.
    # This means use the mask to cut out each blob at each z level of the original volume.
    segmented_volumes = []
    for i, volume_blob_num in enumerate(blobs_resized):
        if i >= len(blobs):
            break

        arNums, hasPlage, blobLat, blobLon = get_AR_num(labeled, blobs[i], arList, i + 1, target_ar_gen)

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

        segmented_volumes.append((bx_3D_blob, by_3D_blob, bz_3D_blob, arNums, blobLat, blobLon, i + 1))
    
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

    # Each segmented volume represents a separate blob in the SHARP
    segmented_volumes = get_segmented_volumes(bx_3D, by_3D, bz_3D)

    for (bx_3D_blob, by_3D_blob, bz_3D_blob, ar_nums, blob_lat, blob_lon, blob_index) in segmented_volumes:
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
        df = pd.DataFrame(columns=['Filename General', 'Relevant Active Regions', 'Latitude', 'Carrington Longitude', 'Total Magnetic Energy', 'Total Unsigned Current Helicity', 'Total Absolute Net Current Helicity', 'Mean Shear Angle', 'Total Unsigned Volume Vertical Current', 'Twist Parameter Alpha', 'Mean Gradient of Vertical Magnetic Field', 'Mean Gradient of Total Magnetic Field', 'Total Magnitude of Lorentz Force', 'Total Unsigned Magnetic Flux'])
        df.loc[0] = [target_ar_gen, str(ar_nums), blob_lat, blob_lon, tot_mag_energy, tot_unsigned_current_helicity, tot_abs_net_current_helicity, mean_shear_angle, tot_unsigned_volume_vertical_current, twist_param_alpha, mean_grad_vert_mag_field, mean_grad_total_mag_field, tot_mag_lorentz_force, tot_unsigned_mag_flux]
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