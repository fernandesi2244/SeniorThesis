import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

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

def get_point_of_interest(bz_3D):
    """
    Look at the bottom layer of the magnetic field volume formed
    by the 3-D magnetic field components.

    Mask out quiet sun noise just for the next step: Find the positive-weighted
    centroid of the layer as well as the negative-weighted centroid.

    Return the center point of the line connecting the two centroids.
    """

    # NOTE: Each array is 200x400x100
    bz_surf = bz_3D[:, :, 0]

    # Print the average value of the Bz surface
    print('Mean of bz surf:', np.mean(bz_surf))

    quiet_noise = np.abs(bz_surf) <= 100
    bz_surf_without_noise = np.copy(bz_surf)
    bz_surf_without_noise[quiet_noise] = 0

    # Find the positive-weighted centroid of the layer
    bz_surf_positive = np.copy(bz_surf_without_noise)
    bz_surf_positive[bz_surf_positive < 0] = 0

    if np.sum(bz_surf_positive) == 0:
        print('All pixels are quiet sun')
        return bz_surf_positive.shape[0] // 2, bz_surf_positive.shape[1] // 2

    centroid_row, centroid_col = ndimage.center_of_mass(bz_surf_positive)
    centroid_row, centroid_col = round(centroid_row), round(centroid_col)

    # Find the negative-weighted centroid of the layer
    bz_surf_negative = np.copy(bz_surf_without_noise)
    bz_surf_negative[bz_surf_negative > 0] = 0
    centroid_row_neg, centroid_col_neg = ndimage.center_of_mass(bz_surf_negative)
    centroid_row_neg, centroid_col_neg = round(centroid_row_neg), round(centroid_col_neg)

    # Find the center point of the line connecting the two centroids
    center_row = (centroid_row + centroid_row_neg) // 2
    center_col = (centroid_col + centroid_col_neg) // 2

    # Plot the Bz map along with the centroids, a line connecting them, and a point
    # at the center of that line.
    plt.figure()
    plt.imshow(bz_surf, cmap='gray')
    plt.plot(centroid_col, centroid_row, 'ro')
    plt.plot(centroid_col_neg, centroid_row_neg, 'bo')
    plt.plot([centroid_col, centroid_col_neg], [centroid_row, centroid_row_neg], 'g')
    plt.plot(center_col, center_row, 'yo')
    plt.show()

    return center_row, center_col

def get_volume_region_of_interest(bx_3D, by_3D, bz_3D):
    """
    Get the region of interest in the 3D magnetic field volume.

    Look at the center point of the line connecting the two centroids.
    Find which height in the volume 2 MegaMeters corresponds to.
    Make a box centered at that point and height
    that extends 10 pixels in each cardinal direction. This is the region of
    interest. Return it. If the region cannot be constructed because the
    center point is too close to the edge, construct as much as possible and
    fill in the remaining space with zeros.
    """

    center_row, center_col = get_point_of_interest(bz_3D)

    # Find the height in the volume that corresponds to 2 MegaMeters
    height = 2
    height_index = 20 # TODO: Find the height index that corresponds to 2 MegaMeters

    # Construct a box centered at the center point that extends 25 pixels in each
    # cardinal direction. If the region cannot be constructed because the center
    # point is too close to the edge, construct as much as possible and fill in
    # the remaining space with zeros.
    region = np.zeros((20, 20, 20, 3))

    # Construct the region
    for i in range(20):
        for j in range(20):
            for k in range(20):
                if center_row - 10 + i >= 0 and center_row - 10 + i < 200 and \
                   center_col - 10 + j >= 0 and center_col - 10 + j < 400 and \
                   height_index - 10 + k >= 0 and height_index - 10 + k < 100:
                    region[i, j, k, 0] = bx_3D[center_row - 10 + i, center_col - 10 + j, height_index - 10 + k]
                    region[i, j, k, 1] = by_3D[center_row - 10 + i, center_col - 10 + j, height_index - 10 + k]
                    region[i, j, k, 2] = bz_3D[center_row - 10 + i, center_col - 10 + j, height_index - 10 + k]

    return region

if __name__ == "__main__":
    bx_3D, by_3D, bz_3D = load_volume_components('C:/Users/ibfernan/Documents/SampleVolumes/Bout_hmi.sharp_cea_720s.7115.20170903_050000_TAI.bin')
    region_of_interest = get_volume_region_of_interest(bx_3D, by_3D, bz_3D)
    print(region_of_interest.shape)
    # plot bottom layer of Bz
    plt.imshow(region_of_interest[:, :, 0, 2], cmap='gray')
    plt.show()

    bx_3D, by_3D, bz_3D = load_volume_components('C:/Users/ibfernan/Documents/VolumeRunTest/Bout_hmi.sharp_cea_720s.10000.20230826_210000_TAI.bin')
    region_of_interest = get_volume_region_of_interest(bx_3D, by_3D, bz_3D)
    plt.imshow(region_of_interest[:, :, 0, 2], cmap='gray')
    plt.show()

    bx_3D, by_3D, bz_3D = load_volume_components('C:/Users/ibfernan/Documents/VolumeRunTest/Bout_hmi.sharp_cea_720s.10495.20231219_130000_TAI.bin')
    region_of_interest = get_volume_region_of_interest(bx_3D, by_3D, bz_3D)
    plt.imshow(region_of_interest[:, :, 0, 2], cmap='gray')
    plt.show()

    bx_3D, by_3D, bz_3D = load_volume_components('C:/Users/ibfernan/Documents/VolumeRunTest/Bout_hmi.sharp_cea_720s.10495.20231219_130000_TAI_take2.bin')
    region_of_interest = get_volume_region_of_interest(bx_3D, by_3D, bz_3D)
    plt.imshow(region_of_interest[:, :, 0, 2], cmap='gray')
    plt.show()

    bx_3D, by_3D, bz_3D = load_volume_components('C:/Users/ibfernan/Documents/VolumeRunTest/Bout_hmi.sharp_cea_720s.10965.20240314_230000_TAI.bin')
    region_of_interest = get_volume_region_of_interest(bx_3D, by_3D, bz_3D)
    plt.imshow(region_of_interest[:, :, 0, 2], cmap='gray')
    plt.show()

    bx_3D, by_3D, bz_3D = load_volume_components('C:/Users/ibfernan/Documents/VolumeRunTest/Bout_hmi.sharp_cea_720s.10965.20240314_230000_TAI_take2.bin')
    region_of_interest = get_volume_region_of_interest(bx_3D, by_3D, bz_3D)
    plt.imshow(region_of_interest[:, :, 0, 2], cmap='gray')
    plt.show()
    