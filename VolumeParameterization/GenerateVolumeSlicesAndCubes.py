import os
import sys
import pathlib
import sunpy.map
from skimage.transform import resize
from scipy.ndimage import label, generate_binary_structure
import numpy as np
from tqdm import tqdm

rootDir = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.insert(1, os.path.join(rootDir))

from Utils import get_region_of_interest_planes_and_cube, load_volume_components

GENERATED_VOLUMES_PATH_SINGLE_BLOB = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumes'
GENERATED_VOLUMES_PATH_MULTI_BLOB = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumesMultiblob'

REGULAR_SHARED_DATA_DIR = os.path.join(os.sep + 'share', 'development', 'data', 'drms', 'MagPy_Shared_Data')
DEFINITIVE_SHARP_DATA_DIR = os.path.join(REGULAR_SHARED_DATA_DIR, 'TrainingData' + os.sep)

OUTPUT_PATH = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/VolumeSlicesAndCubes'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

"""
Go through all the generated volumes and use the util function to get the planes and cubes
for each of them. Save the planes and cubes in the output path.
"""

def generate_slices_and_cubes(volume_dir):
    volume_files = os.listdir(volume_dir)
    volume_files.sort()
    for volume_path in tqdm(volume_files, desc='Generating slices and cubes for volumes in ' + volume_dir):
        # e.g., Bout_hmi.sharp_cea_720s.7115.20170903_050000_TAI.bin
        filename_general = volume_path[5:-4] # hmi...TAI

        bx_3D, by_3D, bz_3D = load_volume_components(os.path.join(volume_dir, volume_path))

        associated_bitmap_path = os.path.join(DEFINITIVE_SHARP_DATA_DIR, filename_general + '.bitmap.fits')
        bitmap = sunpy.map.Map(associated_bitmap_path)

        # Resize bitmap data to 200x400
        bitmap_resized = resize(bitmap.data, (200, 400), anti_aliasing=True, preserve_range=True)

        # weak and strong field pixels within the HARP = (33, 34). A little bit of error allowed here due to
        # interpolation from resizing function. The next lowest values in the bitmap are low enough that we
        # can easily look from 30 and up.
        mask_resized = bitmap_resized > 30
        blob_mask_resized = bitmap_resized*mask_resized.astype(int)*1.

        # Separate out blobs
        s = generate_binary_structure(2,2)  # Allows diagonal pixels to be considered part of the same blob

        labeled_resized, nblobs_resized = label(blob_mask_resized, structure=s)

        # Sort blobs in order from greatest area to least area (relevant to AR # identification in MagPy, so needed for cross-referencing)
        # The assumption is that resizing the bitmap won't change the relative sizes of the blobs to each other.
        blobs_resized = [i for i in range(1, nblobs_resized+1)]
        blobs_resized = sorted(blobs_resized, key=lambda x: np.count_nonzero(labeled_resized == x), reverse=True)

        # Iterate through each blob and get the planes and cubes
        for i in range(1, nblobs_resized+1):
            curr_blob_index = i

            curr_blob = blobs_resized[curr_blob_index - 1]

            planes_xy_path = os.path.join(OUTPUT_PATH, filename_general + '_blob' + str(curr_blob_index) + '_planes_xy.npy')
            planes_xz_path = os.path.join(OUTPUT_PATH, filename_general + '_blob' + str(curr_blob_index) + '_planes_xz.npy')
            planes_yz_path = os.path.join(OUTPUT_PATH, filename_general + '_blob' + str(curr_blob_index) + '_planes_yz.npy')
            cube_path = os.path.join(OUTPUT_PATH, filename_general + '_blob' + str(curr_blob_index) + '_cube.npy')

            if os.path.exists(planes_xy_path) and os.path.exists(planes_xz_path) and os.path.exists(planes_yz_path) and os.path.exists(cube_path):
                continue

            # Create a new volume for the blob. That is, mask out ~blob pixels at every height of the volume, which is in the resized scale.
            mask = labeled_resized == curr_blob

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

            planes_xy, planes_xz, planes_yz, cube = get_region_of_interest_planes_and_cube(bitmap, bx_3D_blob, by_3D_blob, bz_3D_blob)

            # Save the planes and cube
            np.save(planes_xy_path, planes_xy)
            np.save(planes_xz_path, planes_xz)
            np.save(planes_yz_path, planes_yz)
            np.save(cube_path, cube)


print('Generating slices and cubes for volumes in ' + GENERATED_VOLUMES_PATH_SINGLE_BLOB)
generate_slices_and_cubes(GENERATED_VOLUMES_PATH_SINGLE_BLOB)

print('Generating slices and cubes for volumes in ' + GENERATED_VOLUMES_PATH_MULTI_BLOB)
generate_slices_and_cubes(GENERATED_VOLUMES_PATH_MULTI_BLOB)

print('Finished generating slices and cubes for all volumes.')
