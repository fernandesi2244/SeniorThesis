import struct
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
import os

# Generate the movie using moviepy
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

OUTPUT_DIR = '/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumeImages'

def load_save_3D_output(filename="Bout.bin", alf=-99.9): # Read in 3-D magnetic field in the box from C-code output.
    # Load binary data from file
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.float64)  # Assuming double precision float

    nx, ny, nz, nd = 200, 400, 100, 3

    # Set dimensions from 'grid.ini'
    # with open('grid.ini', 'r') as f:
    #     params = f.readlines()
    #     nx, ny, nz, nd = int(params[1]), int(params[3]), int(params[5]), int(params[9])
        
    nynz = ny * nz
    nxnynz = nx * ny * nz

    # Initialize 3D magnetic field arrays
    bx3D = data[:nxnynz].reshape((nx, ny, nz))
    by3D = data[nxnynz:2*nxnynz].reshape((nx, ny, nz))
    bz3D = data[2*nxnynz:3*nxnynz].reshape((nx, ny, nz))
   
    # Calculate magnitude of B-field
    b3dabs = np.sum(bx3D**2 + by3D**2 + bz3D**2)
    print('magnetic field energy = ',b3dabs)
    print('Shape of bx3D:', bx3D.shape)

    return bx3D, by3D, bz3D
    
    # Save B-field data to a file (equivalent to .sav in IDL)

all_files = os.listdir('/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumes')
#shuffle files
import random
random.shuffle(all_files)
#select the first 1000 files
selected_files = all_files[:1000]

# For each file, generate the bz_photo for it as done above.
# Aggregate all of the bz_photos into a movie file (e.g. bz_movie.mp4).

files0 = []
files25 = []
files50 = []
files75 = []
files99 = []

levels = [0, 25, 50, 75, 99]

# for file in selected_files:
#     levels = [0, 25, 50, 75, 99]
#     bz_photo_overall = load_save_3D_output('/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumes/' + file)[2]
#     for level in levels:
#         bz_photo = bz_photo_overall[:, :, level]

#         # Clear matplotlib
#         plt.clf()
#         cmap = cm.get_cmap('coolwarm', 51)  # Custom colormap
#         fig, ax = plt.subplots()
#         brsurf = ax.imshow(bz_photo, origin='lower', cmap=cmap, interpolation='spline16', vmin=-600, vmax=600)
#         ax.set_title(f'Bz Field Level {level}')
#         fig.colorbar(brsurf, ax=ax)

#         fileName = file[:-4] + f'_level{level}.png'
#         completeFileName = os.path.join(OUTPUT_DIR, fileName)

#         correctList = eval(f'files{level}')
#         correctList.append(completeFileName)

#         # Save the figure
#         plt.savefig(completeFileName)
#         plt.close()

# for level in levels:
#     correctList = eval(f'files{level}')
#     clip = ImageSequenceClip(correctList, fps=8)
#     clip.write_videofile(f'field_volume_movie_level{level}.mp4')

#     # Delete the files
#     for file in correctList:
#         os.remove(file)

# Now make all the level movies for the life of a single SHARP
# Recall that output files have format Bout_hmi.sharp_cea_720s.3999.20140418_090000_TAI
# One SHARP is the portion hmi.sharp_cea_720s.3999. Get all the out files for this SHARP
# and make a movie for each level. Pick 5 random SHARPs to do this for.

def contains_same_SHARP(sharp_outputs):
    sharp_numbers = [file.split('.')[2] for file in sharp_outputs]
    sharp_numbers_set = set(sharp_numbers)
    return len(sharp_numbers) != len(sharp_numbers_set)

random_SHARP_outputs = ['Bout_hmi.sharp_cea_720s.4698.20141023_090000_TAI.bin',
                        'Bout_hmi.sharp_cea_720s.7115.20170903_050000_TAI.bin',
                        'Bout_hmi.sharp_cea_720s.4536.20140906_170000_TAI.bin']

# while contains_same_SHARP(random_SHARP_outputs):
#     random_SHARP_outputs = random.sample(all_files, 5)

# for SHARP_output in random_SHARP_outputs:
#     SHARP = SHARP_output.split('.')[2]
#     SHARP_outputs = [file for file in all_files if SHARP in file]
#     SHARP_outputs = sorted(SHARP_outputs)
#     for level in levels:
#         files = []
#         for file in SHARP_outputs:
#             bz_photo_overall = load_save_3D_output('/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumes/' + file)[2]
#             bz_photo = bz_photo_overall[:, :, level]

#             # Clear matplotlib
#             plt.clf()
#             cmap = cm.get_cmap('coolwarm', 51)  # Custom colormap
#             fig, ax = plt.subplots()
#             brsurf = ax.imshow(bz_photo, origin='lower', cmap=cmap, interpolation='spline16', vmin=-600, vmax=600)
#             ax.set_title(f'Bz Field Level {level}')
#             fig.colorbar(brsurf, ax=ax)

#             fileName = file[:-4] + f'_level{level}.png'
#             completeFileName = os.path.join(OUTPUT_DIR, fileName)

#             files.append(completeFileName)

#             # Save the figure
#             plt.savefig(completeFileName)
#             plt.close()

#         clip = ImageSequenceClip(files, fps=8)
#         clip.write_videofile(f'field_volume_movie_SHARP{SHARP}_level{level}.mp4')

#         # Delete the files
#         for file in files:
#             os.remove(file)

# Now, for each of those SHARPs, make a movie that goes through all levels just at that specific time. Use the same 5 random SHARPs as above.
for SHARP_output in random_SHARP_outputs:
    # Movie should cycle through all levels from 0 to 99 at a 1 level increment
    SHARP = SHARP_output.split('.')[2]
    levels = list(range(100))
    files = []
    bz_photo_overall = load_save_3D_output('/mnt/horton_share/development/data/drms/MagPy_Shared_Data/DefinitiveFieldVolumes/' + SHARP_output)[2]
    for level in levels:
        bz_photo = bz_photo_overall[:, :, level]

        # Clear matplotlib
        plt.clf()
        cmap = cm.get_cmap('coolwarm', 51)
        fig, ax = plt.subplots()
        brsurf = ax.imshow(bz_photo, origin='lower', cmap=cmap, interpolation='spline16', vmin=-600, vmax=600)
        ax.set_title(f'Bz Field Level {level}')
        fig.colorbar(brsurf, ax=ax)

        fileName = SHARP_output[:-4] + f'_level{level}.png'
        completeFileName = os.path.join(OUTPUT_DIR, fileName)

        files.append(completeFileName)

        # Save the figure
        plt.savefig(completeFileName)
        plt.close()

    clip = ImageSequenceClip(files, fps=8)
    clip.write_videofile(f'field_volume_movie_SHARP{SHARP}_all_levels.mp4')

    # Delete the files
    for file in files:
        os.remove(file)

