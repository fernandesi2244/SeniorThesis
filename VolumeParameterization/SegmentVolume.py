import Utils
import sunpy.map
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np

bx_3D, by_3D, bz_3D = Utils.load_volume_components("InputData/Bout_hmi.sharp_cea_720s.1449.20120306_170000_TAI.bin")
bitmap = sunpy.map.Map('InputData/hmi.sharp_cea_720s.1449.20120306_170000_TAI.bitmap.fits').data
br = sunpy.map.Map('InputData/hmi.sharp_cea_720s.1449.20120306_170000_TAI.Br.fits').data

print('bx_3D shape:', bx_3D.shape)


bz_photospheric_level = bz_3D[:, :, 0]
bz_halfway_level = bz_3D[:, :, 50]
bz_top_level = bz_3D[:, :, 99]

cmap = cm.get_cmap('coolwarm', 51)  # Custom colormap
fig, ax = plt.subplots()
brsurf = ax.imshow(bz_photospheric_level,origin='lower',cmap = cmap,interpolation='spline16', vmin=-600,vmax=600)
ax.set_title('Bz Field')
fig.colorbar(brsurf, ax=ax)
plt.show()

# Print distinct values in bitmap
print(np.unique(bitmap))

# Resize bitmap and Br to 200x400
bitmap = resize(bitmap, (200, 400), anti_aliasing=True, preserve_range=True)
br = resize(br, (200, 400), anti_aliasing=True, preserve_range=True)

print(np.unique(bitmap)[-200:-100])

# Visualize bitmap
fig, ax = plt.subplots()
bitmap_axImg = ax.imshow(bitmap, cmap='gray', origin='lower')
ax.set_title('Bitmap')
fig.colorbar(bitmap_axImg, ax=ax)
plt.show()

# Make a mask out of this bitmap and apply it to bz_photospheric_level and visualize the result
mask = bitmap > 30 # TODO: should this be a bit lower because of the interpolation?
masked_bz = bz_photospheric_level * mask

fig, ax = plt.subplots()
masked_bz_surf = ax.imshow(masked_bz, origin='lower', cmap=cmap, interpolation='spline16', vmin=-600, vmax=600)
ax.set_title('Masked Bz Field')
fig.colorbar(masked_bz_surf, ax=ax)
plt.show()

exit()

# Apply mask to middle level of bz and visualize the result
masked_bz_halfway = bz_halfway_level * mask

fig, ax = plt.subplots()
masked_bz_halfway_surf = ax.imshow(masked_bz_halfway, origin='lower', cmap=cmap, interpolation='spline16', vmin=-600, vmax=600)
ax.set_title('Masked Bz Field at Middle Level')
fig.colorbar(masked_bz_halfway_surf, ax=ax)
plt.show()

# Apply mask to top level of bz and visualize the result
masked_bz_top = bz_top_level * mask

fig, ax = plt.subplots()
masked_bz_top_surf = ax.imshow(masked_bz_top, origin='lower', cmap=cmap, interpolation='spline16', vmin=-600, vmax=600)
ax.set_title('Masked Bz Field at Top Level')
fig.colorbar(masked_bz_top_surf, ax=ax)
plt.show()

# NOW, try to broadcast the mask to the 3D volume, and let's visualize
# the bottom, middle, and top levels of the resulting volume to confirm it's the same.
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

# Visualize the bottom, middle, and top levels of the masked 3D volume
fig, ax = plt.subplots()
bx_3D_blob_surf = ax.imshow(bx_3D_blob[:, :, 0], origin='lower', cmap=cmap, interpolation='spline16', vmin=-600, vmax=600)
ax.set_title('NEW - Masked Bz Field at Bottom Level')
fig.colorbar(bx_3D_blob_surf, ax=ax)
plt.show()

fig, ax = plt.subplots()
bx_3D_blob_surf = ax.imshow(bx_3D_blob[:, :, 50], origin='lower', cmap=cmap, interpolation='spline16', vmin=-600, vmax=600)
ax.set_title('NEW - Masked Bz Field at Middle Level')
fig.colorbar(bx_3D_blob_surf, ax=ax)
plt.show()

fig, ax = plt.subplots()
bx_3D_blob_surf = ax.imshow(bx_3D_blob[:, :, 99], origin='lower', cmap=cmap, interpolation='spline16', vmin=-600, vmax=600)
ax.set_title('NEW - Masked Bz Field at Top Level')
fig.colorbar(bx_3D_blob_surf, ax=ax)
plt.show()

# # Apply the mask to the Br component
# masked_br = br * mask

# fig, ax = plt.subplots()
# masked_br_surf = ax.imshow(masked_br, origin='lower', cmap=cmap, interpolation='spline16', vmin=-600, vmax=600)
# ax.set_title('Masked Br Field After Resizing')
# fig.colorbar(masked_br_surf, ax=ax)
# plt.show()
