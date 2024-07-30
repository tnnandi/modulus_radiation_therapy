import os
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
import nibabel as nib

mri_file = './files/brain_mask_anl.nii'
# mri_file = './files/gray_matter_anl.nii'
# mri_file = './files/white_mater_anl.nii'
img = nib.load(mri_file)

hdr = img.header

print("Shape: ", img.shape)
print("Voxel size: ", hdr.get_zooms())
print("Units of measurement: ", hdr.get_xyzt_units())

img_data = img.get_fdata()

print("Type of img_data: ", type(img_data))
print("Shape of img_data: ", img_data.shape) # (297, 248, 31) for brain_mask_anl

set_trace()
mid_vox = img_data[148:150, 123:125, 14:16]
print(mid_vox)

mid_slice_x = img_data[148, :, :]

# Note that the transpose the slice (using the .T attribute).
# This is because imshow plots the first dimension on the y-axis and the
# second on the x-axis, but we'd like to plot the first on the x-axis and the
# second on the y-axis. Also, the origin to "lower", as the data was saved in
# "cartesian" coordinates.
# plt.imshow(mid_slice_x.T, cmap='gray', origin='lower')
# plt.xlabel('First axis')
# plt.ylabel('Second axis')
# plt.colorbar(label='Signal intensity')
# plt.show()

fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

ax[0].imshow(img_data[148, :, :].T, origin='lower', cmap='gray')
ax[0].set_xlabel('Second dim voxel coords.', fontsize=12)
ax[0].set_ylabel('Third dim voxel coords', fontsize=12)
ax[0].set_title('First dimension, slice nr. 70', fontsize=15)

ax[1].imshow(img_data[:, 115, :].T, origin='lower', cmap='gray')
ax[1].set_xlabel('First dim voxel coords.', fontsize=12)
ax[1].set_ylabel('Third dim voxel coords', fontsize=12)
ax[1].set_title('Second dimension, slice nr. 100', fontsize=15)

ax[2].imshow(img_data[:, :, 15].T, origin='lower', cmap='gray')
ax[2].set_xlabel('First dim voxel coords.', fontsize=12)
ax[2].set_ylabel('Second dim voxel coords', fontsize=12)
ax[2].set_title('Third dimension, slice nr. 100', fontsize=15)

fig.tight_layout()
plt.show()