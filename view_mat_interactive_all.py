from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import mpld3
from pdb import set_trace


# mat_data = loadmat('/mnt/c/Users/tnandi/Downloads/modulus/files/anl_domain.mat')
mat_data = loadmat('/mnt/c/Users/tnandi/Downloads/modulus/files/Updated_Geometry-06_28_2024/anl_domain.mat')

brain_mask = mat_data['brain_mask']
gray_matter = mat_data['gray_matter']
white_matter = mat_data['white_matter']

# set_trace()

# Calculate the index of the middle slice
middle_slice_index = brain_mask.shape[2] // 2

# Plot initial slices
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(left=0.25, bottom=0.25)

# Display the middle slices for each dataset
brain_mask_img = axs[0].imshow(brain_mask[:, :, middle_slice_index], cmap='gray')
axs[0].set_title('Brain Mask')
gray_matter_img = axs[1].imshow(gray_matter[:, :, middle_slice_index], cmap='gray')
axs[1].set_title('Gray Matter')
white_matter_img = axs[2].imshow(white_matter[:, :, middle_slice_index], cmap='gray')
axs[2].set_title('White Matter')

# Define initial slice index
slice_index = middle_slice_index

# Create slider
axcolor = 'lightgoldenrodyellow'
axslice = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
slice_slider = Slider(axslice, 'Slice', 0, brain_mask.shape[2] - 1, valinit=slice_index, valstep=1)

# Update function for the slider
def update(val):
    slice_index = int(slice_slider.val)
    brain_mask_img.set_data(brain_mask[:, :, slice_index])
    gray_matter_img.set_data(gray_matter[:, :, slice_index])
    white_matter_img.set_data(white_matter[:, :, slice_index])
    fig.canvas.draw_idle()

# Connect the update function to the slider
slice_slider.on_changed(update)


plt.show()
