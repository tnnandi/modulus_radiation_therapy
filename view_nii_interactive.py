import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import nibabel as nib

# Load the .nii file
nii_file = nib.load('./files/brain_mask_anl.nii')
data = nii_file.get_fdata()

# Setup the figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

# Initial slice index
slice_index = data.shape[2] // 2

# Display the initial slice
l = plt.imshow(data[:, :, slice_index], cmap='gray')

# Create a slider for navigating slices
axcolor = 'lightgoldenrodyellow'
axslice = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
slice_slider = Slider(axslice, 'Slice', 0, data.shape[2] - 1, valinit=slice_index, valstep=1)

# Update function for the slider
def update(val):
    slice_index = int(slice_slider.val)
    l.set_data(data[:, :, slice_index])
    fig.canvas.draw_idle()

# Connect the update function to the slider
slice_slider.on_changed(update)

plt.show()

