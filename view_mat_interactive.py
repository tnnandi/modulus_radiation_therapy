from scipy.io import loadmat
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from pdb import set_trace
from matplotlib.widgets import Slider

mat_data = loadmat('/mnt/c/Users/tnandi/Downloads/modulus/files/anl_domain.mat')
print(mat_data.keys())  # ['brain_mask', 'gray_matter', 'white_matter']

# data = mat_data['brain_mask']
# data = mat_data['gray_matter']
data = mat_data['white_matter']

# Calculate the index of the middle slice
middle_slice_index = data.shape[2] // 2

# Display the middle slice
# plt.imshow(data[:, :, middle_slice_index], cmap='gray')
# plt.colorbar()
# plt.title(f'Gray Matter - Slice {middle_slice_index}')
# plt.show()

# Plot initial slice
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
slice_index = data.shape[2] // 2
l = plt.imshow(data[:, :, slice_index], cmap='gray')

# Create slider
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
