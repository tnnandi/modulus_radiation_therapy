from scipy.io import loadmat
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from pdb import set_trace

mat_data = loadmat('/mnt/c/Users/tnandi/Downloads/modulus/files/anl_domain.mat')
print(mat_data.keys())  

gray_matter = mat_data['gray_matter']  

# Calculate the index of the middle slice
middle_slice_index = gray_matter.shape[2] // 2

# Display the middle slice
# plt.imshow(gray_matter[:, :, middle_slice_index], cmap='gray')
# plt.colorbar()
# plt.title(f'Gray Matter - Slice {middle_slice_index}')
# plt.show()


def plot_slice(slice_index):
    plt.imshow(gray_matter[:, :, slice_index], cmap='gray')
    plt.colorbar()
    plt.title(f'Gray Matter - Slice {slice_index}')
    plt.show()

slice_slider = widgets.IntSlider(min=0,
                                 max=gray_matter.shape[2] - 1,
                                 step=1,
                                 value=middle_slice_index,
                                 description='Slice')

widgets.interactive(plot_slice, slice_index=slice_slider)


# set_trace()

