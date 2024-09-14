from scipy.io import loadmat
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from pdb import set_trace
from matplotlib.widgets import Slider
import h5py
import vtk
from vtk.util import numpy_support
import numpy as np

# conda activate modulus_env_py3p10

# mat_data = loadmat('/mnt/c/Users/tnandi/Downloads/modulus/files//ANL_MDACC_UTAustin_PINNs-selected/Example_Case_1/tumor_1.mat') # doesn't work for the matlab v7.3 files
with h5py.File(
        '/mnt/c/Users/tnandi/Downloads/modulus/files//ANL_MDACC_UTAustin_PINNs-selected/Example_Case_2/tumor_2.mat',
        'r') as file:
    print("Keys: %s" % file.keys())  # ['N_meas', 'day', 'params', 'treat_time_rt']
    # dataset = file['your_dataset_name'][:]
    N_meas = file['N_meas'][:]
    day = file['day'][
          :]  # measurement time points, i.e., time points for the data collection (not treatment times). In this case, the "measurement" corresponds to numerical data

    proliferation_rate = file['params'][0, :]
    diffusion_coeff = file['params'][1, :]
    RT_alpha = file['params'][2, :]

    treatment_times = file['treat_time_rt'][0, :]
    dose = file['treat_time_rt'][1, :]

    # set_trace()

# dataset "day": shape (9, 1), type "<f8">
# dataset "N_meas": shape (9, 99, 281, 374), type "<f8">   # [day, _, _, _]
# dataset "params": shape (3, 1), type "<f8"
# dataset "treat_time_rt": shape (2, 30), type "<f8">

# Description of parameters:
# params(1) =  Proliferation (1/day)
# params(2) =  Diffusion (mm^2/day)
# params(3) = RT term (Gy^-1).  This is the “alpha” parameter from the Linear quadratic model, I assume an alpha to beta ratio of 10.
#
# treat_time_rt(:,1) = (an array listing the days of each visit of RT. Typically Monday-Friday for 6 weeks)
# treat_time_rt(:,2) =  (amount of RT given at each visit in Gys)

# For Example_Case_1,2,3:

# treatment_times
# array([32., 33., 34., 35., 36., 39., 40., 41., 42., 43., 46., 47., 48.,
#        49., 50., 53., 54., 55., 56., 57., 60., 61., 62., 63., 64., 67.,
#        68., 69., 70., 71.])
# dose [Gy]
# array([2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
#        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])

# day
# array([[  0.],
#        [ 38.],
#        [ 45.],
#        [ 52.],
#        [ 59.],
#        [ 66.],
#        [101.],
#        [131.],
#        [161.]])


# params = [proliferation_rate, diffusion_coeff, alpha]
# For Example_Case_1:
# params = [0.05, 0.125, 0.035]

# For Example_Case_2:
# params = [0.02, 0.125, 0.035]

# For Example_Case_3:
# params = [0.3, 0.125, 0.035]

# Summary: Only the proliferation rate varies among the above 3 cases (we can parameterize proliferation rate)

# sample plot
# dataset "N_meas": shape (9, 99, 281, 374), type "<f8">  # (281 x 374) is the in-plane resolution


# fig, ax = plt.subplots()
# # assuming the second dim corresponds to the slice index
# contour_plot = ax.contourf(N_meas[8, 50, :, :], levels=50, cmap='viridis')
# plt.colorbar(contour_plot, ax=ax, label='N')
# # ax.set_title(f'Tumor density distribution at day {day[0, 0]:.1f}, slice {slice_index}')
# # plt.show()
# # set_trace()

day_array = np.array([0, 38, 45, 52, 59, 66, 101, 131, 161])
day_indices = range(len(day_array))
slice_indices = range(99)

slice_index_fixed = 50
day_index_fixed = 2
vmin = 0
vmax = 1.0

# fig, axs = plt.subplots(3, 3, figsize=(15, 15))
# proxy_artist = plt.cm.ScalarMappable(cmap='viridis')
# proxy_artist.set_array([])
# proxy_artist.set_clim(vmin, vmax)

# for i, day_index in enumerate(day_indices):
#     day = day_array[day_index]
#     ax = axs[i // 3, i % 3]  # position in the 3x3 grid
#     contour_plot = ax.contourf(N_meas[day_index, slice_index_fixed, :, :], levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
#     ax.set_title(f'Tumor density at day {day}, slice {slice_index_fixed}')
#
# cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
# fig.colorbar(proxy_artist, cax=cbar_ax, label='N', orientation='vertical')
# plt.tight_layout(rect=[0, 0, 0.95, 1])
# plt.show()

# for i, slice_index in enumerate(slice_indices):
#     ax = axs[i // 3, i % 3]  # position in the 3x3 grid
#     day = day_array[day_index_fixed]
#     contour_plot = ax.contourf(N_meas[day_index_fixed, slice_index, :, :], levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
#     ax.set_title(f'Tumor density at day {day}, slice {slice_index}')
#
# cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
# fig.colorbar(proxy_artist, cax=cbar_ax, label='N', orientation='vertical')
# plt.tight_layout(rect=[0, 0, 0.95, 1])
# plt.show()


# set_trace()
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

slide_slices = True
slide_days = False

if slide_slices:
    # initial slice index (middle slice)
    slice_index = N_meas.shape[1] // 2
    contour_plot = ax.contourf(N_meas[day_index_fixed, slice_index, :, :], levels=50, cmap='viridis')
    plt.colorbar(contour_plot, ax=ax, label='N')
    ax.set_title(f'Tumor density distribution at day {day_array[day_index_fixed]}, slice {slice_index}')
    # set_trace()

    axcolor = 'lightgoldenrodyellow'
    axslice = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    slice_slider = Slider(axslice, 'Slice', 0, N_meas.shape[1] - 1, valinit=slice_index, valstep=1)


    def update(val):
        slice_index = int(slice_slider.val)
        ax.clear()
        contour_plot = ax.contourf(N_meas[day_index_fixed, slice_index, :, :], levels=50, cmap='viridis')
        global cbar
        cbar.remove()

        cbar = plt.colorbar(contour_plot, ax=ax, label='N')

        ax.set_title(f'Tumor density distribution at day {day_array[day_index_fixed]}, slice {slice_index}')
        fig.canvas.draw_idle()


    slice_slider.on_changed(update)
    plt.show()

if slide_days:
    # initial slice index (middle slice)
    slice_index = N_meas.shape[1] // 2
    day_index = 0  # start from day 0
    # contour_plot = ax.contourf(N_meas[day_index, :, :, slice_index], levels=50, cmap='viridis')
    contour_plot = ax.contourf(N_meas[day_index, slice_index, :, :], levels=50, cmap='viridis')
    plt.colorbar(contour_plot, ax=ax, label='NN')
    ax.set_title(f'Tumor density distribution at day {day[0, 0]:.1f}, slice {slice_index}')
    # set_trace()

    axcolor = 'lightgoldenrodyellow'
    # axslice = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    day_slider = Slider(axslice, 'Day', 0, N_meas.shape[0] - 1, valinit=day_index, valstep=1)


    def update(val):
        day_index = int(day_slider.val)
        ax.clear()
        contour_plot = ax.contourf(N_meas[day_index, slice_index, :, :], levels=50, cmap='viridis')
        global cbar
        cbar.remove()

        cbar = plt.colorbar(contour_plot, ax=ax, label='N')

        ax.set_title(f'Tumor density distribution at day {day[day_index, 0]:.1f}, slice {slice_index}')
        fig.canvas.draw_idle()


    day_slider.on_changed(update)
    plt.show()

# set_trace()
# # extract the 3D volume data for the first time point
# N_meas_3d = N_meas[0, :, :, :]
#
# dx = 0.5
# dy = 0.5
# dz = 0.5
#
# # Convert the numpy array to VTK array
# vtk_data_array = numpy_support.numpy_to_vtk(num_array=N_meas_3d.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
#
# # Create the VTK image data object
# image_data = vtk.vtkImageData()
# image_data.SetDimensions(N_meas_3d.shape)
# image_data.SetSpacing(dx, dy, dz)
#
# # Add the VTK array as a scalar field to the image data
# image_data.GetPointData().SetScalars(vtk_data_array)
# set_trace()
# # Write the data to a .vti file
# vti_writer = vtk.vtkXMLImageDataWriter()
# vti_writer.SetFileName('./N_meas_3d_volume.vti')
# vti_writer.SetInputData(image_data)
# vti_writer.Write()
