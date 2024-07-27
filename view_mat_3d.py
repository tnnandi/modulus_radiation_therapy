from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pdb import set_trace
# use the conda env "modulus_env_py3p10"

mat_data = loadmat('/mnt/c/Users/tnandi/Downloads/modulus/files/anl_domain.mat')
# mat_data.keys()
# dict_keys(['__header__', '__version__', '__globals__', 'brain_mask', 'gray_matter', 'white_matter'])
data = mat_data['brain_mask']
# data.shape
# (297, 248, 31)
set_trace()
# plot every n'th point
n = 5
data = data[::n, ::n, ::n]

print("Shape of the 3D brain mask:", data.shape)

# to visualize, we'll create a meshgrid and plot points where the mask is 1
# x, y, z = np.indices(data.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#ax.voxels(x, y, z, data == 1, facecolors='blue', edgecolor='k')
voxel_data = (data == 1)
ax.voxels(voxel_data, facecolors='blue', edgecolor='k')


plt.show()

