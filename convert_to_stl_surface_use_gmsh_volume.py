import scipy.io
import numpy as np
from skimage import measure
import trimesh
import pygalmesh  # conda install conda-forge::pygalmesh
import meshio  # pip install meshio
import tetgen
from pdb import set_trace
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the MAT file
# mat_file_path = '/mnt/c/Users/tnandi/Downloads/modulus/files/Updated_Geometry-06_28_2024/anl_domain.mat'
mat_file_path = './files/Updated_Geometry-06_28_2024/anl_domain.mat'
mat_data = scipy.io.loadmat(mat_file_path)

# Extract the masks
brain = mat_data['brain_mask']

# Define the voxel size (in mm)
voxel_size_in_plane = 0.5  # mm
voxel_size_slice_direction = 0.5 # 2  # mm

# Subsample the brain mask (e.g., by a factor of 2)
subsample_factor = 4
# brain_subsampled = brain[::subsample_factor, ::subsample_factor, ::subsample_factor]
brain_subsampled = brain[::subsample_factor, ::subsample_factor, ::]

# Generate surface mesh using marching_cubes
verts, faces, normals, values = measure.marching_cubes(brain_subsampled, level=0.5)

# Scale vertices by voxel size
verts[:, 0] *= voxel_size_in_plane * subsample_factor
verts[:, 1] *= voxel_size_in_plane * subsample_factor
verts[:, 2] *= voxel_size_slice_direction

# Create surface mesh
surface_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        surface_mesh.vectors[i][j] = verts[f[j]]

print("Number of points in the surface mesh: ", len(surface_mesh.vectors))
print("Surface mesh bounds: ", surface_mesh.min_, surface_mesh.max_)
surface_mesh.save('brain_surface_subsample_4X.stl')

# use gmsh to create the volume mesh from the above surface mesh

set_trace()
# # Generate volume mesh from surface mesh using tetgen
# tetgen_mesh = tetgen.TetGen(surface_mesh.vectors.reshape(-1, 3), faces - 1)
# tetgen_mesh.tetrahedralize() # try this on polaris
# volume_mesh = tetgen_mesh.grid
#
# print("Number of points in the volume mesh: ", len(volume_mesh.vectors))
# print("Volume mesh bounds: ", volume_mesh.min_, volume_mesh.max_)
# set_trace()