from scipy.io import loadmat
import numpy as np
from skimage import measure
import trimesh

# Do a "conda activate modulus_env_py3p10" on Dell laptop

# Load the data
mat_data = loadmat('/mnt/c/Users/tnandi/Downloads/modulus/files/anl_domain.mat')

req_string = 'brain_mask'

# Extract the datasets
data = mat_data[req_string]

# Choose one of the datasets to create the 3D volume (for example, gray_matter)
volume_data = data

# Define the spacing parameters (in millimeters)
slice_spacing = 2.0  # Distance between slices
pixel_spacing = 0.5  # In-plane pixel spacing

# Rescale the volume to account for spacing
scale = np.array([pixel_spacing, pixel_spacing, slice_spacing])
scaled_volume_data = np.transpose(volume_data, (1, 0, 2))

# Generate the surface mesh using the marching cubes algorithm
verts, faces, normals, values = measure.marching_cubes(scaled_volume_data, level=0.5)

# Apply the scaling to the vertices
verts *= scale

# Create a mesh object
mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

# Save the mesh as an STL file
mesh.export(req_string + '_mesh.stl')

print("STL file created successfully!")
