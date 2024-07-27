from scipy.io import loadmat
import numpy as np
from skimage import measure, morphology
import trimesh

# Load the data
mat_data = loadmat('/mnt/c/Users/tnandi/Downloads/modulus/files/anl_domain.mat')

# Extract the datasets
brain_mask = mat_data['brain_mask']
gray_matter = mat_data['gray_matter']
white_matter = mat_data['white_matter']

# Choose one of the datasets to create the 3D volume (for example, gray_matter)
volume_data = gray_matter

# Define the spacing parameters (in millimeters)
slice_spacing = 5.0  # Distance between slices
pixel_spacing = 1.0  # In-plane pixel spacing

# Rescale the volume to account for spacing
scale = np.array([pixel_spacing, pixel_spacing, slice_spacing])
scaled_volume_data = np.transpose(volume_data, (1, 0, 2))

# Preprocess the data to fill holes and remove noise
# Fill holes
filled_volume = morphology.binary_closing(scaled_volume_data, morphology.ball(1))

# Smooth the volume
smoothed_volume = morphology.binary_opening(filled_volume, morphology.ball(1))

# Generate the surface mesh using the marching cubes algorithm
verts, faces, normals, values = measure.marching_cubes(smoothed_volume, level=0.5)

# Apply the scaling to the vertices
verts *= scale

# Create a mesh object
mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

# Post-process the mesh to ensure it is watertight
mesh.remove_degenerate_faces()
mesh.fill_holes()
mesh.remove_duplicate_faces()
mesh.remove_infinite_values()
mesh.process(validate=True)

# Save the mesh as an STL file
mesh.export('gray_matter_mesh_watertight.stl')

print("Watertight STL file created successfully!")
