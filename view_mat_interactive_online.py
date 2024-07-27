import numpy as np
import plotly.graph_objs as go
import plotly.subplots as sp
from scipy.io import loadmat
import chart_studio.plotly as py
import chart_studio
import gzip
import pickle

# Replace with your Plotly Chart Studio credentials
chart_studio.tools.set_credentials_file(username='TarakNathNandi', api_key='g9KIXJ8bDwOcHKFbCrcD')

# Load the .mat file
# mat_data = loadmat('/mnt/c/Users/tnandi/Downloads/modulus/files/anl_domain.mat')  # Ensure the .mat file is in the same directory or provide the correct path

# Load the .mat file and compress data
def load_and_compress(file_path):
    with open(file_path, 'rb') as f:
        mat_data = loadmat(f)
    return mat_data

def compress_data(data, factor=2):
    compressed_data = data[::factor, ::factor, :]
    return compressed_data

# Load and compress data
mat_data = load_and_compress('/mnt/c/Users/tnandi/Downloads/modulus/files/anl_domain.mat')
brain_mask = compress_data(mat_data['brain_mask'])
gray_matter = compress_data(mat_data['gray_matter'])
white_matter = compress_data(mat_data['white_matter'])

# Calculate the index of the middle slice
middle_slice_index = brain_mask.shape[2] // 2

# Define a custom discrete colorscale
discrete_colorscale = [[0, 'black'], [1, 'white']]

# Create a subplot figure
fig = sp.make_subplots(rows=1, cols=3, subplot_titles=('Brain Mask', 'Gray Matter', 'White Matter'))

# Add initial traces for the middle slice with coloraxis
brain_mask_trace = go.Heatmap(z=brain_mask[:, :, middle_slice_index], colorscale=discrete_colorscale, zmin=0, zmax=1, showscale=False, coloraxis="coloraxis")
gray_matter_trace = go.Heatmap(z=gray_matter[:, :, middle_slice_index], colorscale=discrete_colorscale, zmin=0, zmax=1, showscale=False, coloraxis="coloraxis")
white_matter_trace = go.Heatmap(z=white_matter[:, :, middle_slice_index], colorscale=discrete_colorscale, zmin=0, zmax=1, showscale=False, coloraxis="coloraxis")

fig.add_trace(brain_mask_trace, row=1, col=1)
fig.add_trace(gray_matter_trace, row=1, col=2)
fig.add_trace(white_matter_trace, row=1, col=3)

# Update layout for aspect ratio and add coloraxis
fig.update_layout(
    height=600,
    width=1200,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(scaleanchor='y', scaleratio=1),
    yaxis=dict(scaleanchor='x', scaleratio=1),
    xaxis2=dict(scaleanchor='y2', scaleratio=1),
    yaxis2=dict(scaleanchor='x2', scaleratio=1),
    xaxis3=dict(scaleanchor='y3', scaleratio=1),
    yaxis3=dict(scaleanchor='x3', scaleratio=1),
    coloraxis=dict(
        colorscale=discrete_colorscale,
        colorbar=dict(
            tickvals=[0, 1],
            ticktext=['0', '1']
        )
    )
)

# Create slider steps
steps = []
for i in range(brain_mask.shape[2]):
    step = dict(
        method='update',
        args=[{
            'z': [brain_mask[:, :, i], gray_matter[:, :, i], white_matter[:, :, i]]
        }],
        label=str(i)
    )
    steps.append(step)

# Add slider to the layout
fig.update_layout(
    sliders=[{
        'active': middle_slice_index,
        'pad': {'t': 50},
        'steps': steps
    }]
)

# Upload the figure to Plotly Chart Studio
plot_url = py.plot(fig, filename='Brain MRI Slices Viewer', auto_open=False)

print(f"Plot URL: {plot_url}")