# import nibabel as nib
#
# nii_file = nib.load('./files/brain_mask_anl.nii')
# data = nii_file.get_fdata()
# print(data.shape)  # This shows the dimensions of the imaging data.


import tkinter as tk
from tkinter import ttk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import nibabel as nib

# Ensure using the TkAgg matplotlib backend
matplotlib.use('TkAgg')


class NiiViewer:
    def __init__(self, master, nii_data):
        self.master = master
        self.nii_data = nii_data

        # Setup the matplotlib figure and axes
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Setup the canvas to embed the figure in the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Slider for navigating through slices
        self.slice_slider = ttk.Scale(master, from_=0, to=self.nii_data.shape[2] - 1, orient=tk.HORIZONTAL,
                                      command=self.update_plot)
        self.slice_slider.pack(side=tk.BOTTOM, fill=tk.X, expand=True)

        # Initialize plot
        self.update_plot(0)

    def update_plot(self, val):
        # Clear the current plot
        self.ax.clear()

        # Convert the slider value to an integer because it comes as a string
        slice_index = int(float(val))

        # Plot the selected slice
        self.ax.imshow(self.nii_data[:, :, slice_index], cmap='gray')

        # Redraw the canvas
        self.canvas.draw()


if __name__ == "__main__":
    # Load NIfTI file
    nii_file = nib.load('./files/brain_mask_anl.nii')
    data = nii_file.get_fdata()

    # Create the Tkinter window
    root = tk.Tk()
    root.wm_title("NIfTI Viewer")

    # Initialize and pack the NiiViewer
    viewer = NiiViewer(root, data)

    # Start the Tkinter event loop
    tk.mainloop()


