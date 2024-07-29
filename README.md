Codes for PINNs (based on Nvidia's Modulus package) for radiation therapy planning using mechanistic models for tumor growth and response to radiation

## Steps:

1. View brain cross sections from MRI scans using [view_mat_interactive_all.py](https://github.com/tnnandi/modulus_radiation_therapy/blob/main/view_mat_interactive_all.py) to ensure the geometry is well represented for meshing
2. Generate surface mesh (STL) from the cross sections using [convert_to_stl_surface_use_gmsh_volume.py](https://github.com/tnnandi/modulus_radiation_therapy/blob/main/convert_to_stl_surface_use_gmsh_volume.py) 
3. Generate volume mesh (STL) from the surface mesh using [gmsh](https://gmsh.info/). Generated meshes for a single patient are available at [Box](https://anl.box.com/s/tlyfb74wyuma0jm4zha8zfrwcspxshlb)
4. Execute [brain_param_D_time.py](https://github.com/tnnandi/modulus_radiation_therapy/blob/main/modulus-sym/examples/brain_RT/brain_param_D_time.py) using "mpirun -np <num_gpus> python brain_param_D_time.py"

Note: Step 4 needs to be carried out within a singularity shell for Nvidia's Modulus and the STL files need to be placed at the approriate locations 

## Implementation of the governing equations

The equation for the tumor growth and response to radiation is given by:

$$
\frac{\partial \hat{N}_T(x,t)}{\partial t} = \underbrace{\nabla \cdot (D_T \nabla \hat{N}_T(x,t))}_{\text{Diffusion}} + \underbrace{k_{p,T} \hat{N}_T(x,t) \left(1 - \frac{\hat{N}_T(x,t)}{\theta_T} \right)}_{\text{Proliferation}}
$$

## Assignment of physical parameters and the parameterized quantities



