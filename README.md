Codes for PINNs (based on Nvidia's Modulus package) for radiation therapy planning using mechanistic models for tumor growth and response to radiation

## Objective

Conventional RT for cancers usually utilizes a single set of imaging acquired prior to the start of treatment and selects a treatment plan that has been shown to work well 
on an ``average patient'' However, cancer is a complex, evolving system that exhibits
significant inter-patient variations that depend on various factors, 
including the underlying genomic instability and tumor microenvironment. 
Physics-informed neural networks (PINNs) are being developed in this repo by integrating 
mechanistic modeling with MRI data to model the spatio-temporal response of 
tumors to RT, to create therapy plans based on individual tumor biology. 

Such a PINN model, trained over a broad range of relevant parameters 
(e.g., diffusion coefficients, tumor proliferation rates, RT dose) and 
regularized/guided using available MRI data, 
can be used to carry out near-instantaneous predictions for the tumor trajectory 
based on parameters corresponding to any new patient, 
thus streamlining RT planning and precluding the need for a huge number of 
parameter-specific computer simulations.

## Steps:

1. View brain cross sections from MRI scans using [view_mat_interactive_all.py](https://github.com/tnnandi/modulus_radiation_therapy/blob/main/view_mat_interactive_all.py) to ensure the geometry is well represented for meshing
2. Generate surface mesh (STL) from the cross sections using [convert_to_stl_surface_use_gmsh_volume.py](https://github.com/tnnandi/modulus_radiation_therapy/blob/main/convert_to_stl_surface_use_gmsh_volume.py) 
3. Generate volume mesh (STL) from the surface mesh using [gmsh](https://gmsh.info/). Generated meshes for a single patient are available at [Box](https://anl.box.com/s/tlyfb74wyuma0jm4zha8zfrwcspxshlb)
4. Execute [brain_param_D_time.py](https://github.com/tnnandi/modulus_radiation_therapy/blob/main/modulus-sym/examples/brain_RT/brain_param_D_time.py) using "mpirun -np <num_gpus> python brain_param_D_time.py"

Note: Step 4 needs to be carried out within a singularity shell for Nvidia's Modulus and the STL files need to be placed at the approriate locations 

## Implementation of the governing equations

The single-species model for tumor growth and response to radiation is given by:
$\frac{\partial \hat{N}_T(x,t)}{\partial t} = \nabla \cdot (D_T \nabla \hat{N}_T(x,t)) + \kappa\hat{N}_T(x,t)(1 - \frac{\hat{N}_T(x,t)}{\theta_T}) + R$

where the terms on the LHS represented the temporal evolution of the normalized tumor density $\hat{N}$. The first and the second terms on the RHS represent diffusion and proliferation, respectively. 
$R$ represents the response to radiation therapy (and chemotherapy, if present) term. 

${N}_T(x,t)$: Normalized tumor density \
$D_T$: Tumor cell diffusion coefficient \
$\kappa$: Tumor cell proliferation rate \
$\theta_T$: Carrying capacity of the tumor \

The normalized tumor density following the treatment event is given by:

$$
\hat{N}_{i,\text{post}}(x,t) = \hat{N}_{i,\text{pre}}(x,t) \text{SF}_{\text{RT+CT}}(x,t), 
$$

where the normalized tumor density following the treatment event, $\hat{N}_{i,\text{post}}$, was assigned to the product of the pre-treatment normalized tumor density, $\hat{N}_{i,\text{pre}}$, and the surviving fractions of cells due to a single dose of combined radiotherapy and chemotherapy, $\text{SF}_{\text{RT+CT}}$. 

$$
\text{SF}_{\text{RT+CT}}(x,t) = e^{-\alpha \cdot \text{Dose}(x,t) \left(1 + \frac{\text{Dose}(x,t)}{\alpha/\beta} \right)}, 
$$

where $\alpha$ is a treatment sensitivity term, $\text{Dose}(x,t)$ is the dose of RT+CT given in a single fraction, and $\alpha/\beta$ is the ratio of the linear and quadratic sensitivity terms set to a fixed value of 5.6 Gy.

## Assignment of physical parameters and the parameterized quantities



