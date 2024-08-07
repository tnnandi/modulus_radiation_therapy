import os
import warnings
import wandb
from pdb import set_trace
from typing import Optional, Dict, Tuple, Union, List
from modulus.sym.key import Key
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from sympy import Symbol, sqrt, Max, Eq
# from modulus.launch.logging import LaunchLogger, PythonLogger, initialize_wandb
from modulus.sym.geometry import Bounds, Parameterization, Parameter
import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.key import Key
from modulus.sym.node import Node
# from modulus.sym.eq.pdes.navier_stokes import NavierStokes
# from modulus.sym.eq.pdes.advection_diffusion import AdvectionDiffusion
from modulus.sym.eq.pdes.diffusion import Diffusion
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.eq.pdes.basic import GradNormal
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.geometry.tessellation import Tessellation

from modulus.models.layers import FCLayer, Conv1dFCLayer
from modulus.sym.models.activation import Activation, get_activation_fn
from modulus.sym.models.arch import Arch

# cd /mnt/Modulus_24p04/modulus-sym/examples/brain_RT
# use "mpirun -np 4 python brain_param_D_time.py"

import sys
# print(sys.path)
# sys.path.append('/mnt/Modulus_24p04/modulus-sym/modulus/sym/eq/pdes')
# sys.path.append('/mnt/Modulus_24p04/modulus-sym/writable_dir/lib/python3.10/site-packages')
# from modulus.sym.eq.pdes.diffusion_proliferation_source import DiffusionProliferationSource

# for now, importing it from a file in the same dir due to pip install issues for the eq/pdes dir
from diffusion_proliferation_treatment import DiffusionProliferationTreatment

# wandb.init(project="modulus_brain_RT", entity='tnnandi')

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # read stl files to make geometry
    point_path = to_absolute_path("./stl_files")

    farfield_mesh = Tessellation.from_stl(
        point_path + "/brain_surface_subsample_4X.stl", airtight=False
    )

    interior_mesh = Tessellation.from_stl(
        point_path + "/brain_volume_subsample_4X.stl", airtight=True
    )

    # Mesh bounds
    print("Bounds before scaling")
    print(farfield_mesh.bounds)
    print(interior_mesh.bounds)
    # bound_ranges: {x: (51.0, 239.0), y: (57.0, 197.0), z: (9.25, 109.25)} param_ranges: {}
    # bound_ranges: {x: (51.0, 239.0), y: (57.0, 197.0), z: (9.25, 109.25)} param_ranges: {}

    scale = 1  # keep dims in mm # 1e-3 # dimensions are in mm
    farfield_mesh.scale(scale)
    interior_mesh.scale(scale)

    # diffusion coefficient
    D_value = 0.5

    # proliferation rate and carrying capacity
    k_p_value = 0.5
    theta_value = 0.1

    # dose related parameters (with parameterized Dose)
    Dose_symbol = Symbol("Dose")
    Dose_range = (0.0, 10.0)
    alpha_value = 0.5
    alpha_by_beta_value = 10

    print("Bounds after scaling")
    print(farfield_mesh.bounds)
    print(interior_mesh.bounds)

    # set_trace()

    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")

    # for now, using the continuous time approach (later, we can move to moving window approach for time evolution over larger durations)
    # parameterized time
    time_symbol = Symbol("t")
    time_range = (0.0, 10.0)  # in days

    param_ranges = Parameterization({Dose_symbol: Dose_range, time_symbol: time_range})

    # make brain domain
    domain = Domain()

    # Need to reformulate N so that it is the normalized tumor density and is between 0 and 1
    # or can add a Sigmoid activation after the final layer of the NN
    tumor_diffusion_proliferation_source_eq = DiffusionProliferationTreatment(T="N",
                                                                           D=D_value,
                                                                           k_p=k_p_value,
                                                                           theta=theta_value,
                                                                           alpha=alpha_value,
                                                                           alpha_by_beta=alpha_by_beta_value,
                                                                           dim=3,
                                                                           time=True)  # the equation will be solved for "N": the normalized tumor density

    # override defaults
    cfg.arch.fully_connected.layer_size = 128
    cfg.arch.fully_connected.nr_layers = 4

    # input_keys = [Key("x"), Key("y"), Key("z"), Key("t"), Key("Dose")]
    # output_keys = [Key("N")]

    tumor_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t"), Key("Dose")],  # add "D" and "t" as additional input (parameterized)
        output_keys=[Key("N")],
        cfg=cfg.arch.fully_connected,
        )
    # set_trace()
    for i in range(10):
        print("########################################################")
    # Print the total number of trainable parameters
    total_params = count_trainable_params(tumor_net)
    print(f"Total number of trainable parameters: {total_params}")

    grad_normal_N = GradNormal("N", dim=3)

    nodes = (
            tumor_diffusion_proliferation_source_eq.make_nodes()
            + [tumor_net.make_node(name="flow_network")]
            + grad_normal_N.make_nodes()
    )

    # add constraints to solver

    # farfield BC (surface boundary of the brain)
    farfield = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=farfield_mesh,
        # outvar={"N": 0},
        outvar={"normal_gradient_N": 0}, # set the surface normal gradient of N to zero
        batch_size=cfg.batch_size.farfield,
        lambda_weighting={"normal_gradient_N": 1.0},
        parameterization=param_ranges
    )
    domain.add_constraint(farfield, "farfield")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"diffusion_proliferation_source_N": 0},  # can add a source term
        batch_size=cfg.batch_size.interior,
        lambda_weighting={"diffusion_proliferation_source_N": 100.0},
        parameterization=param_ranges,
    )
    domain.add_constraint(interior, "interior")

    # Initial condition: normalized Gaussian distribution
    def initial_tumor_density(x, y, z):
        center_x, center_y, center_z = 130, 120, 60  # mm
        sigma = 5  # standard deviation in mm
        return np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2) / (2 * sigma ** 2))

    # Initial condition constraint

    initial_condition = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"N": initial_tumor_density},
        # representing the initial tumor using a gaussian distribution centered around a specific location
        # outvar={"N": 0},
        batch_size=cfg.batch_size.initial_condition,  # 10
        lambda_weighting={"N": 50.0},
        parameterization={Dose_symbol: Dose_range, time_symbol: 0}  # fix t=0
    )
    domain.add_constraint(initial_condition, "initial_condition")

    # Interior inferencer
    inference_times = [0.0, 2.0, 5.0, 10.0]
    inference_Doses = [0, 5, 10, 20]

    interior_points = interior_mesh.sample_interior(cfg.batch_size.inference)
    time_index = 0
    for time_value in inference_times:
        Dose_index = 0
        for Dose_value in inference_Doses:
            # print(Dose_value, type(Dose_value))
            invar = {
                "x": interior_points["x"],
                "y": interior_points["y"],
                "z": interior_points["z"],
                "t": np.full_like(interior_points["x"], time_value),
                "Dose": np.full_like(interior_points["x"], Dose_value)
            }
            interior_inferencer = PointwiseInferencer(
                nodes=nodes,
                invar=invar,
                output_names=["N"],
                batch_size=cfg.batch_size.inference,
            )
            domain.add_inferencer(interior_inferencer, "Inference" + "_t_" + str(time_index) + "_Dose_" + str(Dose_index))
            Dose_index += 1
        time_index += 1
    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
