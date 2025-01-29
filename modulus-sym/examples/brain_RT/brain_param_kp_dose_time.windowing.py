# Following the https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/intermediate/moving_time_window.html case
# To run
# cd /mnt/Modulus_24p04/modulus-sym/examples/brain_RT
# use "mpirun -np 4 python brain_param_kp_dose_time.windowing.py"


import os
import sys
import warnings
import wandb
from pdb import set_trace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Dict, Tuple, Union, List
from modulus.sym.key import Key
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from sympy import Symbol, sqrt, Max, Eq
from modulus.sym.geometry import Bounds, Parameterization, Parameter
import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from sequential import SequentialSolver # importing a custom version of SequentialSolver instead of the default one
from moving_time_window import MovingTimeWindowArch
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
from modulus.sym.eq.pdes.diffusion import Diffusion
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.eq.pdes.basic import GradNormal
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.geometry.tessellation import Tessellation

from modulus.models.layers import FCLayer, Conv1dFCLayer
from modulus.sym.models.activation import Activation, get_activation_fn
from modulus.sym.models.arch import Arch

import sys
# print(sys.path)
# sys.path.append('/mnt/Modulus_24p04/modulus-sym/modulus/sym/eq/pdes')
# sys.path.append('/mnt/Modulus_24p04/modulus-sym/writable_dir/lib/python3.10/site-packages')
# from modulus.sym.eq.pdes.diffusion_proliferation_source import DiffusionProliferationSource

# for now, importing it from a file in the same dir due to pip install issues for the eq/pdes dir
from diffusion_proliferation_treatment import DiffusionProliferationTreatment


wandb.init(project="modulus_brain_RT", entity='tnnandi')

# formulation for parameterized proliferation rate (k_p) and time

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    cfg.network_dir = "./brain_param_kp_dose_time.lambda_1_100_100_100.updatedIC_maxsteps50K_decaysteps10K"
    os.makedirs(cfg.network_dir, exist_ok=True)
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

    # fixed diffusion coefficient, carrying capacity, dose, alpha, alpha_by_beta
    D_value = 0.125
    theta_value = 0.1
    # dose_value = 2 # try with a very small number and a very large number (e.g., 0 and 200) [original dose value: 2 Gy]
    alpha_value = 0.035
    alpha_by_beta_value = 10

    print("Bounds after scaling")
    print(farfield_mesh.bounds)
    print(interior_mesh.bounds)

    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")

    # moving window approach for time evolution over larger durations
    # parameterized time
    time_window_size = 1.0
    time_symbol = Symbol("t")
    time_range = {time_symbol: (0, time_window_size)}
    time_range = (0, time_window_size)
    nr_time_windows = 20

    # parameterized proliferation constant
    kp_symbol = Symbol("kp")
    kp_range = (0.0, 0.2)

    # parameterized dose
    dose_symbol = Symbol("dose")
    dose_range = (0.0, 8.0)

    param_ranges = Parameterization({kp_symbol: kp_range, dose_symbol: dose_range, time_symbol: time_range})

    # Can reformulate N so that it is the normalized tumor density and is between 0 and 1
    # or can add a Sigmoid activation after the final layer of the NN
    tumor_diffusion_proliferation_source_eq = DiffusionProliferationTreatment(T="N",  # the equation will be solved for "N": the normalized tumor density
                                                                              D=D_value,
                                                                              k_p=kp_symbol,
                                                                              theta=theta_value,
                                                                              alpha=alpha_value,
                                                                              alpha_by_beta=alpha_by_beta_value,
                                                                              dose=dose_symbol,
                                                                              dim=3,
                                                                              time=True)  

    print(tumor_diffusion_proliferation_source_eq.pprint())
    # set_trace()
    # override defaults
    cfg.arch.fully_connected.layer_size = 64
    cfg.arch.fully_connected.nr_layers = 4

    input_keys = [Key("x"), Key("y"), Key("z"), Key("t"), Key("kp"), Key("dose")]
    output_keys = [Key("N")]

    tumor_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t"), Key("kp"), Key("dose")],  # add "kp" and "t" as additional input (parameterized)
        output_keys=[Key("N")],
        cfg=cfg.arch.fully_connected,
    )

    t_treatment = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19]

    time_window_net = MovingTimeWindowArch(tumor_net,
                                           time_window_size,
                                           alpha=alpha_value,
                                           alpha_by_beta=alpha_by_beta_value,
                                           t_treatment=t_treatment)

    total_params_tumor_net = count_trainable_params(tumor_net)
    print(f"Total number of trainable parameters for tumor_net: {total_params_tumor_net}")
    print("########################################################")

    grad_normal_N = GradNormal("N", dim=3)

    nodes = (
            tumor_diffusion_proliferation_source_eq.make_nodes()
            + [time_window_net.make_node(name="time_window_network")]
            + grad_normal_N.make_nodes()
    )

    # add constraints to solver

    # make IC domain
    ic_domain = Domain("initial_conditions")
    # make moving time window domain
    window_domain = Domain("time_window")

    # farfield BC (surface boundary of the brain)
    # should be in both IC as well as the windowed domains
    farfield = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=farfield_mesh,
        outvar={"normal_gradient_N": 0},  # set the surface normal gradient of N to zero
        batch_size=cfg.batch_size.farfield,
        lambda_weighting={"normal_gradient_N": 1.0},
        parameterization=param_ranges
    )
    # ic_domain.add_constraint(farfield, "farfield")
    window_domain.add_constraint(farfield, "farfield")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"diffusion_proliferation_source_N": 0},  # can add a source term
        batch_size=cfg.batch_size.interior,
        lambda_weighting={"diffusion_proliferation_source_N": 100.0},
        parameterization=param_ranges,
    )
    # ic_domain.add_constraint(interior, "interior")
    window_domain.add_constraint(interior, "interior")

    # Initial condition specification for the tumor: normalized Gaussian distribution
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
        lambda_weighting={"N": 100.0},
        parameterization={kp_symbol: kp_range, dose_symbol: dose_range, time_symbol: 0}  # fix t=0 # why are kp, dose and t relevant here?
    )
    ic_domain.add_constraint(initial_condition, "initial_condition")

    # constraint to match with the solution from the previous time window (scaled by SF)
    match_previous_step = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"N_prev_step_diff": 0},
        batch_size=cfg.batch_size.interior,
        parameterization={kp_symbol: kp_range, dose_symbol: dose_range, time_symbol: 0},
        lambda_weighting={"N_prev_step_diff": 100.0},
    )
    window_domain.add_constraint(match_previous_step, "match_previous_step")

    # # function to multiply N from previous time step by survival fraction before using as IC for current window
    # # wrong, this updates the weights and not the values
    # def custom_update_operation():
    #     time_window_net.move_window(SF=SF)

    # Set up the solver with the sequential approach using moving time windows

    # slv = Solver(cfg, ic_domain)

    slv = SequentialSolver(
        cfg,
        [(1, ic_domain), (nr_time_windows, window_domain)],
        # custom_update_operation=custom_update_operation,
        custom_update_operation=time_window_net.move_window
    )

    ################################# INFERENCE ##########################
    # Interior inferencer
    # inference_times = [0.0, 5.0, 10.0, 15.0] # days
    inference_kps = [0.05, 0.2]
    inference_doses = [0.0, 2.0, 8.0]
    # D_mapping = {0.01: "0p01", 0.1: "0p1", 0.8: "0p8"}

    interior_points = interior_mesh.sample_interior(cfg.batch_size.inference)

    dose_index = 0
    for dose_value in inference_doses:
        kp_index = 0
        for kp_value in inference_kps:
            time_index = 0
            # carry out inference for these time instances within each time window
            for i, specific_time in enumerate(np.linspace(0, time_window_size, 3)):
                invar = {
                    "x": interior_points["x"],
                    "y": interior_points["y"],
                    "z": interior_points["z"],
                    "t": np.full_like(interior_points["x"], specific_time),
                    "kp": np.full_like(interior_points["x"], kp_value),
                    "dose": np.full_like(interior_points["x"], dose_value)
                }
                interior_inferencer = PointwiseInferencer(
                    nodes=nodes,
                    invar=invar,
                    output_names=["N"],
                    batch_size=cfg.batch_size.inference,
                )
                ic_domain.add_inferencer(interior_inferencer,
                                      "Inference_dose_" + str(dose_index) +
                                      "_kp_" + str(kp_index) +
                                      "_t_" + str(time_index))
                window_domain.add_inferencer(interior_inferencer,
                                      "Inference_dose_" + str(dose_index) +
                                      "_kp_" + str(kp_index) +
                                      "_t_" + str(time_index))
                time_index += 1
            kp_index += 1
        dose_index += 1

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
