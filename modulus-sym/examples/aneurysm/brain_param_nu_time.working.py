import os
import warnings
from pdb import set_trace
import torch
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
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.advection_diffusion import AdvectionDiffusion
from modulus.sym.eq.pdes.diffusion import Diffusion
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.geometry.tessellation import Tessellation


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # read stl files to make geometry
    point_path = to_absolute_path("./stl_files")

    noslip_mesh = Tessellation.from_stl(
        point_path + "/brain_surface_subsample_4X.stl", airtight=False
    )

    interior_mesh = Tessellation.from_stl(
        point_path + "/brain_volume_subsample_4X.stl", airtight=True
    )

    scale = 1 # dimensions are in mm
    noslip_mesh.scale(scale)
    interior_mesh.scale(scale)

    # Mesh bounds
    print(noslip_mesh.bounds)
    print(interior_mesh.bounds)

    set_trace()

    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")

    # for now, using the continuous time approach (later, we can move to moving window approach for time evolution over larger durations)
    # parameterized time
    time_symbol = Symbol("t")
    time_range = (0.0, 1.0)

    # parameterized diffusion coefficient
    D_symbol = Symbol("D")
    D_range = (0.001, 0.01) # need to check these values to make them realistic

    param_ranges = Parameterization({D_symbol: D_range, time_symbol: time_range})

    # make brain domain
    domain = Domain()

    # Need to reformulate N so that it is the normalized tumor density and is between 0 and 1
    tumor_diffusion_eq = Diffusion(T="N", D="D", dim=3, time=True) # the diffsuion equation will be solved for "N": the normalized tumor density

    tumor_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t"), Key("D")],  # add "D" and "t" as additional input (parameterized)
        output_keys=[Key("N")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
            tumor_diffusion_eq.make_nodes()
            + [tumor_net.make_node(name="flow_network")]
    )

    # add constraints to solver

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=noslip_mesh,
        outvar={"N": 0},
        batch_size=cfg.batch_size.no_slip,
        parameterization=param_ranges
    )
    domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"diffusion_N": 0}, # can add a source term
        batch_size=cfg.batch_size.interior,
        parameterization=param_ranges,
    )
    domain.add_constraint(interior, "interior")

    # Initial condition
    initial_condition = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"N": lambda x, y, z: np.exp(-((x-100)**2 + (y-100)**2 + (z-100)**2) / (2 * 5**2))}, # representing the initial tumor using a gaussian distribution centered around a specific location
        batch_size=10,  # cfg.batch_size.initial_condition,
        lambda_weighting={"N": 1.0},
        parameterization={time_symbol: time_range, D_symbol: D_range},
        # criteria=Eq(Symbol("t"), 0),
    )
    domain.add_constraint(initial_condition, "initial_condition")

    # # Interior inferencer
    # inference_times = [0.0, 0.5, 1.0]
    # inference_nus = [0.001, 0.008, 0.02]
    #
    # interior_points = interior_mesh.sample_interior(1000)
    # for time_value in inference_times:
    #     for nu_value in inference_nus:
    #         invar = {
    #             "x": interior_points["x"],
    #             "y": interior_points["y"],
    #             "z": interior_points["z"],
    #             "t": np.full_like(interior_points["x"], time_value),
    #             "nu": np.full_like(interior_points["x"], nu_value)
    #         }
    #         interior_inferencer = PointwiseInferencer(
    #             nodes=nodes,
    #             invar=invar,
    #             output_names=["N"],
    #             batch_size=cfg.batch_size.interior,
    #         )
    #         inferencer_name = f"tumor_inferencer_t_{time_value}_nu_{nu_value:.3e}"
    #         domain.add_inferencer(interior_inferencer, inferencer_name)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
