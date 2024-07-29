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

# use "mpirun -np 4 python brain_param_D_time.py"

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

    # Mesh bounds
    print("Bounds before scaling")
    print(noslip_mesh.bounds)
    print(interior_mesh.bounds)
    # bound_ranges: {x: (51.0, 239.0), y: (57.0, 197.0), z: (9.25, 109.25)} param_ranges: {}
    # bound_ranges: {x: (51.0, 239.0), y: (57.0, 197.0), z: (9.25, 109.25)} param_ranges: {}

    scale = 1 # keep dims in mm # 1e-3 # dimensions are in mm
    noslip_mesh.scale(scale)
    interior_mesh.scale(scale)

    print("Bounds after scaling")
    print(noslip_mesh.bounds)
    print(interior_mesh.bounds)

    # set_trace()

    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")

    # for now, using the continuous time approach (later, we can move to moving window approach for time evolution over larger durations)
    # parameterized time
    time_symbol = Symbol("t")
    time_range = (0.0, 5.0) # in days

    # parameterized diffusion coefficient
    D_symbol = Symbol("D")
    # D_range = (0.001, 0.01) # need to check these values to make them realistic
    D_range_mm2_day = (0.1, 0.8)  # in mm^2/day (as provided by David)
    # D_range_m2_s = (D_range_mm2_day[0] * 1.157e-11, D_range_mm2_day[1] * 1.157e-11)  # convert to m^2/s # check if these are realistic

    param_ranges = Parameterization({D_symbol: D_range_mm2_day, time_symbol: time_range})

    # make brain domain
    domain = Domain()

    # Need to reformulate N so that it is the normalized tumor density and is between 0 and 1
    # or can add a Sigmoid activation after the final layer of the NN
    tumor_diffusion_eq = Diffusion(T="N", D=D_symbol, dim=3, time=True) # the diffsuion equation will be solved for "N": the normalized tumor density

    # override defaults
    cfg.arch.fully_connected.layer_size = 128
    cfg.arch.fully_connected.nr_layers = 4
    # cfg.arch.fully_connected.activation_fn = [torch.nn.ReLU()] * 3 + [torch.nn.Sigmoid()]

    tumor_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t"), Key("D")],  # add "D" and "t" as additional input (parameterized)
        output_keys=[Key("N")],
        cfg=cfg.arch.fully_connected,
        # activation_fn=torch.nn.Sigmoid() # to constrain N between 0 and 1
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

    # Initial condition: normalized Gaussian distribution
    def initial_tumor_density(x, y, z):
        center_x, center_y, center_z = 100, 100, 50  # mm
        sigma = 2  # standard deviation in mm
        return np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2) / (2 * sigma ** 2))

    # Initial condition constraint

    initial_condition = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"N": initial_tumor_density}, # representing the initial tumor using a gaussian distribution centered around a specific location
        # outvar={"N": 0},
        batch_size=cfg.batch_size.initial_condition, # 10
        lambda_weighting={"N": 1.0},
        parameterization={D_symbol: D_range_mm2_day, time_symbol: 0} # fix t=0
    )
    domain.add_constraint(initial_condition, "initial_condition")

    # Interior inferencer
    inference_times = [0.0, 2.5, 5.0]
    inference_Ds = [0.1, 0.4, 0.8]
    # D_mapping = {0.01: "0p01", 0.1: "0p1", 0.8: "0p8"}

    interior_points = interior_mesh.sample_interior(cfg.batch_size.inference)
    time_index = 0
    for time_value in inference_times:
        D_index = 0
        for D_value in inference_Ds:
            # print(D_value, type(D_value))
            invar = {
                "x": interior_points["x"],
                "y": interior_points["y"],
                "z": interior_points["z"],
                "t": np.full_like(interior_points["x"], time_value),
                "D": np.full_like(interior_points["x"], D_value)
            }
            interior_inferencer = PointwiseInferencer(
                nodes=nodes,
                invar=invar,
                output_names=["N"],
                batch_size=cfg.batch_size.inference,
            )
            domain.add_inferencer(interior_inferencer, "Inference" + "_t_" + str(time_index) + "_D_" + str(D_index))
            D_index += 1
        time_index += 1
    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
