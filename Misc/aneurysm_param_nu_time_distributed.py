import os
import warnings

import torch
import numpy as np
from sympy import Symbol, sqrt, Max, Eq
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
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.geometry.tessellation import Tessellation
from modulus.sym.distributed import DistributedManager


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # Initialize the DistributedManager.
    # currently torchrun (or any other pytorch compatible launcher), mpirun (OpenMPI) and SLURM based launchers are supported.
    DistributedManager.initialize()
    # initialize distributed manager
    dist = DistributedManager()

    # read stl files to make geometry
    point_path = to_absolute_path("./stl_files")
    inlet_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_inlet.stl", airtight=False
    )
    outlet_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_outlet.stl", airtight=False
    )
    noslip_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_noslip.stl", airtight=False
    )
    integral_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_integral.stl", airtight=False
    )
    interior_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_closed.stl", airtight=True
    )

    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")

    # for now, using the continuous time approach (later, we can move to moving window approach for time evolution over larger durations)
    # parameterized time
    time_symbol = Symbol("t")
    time_range = (0.0, 1.0)

    # params
    inlet_vel = 1.5

    # parameterized viscosity
    nu = Symbol("nu")
    nu_range = (0.001, 0.01)

    param_ranges = Parameterization({nu: nu_range, time_symbol: time_range})


    # inlet velocity profile
    def circular_parabola(x, y, z, center, normal, radius, max_vel):
        centered_x = x - center[0]
        centered_y = y - center[1]
        centered_z = z - center[2]
        distance = sqrt(centered_x**2 + centered_y**2 + centered_z**2)
        parabola = max_vel * Max((1 - (distance / radius) ** 2), 0)
        return normal[0] * parabola, normal[1] * parabola, normal[2] * parabola

    # normalize meshes
    def normalize_mesh(mesh, center, scale):
        mesh = mesh.translate([-c for c in center])
        mesh = mesh.scale(scale)
        return mesh

    # normalize invars
    def normalize_invar(invar, center, scale, dims=2):
        invar["x"] -= center[0]
        invar["y"] -= center[1]
        invar["z"] -= center[2]
        invar["x"] *= scale
        invar["y"] *= scale
        invar["z"] *= scale
        if "area" in invar.keys():
            invar["area"] *= scale**dims
        return invar

    # scale and normalize mesh and openfoam data
    center = (-18.40381048596882, -50.285383353981196, 12.848136936899031)
    scale = 0.4
    inlet_mesh = normalize_mesh(inlet_mesh, center, scale)
    outlet_mesh = normalize_mesh(outlet_mesh, center, scale)
    noslip_mesh = normalize_mesh(noslip_mesh, center, scale)
    integral_mesh = normalize_mesh(integral_mesh, center, scale)
    interior_mesh = normalize_mesh(interior_mesh, center, scale)

    # geom params
    inlet_normal = (0.8526, -0.428, 0.299)
    inlet_area = 21.1284 * (scale**2)
    inlet_center = (-4.24298030045776, 4.082857101816247, -4.637790193399717)
    inlet_radius = np.sqrt(inlet_area / np.pi)
    outlet_normal = (0.33179, 0.43424, 0.83747)
    outlet_area = 12.0773 * (scale**2)
    outlet_radius = np.sqrt(outlet_area / np.pi)

    # make aneurysm domain
    domain = Domain()

    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nu * scale, rho=1.0, dim=3, time=True)
    normal_dot_vel = NormalDotVec(["u", "v", "w"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t"), Key("nu")], # add "nu" and "t" as additional input
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )

    # Set up DistributedDataParallel if using more than a single process.
    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            arch = torch.nn.parallel.DistributedDataParallel(
                arch.model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # add constraints to solver
    # inlet
    u, v, w = circular_parabola(
        x,
        y,
        z,
        center=inlet_center,
        normal=inlet_normal,
        radius=inlet_radius,
        max_vel=inlet_vel,
    )
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"u": u, "v": v, "w": w},
        batch_size=cfg.batch_size.inlet,
        parameterization=param_ranges
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_mesh,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        parameterization=param_ranges
    )
    domain.add_constraint(outlet, "outlet")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=noslip_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
        parameterization=param_ranges
    )
    domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=param_ranges,
    )
    domain.add_constraint(interior, "interior")

    # Initial condition
    initial_condition = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"u": 0, "v": 0, "w": 0, "p": 0},
        batch_size=10, #cfg.batch_size.initial_condition,
        lambda_weighting={"u": 1.0, "v": 1.0, "w": 1.0, "p": 1.0},
        parameterization={time_symbol: 0.0, nu: nu_range},
        # criteria=Eq(Symbol("t"), 0),
    )
    domain.add_constraint(initial_condition, "initial_condition")

    # # Get the inlet flow rate (assuming parabolic profile)
    # inlet_flow_rate = 2.0 / 3.0 * inlet_area * inlet_vel
    #
    # # Integral Continuity 1
    # integral_continuity = IntegralBoundaryConstraint(
    #     nodes=nodes,
    #     geometry=outlet_mesh,
    #     outvar={"normal_dot_vel": inlet_flow_rate},
    #     batch_size=1,
    #     integral_batch_size=cfg.batch_size.integral_continuity,
    #     lambda_weighting={"normal_dot_vel": 0.1},
    # )
    # domain.add_constraint(integral_continuity, "integral_continuity_1")
    #
    # # Integral Continuity 2
    # integral_continuity = IntegralBoundaryConstraint(
    #     nodes=nodes,
    #     geometry=integral_mesh,
    #     outvar={"normal_dot_vel": -inlet_flow_rate},
    #     batch_size=1,
    #     integral_batch_size=cfg.batch_size.integral_continuity,
    #     lambda_weighting={"normal_dot_vel": 0.1},
    # )
    # domain.add_constraint(integral_continuity, "integral_continuity_1")

    # # add pressure monitor
    # pressure_monitor = PointwiseMonitor(
    #     inlet_mesh.sample_boundary(16),
    #     output_names=["p"],
    #     metrics={"pressure_drop": lambda var: torch.mean(var["p"])},
    #     nodes=nodes,
    # )
    # domain.add_monitor(pressure_monitor)

    # Interior inferencer
    inference_times = [0.0, 0.5, 1.0]
    inference_nus = [0.001, 0.005, 0.01]

    interior_points = interior_mesh.sample_interior(1000)
    for t in inference_times:
        for nu in inference_nus:
            invar = {
                "x": interior_points["x"],
                "y": interior_points["y"],
                "z": interior_points["z"],
                "t": np.full_like(interior_points["x"], t),
                "nu": np.full_like(interior_points["x"], nu)
            }
            interior_inferencer = PointwiseInferencer(
                nodes=nodes,
                invar=invar,
                output_names=["u", "v", "w", "p"],
                batch_size=cfg.batch_size.interior,
            )
            inferencer_name = f"inferencer_t_{t}_nu_{nu}"
            domain.add_inferencer(interior_inferencer, inferencer_name)

    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
