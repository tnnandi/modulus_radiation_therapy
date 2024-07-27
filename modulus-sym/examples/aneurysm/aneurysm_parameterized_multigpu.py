import os
import warnings

import torch
import numpy as np
from sympy import Symbol, sqrt, Max

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.sym.geometry import Bounds, Parameterization, Parameter
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.key import Key
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.geometry.tessellation import Tessellation

@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
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

    # params
    nu = 0.0025

    # Inlet velocity is now parameterized
    inlet_vel = Parameter("inlet_vel")
    # range of values
    inlet_vel_range = (1.0, 20.0) # min & max. The number of points is not chosen here, instead InferenceDomain chooses the inlet velocities where inference is required

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
    ns = NavierStokes(nu=nu * scale, rho=1.0, dim=3, time=False)
    normal_dot_vel = NormalDotVec(["u", "v", "w"])
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("inlet_vel")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )

    # Use DataParallel to distribute computations across multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        flow_net = torch.nn.DataParallel(flow_net)

    flow_net.to('cuda')

    # add the flow network node correctly based on whether DataParallel is used
    if isinstance(flow_net, torch.nn.DataParallel):
        flow_net_node = flow_net.module.make_node(name="flow_network")
    else:
        flow_net_node = flow_net.make_node(name="flow_network")

    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        # + [flow_net.make_node(name="flow_network")
        + [flow_net_node]
    )

    # check if flow_net is correctly instantiated
    # print("Flow Network Outputs: ", flow_net_node.output_keys)
    print("Nodes: ", [node.name for node in nodes])

    # add constraints to solver
    # inlet
    u, v, w = circular_parabola(
        Symbol("x"),
        Symbol("y"),
        Symbol("z"),
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
        parameterization=Parameterization({inlet_vel: inlet_vel_range}),
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_mesh,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        parameterization=Parameterization({inlet_vel: inlet_vel_range})
    )
    domain.add_constraint(outlet, "outlet")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=noslip_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.no_slip,
        parameterization=Parameterization({inlet_vel: inlet_vel_range})
    )
    domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=Parameterization({inlet_vel: inlet_vel_range})
    )
    domain.add_constraint(interior, "interior")

    # Get the inlet flow rate (assuming parabolic profile)
    inlet_flow_rate = 2.0 / 3.0 * inlet_area * inlet_vel

    # Integral Continuity 1
    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_mesh,
        outvar={"normal_dot_vel": inlet_flow_rate},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
        parameterization=Parameterization({inlet_vel: inlet_vel_range})
    )
    domain.add_constraint(integral_continuity, "integral_continuity_1")

    # Integral Continuity 2
    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_mesh,
        outvar={"normal_dot_vel": inlet_flow_rate},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
        parameterization=Parameterization({inlet_vel: inlet_vel_range})
    )
    domain.add_constraint(integral_continuity, "integral_continuity_2")

    # # add pressure monitor
    # pressure_monitor = PointwiseMonitor(
    #     inlet_mesh.sample_boundary(16),
    #     output_names=["p"],
    #     metrics={"pressure_drop": lambda var: torch.mean(var["p"])},
    #     nodes=nodes,
    # )
    # domain.add_monitor(pressure_monitor)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
