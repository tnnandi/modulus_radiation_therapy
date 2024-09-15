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
# use "mpirun -np 4 python brain_param_kp_time.py"

import sys
# print(sys.path)
# sys.path.append('/mnt/Modulus_24p04/modulus-sym/modulus/sym/eq/pdes')
# sys.path.append('/mnt/Modulus_24p04/modulus-sym/writable_dir/lib/python3.10/site-packages')
# from modulus.sym.eq.pdes.diffusion_proliferation_source import DiffusionProliferationSource

# for now, importing it from a file in the same dir due to pip install issues for the eq/pdes dir
from diffusion_proliferation_treatment import DiffusionProliferationTreatment

# wandb.init(project="modulus_brain_RT", entity='tnnandi')

# formulation for parameterized proliferation rate (k_p) and time

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Custom architecture classes to add Sigmoid activation after the final NN layer (codes adapted from sym/models/fully_connected.py)
class FullyConnectedArchCore(nn.Module):
    def __init__(
            self,
            in_features: int = 512,
            layer_size: int = 512,
            out_features: int = 512,  # default value, the actual value will be based on length of output_keys
            nr_layers: int = 6,
            skip_connections: bool = False,
            activation_fn: Activation = Activation.SILU,
            adaptive_activations: bool = False,
            weight_norm: bool = True,
            conv_layers: bool = False,
    ) -> None:
        super().__init__()

        self.skip_connections = skip_connections

        if conv_layers:
            fc_layer = Conv1dFCLayer
        else:
            fc_layer = FCLayer

        if adaptive_activations:
            activation_par = nn.Parameter(torch.ones(1))
        else:
            activation_par = None

        if not isinstance(activation_fn, list):
            activation_fn = [activation_fn] * nr_layers
        if len(activation_fn) < nr_layers:
            activation_fn = activation_fn + [activation_fn[-1]] * (
                    nr_layers - len(activation_fn)
            )

        self.layers = nn.ModuleList()

        layer_in_features = in_features
        for i in range(nr_layers):
            self.layers.append(
                fc_layer(
                    in_features=layer_in_features,
                    out_features=layer_size,
                    activation_fn=get_activation_fn(
                        activation_fn[i], out_features=out_features
                    ),
                    weight_norm=weight_norm,
                    activation_par=activation_par,
                )
            )
            layer_in_features = layer_size

        # modify the final layer to include a Sigmoid activation to constrain N between 0 and 1
        self.final_layer = nn.Sequential(
            fc_layer(
                in_features=layer_size,
                out_features=out_features,
                activation_fn=None,
                weight_norm=False,
                activation_par=None,
            ),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        x_skip: Optional[Tensor] = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    x, x_skip = x + x_skip, x
                else:
                    x_skip = x

        x = self.final_layer(x)
        return x

    def get_weight_list(self):
        weights = [layer.conv.weight for layer in self.layers] + [
            self.final_layer[0].conv.weight  # Accessing the actual linear layer within the Sequential
        ]
        biases = [layer.conv.bias for layer in self.layers] + [
            self.final_layer[0].conv.bias  # Accessing the actual linear layer within the Sequential
        ]
        return weights, biases


class FullyConnectedArch(Arch):
    def __init__(
            self,
            input_keys: List[Key],
            output_keys: List[Key],
            detach_keys: List[Key] = [],
            layer_size: int = 512,
            nr_layers: int = 6,
            activation_fn=Activation.SILU,
            periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
            skip_connections: bool = False,
            adaptive_activations: bool = False,
            weight_norm: bool = True,
    ) -> None:
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            detach_keys=detach_keys,
            periodicity=periodicity,
        )

        if self.periodicity is not None:
            in_features = sum(
                [
                    x.size
                    for x in self.input_keys
                    if x.name not in list(periodicity.keys())
                ]
            ) + +sum(
                [
                    2 * x.size
                    for x in self.input_keys
                    if x.name in list(periodicity.keys())
                ]
            )
        else:
            in_features = sum(self.input_key_dict.values())
        out_features = sum(self.output_key_dict.values())

        self._impl = FullyConnectedArchCore(
            in_features,
            layer_size,
            out_features,
            nr_layers,
            skip_connections,
            activation_fn,
            adaptive_activations,
            weight_norm,
        )

    def _tensor_forward(self, x: Tensor) -> Tensor:
        x = self.process_input(
            x,
            self.input_scales_tensor,
            periodicity=self.periodicity,
            input_dict=self.input_key_dict,
            dim=-1,
        )
        x = self._impl(x)
        x = self.process_output(x, self.output_scales_tensor)
        return x

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        y = self._tensor_forward(x)
        return self.split_output(y, self.output_key_dict, dim=-1)

    def _dict_forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        y = self._impl(x)
        return self.prepare_output(
            y, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )


class CustomFullyConnectedArch(FullyConnectedArch):
    def __init__(
            self,
            input_keys: List[Key],
            output_keys: List[Key],
            detach_keys: List[Key] = [],
            layer_size: int = 512,
            nr_layers: int = 6,
            activation_fn=Activation.SILU,
            periodicity: Union[Dict[str, Tuple[float, float]], None] = None,
            skip_connections: bool = False,
            adaptive_activations: bool = False,
            weight_norm: bool = True,
    ) -> None:
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            detach_keys=detach_keys,
            layer_size=layer_size,
            nr_layers=nr_layers,
            activation_fn=activation_fn,
            periodicity=periodicity,
            skip_connections=skip_connections,
            adaptive_activations=adaptive_activations,
            weight_norm=weight_norm,
        )

        # Use the custom core with the original output features
        self._impl = FullyConnectedArchCore(
            in_features=sum(self.input_key_dict.values()),
            layer_size=layer_size,
            out_features=sum(self.output_key_dict.values()),
            nr_layers=nr_layers,
            skip_connections=skip_connections,
            activation_fn=activation_fn,
            adaptive_activations=adaptive_activations,
            weight_norm=weight_norm,
        )


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    cfg.network_dir = "./brain_param_kp_time.dose_2.lambda_1_100_50.layer_size_128.nr_layers_8_test"
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
    dose_value = 2 # try with a very small number and a very large number (e.g., 0 and 200) [original dose value: 2 Gy]
    alpha_value = 0.035
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
    time_range = (0.0, 20.0)  # in days

    # parameterized proliferation constant
    kp_symbol = Symbol("kp")
    kp_range = (0.01, 0.5)

    param_ranges = Parameterization({kp_symbol: kp_range, time_symbol: time_range})

    # make brain domain
    domain = Domain()

    # Need to reformulate N so that it is the normalized tumor density and is between 0 and 1
    # or can add a Sigmoid activation after the final layer of the NN
    tumor_diffusion_proliferation_source_eq = DiffusionProliferationTreatment(T="N",
                                                                           D=D_value,
                                                                           k_p=kp_symbol,
                                                                           theta=theta_value,
                                                                           alpha=alpha_value,
                                                                           alpha_by_beta=alpha_by_beta_value,
                                                                           dose=dose_value,
                                                                           dim=3,
                                                                           time=True)  # the equation will be solved for "N": the normalized tumor density

    set_trace()
    # override defaults
    cfg.arch.fully_connected.layer_size = 128
    cfg.arch.fully_connected.nr_layers = 8#4

    input_keys = [Key("x"), Key("y"), Key("z"), Key("t"), Key("kp")]
    output_keys = [Key("N")]

    # # Use the custom architecture class
    # tumor_net = CustomFullyConnectedArch(
    #     input_keys=input_keys,
    #     output_keys=output_keys,
    #     layer_size=cfg.arch.fully_connected.layer_size,
    #     nr_layers=cfg.arch.fully_connected.nr_layers,
    # )

    tumor_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t"), Key("kp")],  # add "kp" and "t" as additional input (parameterized)
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
        parameterization={kp_symbol: kp_range, time_symbol: 0}  # fix t=0
    )
    domain.add_constraint(initial_condition, "initial_condition")

    # Interior inferencer
    inference_times = [0.0, 5.0, 10.0, 15.0] # days
    inference_kps = [0.02, 0.05, 0.3]
    # D_mapping = {0.01: "0p01", 0.1: "0p1", 0.8: "0p8"}

    interior_points = interior_mesh.sample_interior(cfg.batch_size.inference)
    kp_index = 0
    for kp_value in inference_kps:
        time_index = 0
        for time_value in inference_times:
            # print(kp_value, type(kp_value))
            invar = {
                "x": interior_points["x"],
                "y": interior_points["y"],
                "z": interior_points["z"],
                "t": np.full_like(interior_points["x"], time_value),
                "kp": np.full_like(interior_points["x"], kp_value)
            }
            interior_inferencer = PointwiseInferencer(
                nodes=nodes,
                invar=invar,
                output_names=["N"],
                batch_size=cfg.batch_size.inference,
            )
            domain.add_inferencer(interior_inferencer, "Inference" + "_kp_" + str(kp_index) + "_t_" + str(time_index))
            time_index += 1
        kp_index += 1
    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
