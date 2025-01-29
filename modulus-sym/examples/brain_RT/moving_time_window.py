# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List
from modulus.sym.key import Key
import copy

import torch
import torch.nn as nn
from torch import Tensor

from modulus.sym.models.arch import Arch
from pdb import set_trace

class MovingTimeWindowArch(Arch):
    """
    Moving time window model the keeps track of
    current time window and previous window.

    Parameters
    ----------
    arch : Arch
        Modulus architecture to use for moving time window.
    window_size : float
        Size of the time window. This will be used to slide
        the window forward every iteration.
    """
    # TN: edited the code below to multiply N from previous time window by SF (that is a function of the parameterized dose)
    # doesn't work when jit and cuda_graphs are set to true. Also has issues with parallel training
    def __init__(
        self,
        arch: Arch,
        window_size: float,
        alpha: float,
        alpha_by_beta: float,
        t_treatment: List[float]
    ) -> None:
        output_keys = (
            arch.output_keys
            + [Key(x.name + "_prev_step") for x in arch.output_keys]
            + [Key(x.name + "_prev_step_diff") for x in arch.output_keys]
        )
        super().__init__(
            input_keys=arch.input_keys,
            output_keys=output_keys,
            periodicity=arch.periodicity,
        )

        # set networks for current and prev time window
        self.arch_prev_step = arch
        self.arch = copy.deepcopy(arch) # doesn't work with jit; so use jit=False in the command line or in the cfg file

        # store time window parameters
        self.window_size = window_size
        self.window_location = nn.Parameter(torch.empty(1), requires_grad=False)
        self.reset_parameters()

        # store alpha and alpha_by_beta for SF calculation
        self.alpha = alpha
        self.alpha_by_beta = alpha_by_beta
        self.t_treatment = t_treatment

        # precompute the tensor for t_treatment to avoid issues during CUDA stream capture
        self.t_treatment_tensor = torch.tensor(self.t_treatment, dtype=torch.float32, device='cuda')
        self.first_window_tensor = torch.tensor(1.0, dtype=torch.float32, device='cuda')

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        with torch.no_grad():
            in_vars["t"] += self.window_location
        y_prev_step = self.arch_prev_step.forward(in_vars)

        # apply scaling to the N from previous time window only for these time instances (RT days)
        # current_time = float(self.window_location.item())
        current_time = self.window_location.clone()
        # print("current time: ", current_time)
        # check if the current time matches any treatment time using tensor operations
        is_treatment_time = torch.any(current_time == self.t_treatment_tensor)
        is_first_window = torch.any(current_time == self.first_window_tensor)
        # print("is_treatment_time: ", is_treatment_time)
        # print("is_first_window: ", is_first_window)
        # print("dose: ", in_vars["dose"])

        # For the IC time window, we should have N values between 0 and 1 though current approach has N limited to around 0.39
        # So, only for the first time window, use softmax to map y_prev_step[key] to [0,1], and then multiply by SF
        # For the first time window (IC), map y_prev_step[key] to [0,1] using softmax if current_time is exactly one
        # if is_first_window:  # Exact match for the first time window
        #     for key in y_prev_step.keys():
        #         # apply softmax to map N to [0, 1]
        #         y_prev_step[key] = torch.softmax(y_prev_step[key], dim=0)


        #  if the time window corresponds to the treatment time, reduce N to account for effects of RT, and then proceed with the calculations for the time window
        if is_treatment_time:
            # calculate SF based on dose
            dose = in_vars["dose"]
            SF = torch.exp(-self.alpha * dose * (1 + dose / self.alpha_by_beta))
            # multiply the output of the previous time window with SF to account for the cells killed by RT; N for the current time window
            for key in y_prev_step.keys():
                y_prev_step[key] = y_prev_step[key] * SF
                # y_prev_step[key] = torch.where(is_treatment_time, y_prev_step[key] * SF, y_prev_step[key])
                # y_prev_step[key] = torch.where(is_treatment_time, y_prev_step[key], y_prev_step[key])

        y = self.arch.forward(in_vars)
        y_keys = list(y.keys())
        for key in y_keys:
            y_prev = y_prev_step[key]
            y[key + "_prev_step"] = y_prev
            y[key + "_prev_step_diff"] = y[key] - y_prev
        return y

    def move_window(self):
        self.window_location.data += self.window_size
        for param, param_prev_step in zip(
            self.arch.parameters(), self.arch_prev_step.parameters()
        ):
            param_prev_step.data = param.detach().clone().data
            param_prev_step.requires_grad = False

    def reset_parameters(self) -> None:
        nn.init.constant_(self.window_location, 0)
