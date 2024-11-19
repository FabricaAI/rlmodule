from typing import Union

from functools import reduce
from itertools import repeat

import torch
import torch.nn as nn


class LogStd(nn.Module):
    """Base class that defines interface for standard deviation computation in GaussianLayer."""

    def __init__(self, device: Union[str, torch.device], input_size: int, output_size: int, cfg):
        """Initialize Standard deviation computation module."""
        super().__init__()
        self.device = device

        self._input_size = input_size
        self._output_size = output_size

        self._clip_log_std = cfg.clip_log_std
        self._log_std_min = cfg.min_log_std
        self._log_std_max = cfg.max_log_std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute standard deviation from layer input."""
        raise NotImplementedError(f"Forward method is not implemented for {type(self)}")

    def clip_std(self, raw_std: torch.Tensor) -> torch.Tensor:
        """Constrain standard deviation."""
        if self._clip_log_std:
            return torch.clamp(raw_std, self._log_std_min, self._log_std_max)
        return raw_std


class ParameterLogStd(LogStd):
    """Class that models standard deviation in GaussianLayer as nn.Parameter.

    This module creates `output_size` nn.Parameters to represent log_std for each output
    action.
    """

    def __init__(self, device: Union[str, torch.device], input_size: int, output_size: int, cfg):
        """Initialize Standard deviation computation module."""
        super().__init__(device, input_size, output_size, cfg)

        self._log_parameter = nn.Parameter(cfg.initial_log_std * torch.ones(self._output_size))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute standard deviation from layer input."""
        return self.clip_std(self._log_parameter)


class NNLogStd(LogStd):
    """Class that models standard deviation in GaussianLayer by user-provided network.

    This module instantiates user provided network and processes its outputs with the linear layer of `output_size`
    with nn.Identity activation function. This way of deriving std may be beneficial compared to using basic
    ParameterLogStd because std computation depends on agent's state.
    Note:
        Currently support only MLP.
    """

    def __init__(self, device: Union[str, torch.device], input_size: int, output_size: int, cfg):
        """Initialize Standard deviation computation module."""
        super().__init__(device, input_size, output_size, cfg)

        network_cfg = cfg.network_cfg
        # If user did not provide any hidden layers
        if network_cfg is None:
            self._net = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.Identity(),
            )
        else:
            network_cfg.input_states = input_size
            hidden_net = network_cfg.module(network_cfg).to(device)

            layers = [
                hidden_net,
                nn.Linear(network_cfg.hidden_units[-1], output_size),
                nn.Identity(),
            ]
            self._net = nn.Sequential(*layers)

        self._net.to(device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute logarithm of standard deviation from layer input."""
        return self.clip_std(self._net(input))


class CombinedLogStd(LogStd):
    """Class that combines functionality of multiple LogStd modules with predefined function.

    Combining multiple sources of LogStd can be beneficial to model more advanced exploration techniques.
    For example combination of ParameterLogStd and NNLogStd with max allows agent to explore in certain
    states (comes from NNLogStd) but also prevents a common problem where usage of NNLogStd leads to
    insufficient exploration in other states.
    """

    def __init__(self, device: Union[str, torch.device], input_size: int, output_size: int, cfg):
        super().__init__(device, input_size, output_size, cfg)

        self._std_modules = [
            module_cfg.class_type(device, input_size, output_size, module_cfg).to(device)
            for module_cfg in cfg.combined_modules
        ]

        if cfg.combination_constants is None:
            self._combination_constants = list(repeat(1.0, len(self._std_modules)))
        else:
            self._combination_constants = cfg.combination_constants
            if len(self._std_modules) != len(self._combination_constants):
                raise ValueError(
                    f"Number of combination constants {len(self._combination_constants)} does not match number of"
                    f" modules {len(self._std_modules)}"
                )

        if cfg.combination_method == "max":
            self._combination_method = torch.max
        elif cfg.combination_method == "min":
            self._combination_method = torch.min
        else:
            raise ValueError(
                f"Combination method `{cfg.combination_method}` for standard deviation combination module unknown."
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute standard deviation as a combination from multiple Std modules."""
        collected_stds = [
            self._combination_constants[module_id] * module(input) for module_id, module in enumerate(self._std_modules)
        ]
        return self.clip_std(reduce(self._combination_method, collected_stds))
