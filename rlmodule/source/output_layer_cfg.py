from typing import Sequence, Union

from dataclasses import MISSING
import gym
import gymnasium

import torch.nn as nn

from rlmodule.source.output_layer import DeterministicLayer, GaussianLayer


# use isaac-lab native configclass if available to avoid double declaration
try:
    from omni.isaac.lab.utils import configclass
except ImportError:
    from rlmodule.source.nvidia_utils import configclass


@configclass
class OutputLayerCfg:
    class_type: type[nn.Module] = MISSING
    """Descendant class of the OutputLayer class implementing corresponding layer."""

    output_size: Union[int, Sequence[int], gym.Space, gymnasium.Space] = None
    """Output size of the output layer. Usually an action (sub)space.
    Can be kept as None if the value is set during execution time.
    """

    output_activation: nn.Module = MISSING
    """Activation function for the output layer"""

    output_scale: float = 1.0
    """Scales output of layer by output scale"""

    clip_actions: bool = False
    """Flag to indicate whether the actions should be clipped. Only supported for gym/gymnasium space"""


@configclass
class GaussianLayerCfg(OutputLayerCfg):
    """Keep configuration parameters for GaussianLayer class."""

    class_type: type[GaussianLayer] = GaussianLayer
    output_activation: type[nn.Module] = nn.Tanh

    clip_log_std: bool = True
    """Flag to indicate whether the log standard deviations should be clipped"""

    min_log_std: float = -20.0
    """Minimum value of the log standard deviation."""

    max_log_std: float = 2.0
    """Maximum value of the log standard deviation"""

    initial_log_std: float = 0.0
    """Initial value for the log standard deviation"""

    reduction: str = "sum"
    """Reduction method for returning the log probability density function: (default: ``"sum"``).
    Supported values are ``"mean"``, ``"sum"``, ``"prod"`` and ``"none"``. If "``none"``, the log probability density
    function is returned as a tensor of shape ``(num_samples, num_actions)`` instead of ``(num_samples, 1)
    """


@configclass
class DeterministicLayerCfg(OutputLayerCfg):
    """Keep configuration parameters for DeterministicLayer class.
    Default values (output_size = 1, output_activation = Identity) are set for a use as value function predictor.
    """

    class_type: type[DeterministicLayer] = DeterministicLayer
    output_size: Union[int, Sequence[int], gym.Space, gymnasium.Space] = 1
    output_activation: nn.Module = nn.Identity
