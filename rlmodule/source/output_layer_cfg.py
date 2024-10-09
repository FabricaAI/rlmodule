from dataclasses import MISSING
from typing import Sequence, Union

import gym
import gymnasium
from rlmodule.source.output_layer import GaussianLayer, DeterministicLayer
import torch.nn as nn

# use isaac-lab native configclass if available to avoid it being declared twice
try:
    from omni.isaac.lab.utils import configclass
except ImportError:
    from rlmodule.source.nvidia_utils import configclass

@configclass
class OutputLayerCfg:
    class_type: type[nn.Module] = MISSING,
    output_size: Union[int, Sequence[int], gym.Space, gymnasium.Space] = MISSING,  
    output_activation: nn.Module = MISSING,

    clip_actions: bool = False,  # TODO what is clip action doing

@configclass
class GaussianLayerCfg(OutputLayerCfg):
    class_type: type[GaussianLayer] = GaussianLayer
    output_activation: type[nn.Module] = nn.Tanh

    clip_log_std: bool = True
    min_log_std: float = -20
    max_log_std: float = 2
    initial_log_std: float = 0
    reduction: str = "sum"

@configclass
class DeterministicLayerCfg(OutputLayerCfg):
    class_type: type[DeterministicLayer] = DeterministicLayer
    output_size: Union[int, Sequence[int], gym.Space, gymnasium.Space] = 1
    output_activation: nn.Module = nn.Identity()