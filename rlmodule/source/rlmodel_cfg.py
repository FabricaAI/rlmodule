from typing import Optional, Union

from dataclasses import MISSING

import torch
import torch.nn as nn

from rlmodule.source.output_layer_cfg import GaussianLayerCfg, OutputLayerCfg
from rlmodule.source.rlmodel import RLModel, SharedRLModel


# use isaac-lab native configclass if available to avoid it being declared twice
try:
    from omni.isaac.lab.utils import configclass
except ImportError:
    from rlmodule.source.nvidia_utils import configclass


@configclass
class BaseRLCfg:
    class_type: type[RLModel] = MISSING
    network: nn.Module = MISSING
    device: Optional[Union[str, torch.device]] = MISSING


@configclass
class RLModelCfg(BaseRLCfg):
    class_type: type[RLModel] = RLModel
    output_layer: type[OutputLayerCfg] = GaussianLayerCfg()


@configclass
class SharedRLModelCfg(BaseRLCfg):
    """Model where policy and value are sharing the function approximator. Only differ in output layer.
    This can be used for PPO, A2C, RPO, TRPO algorithms.
    """

    class_type: type[SharedRLModel] = SharedRLModel
    policy_output_layer: OutputLayerCfg = MISSING
    value_output_layer: OutputLayerCfg = MISSING


@configclass
class SeparatedRLModelCfg:
    """Convenience class to hold two separate instances of RLModelCfg. One for policy and one for value (or critic)."""

    policy: RLModelCfg = MISSING
    """Hold configuration of policy model"""
    value: RLModelCfg = MISSING
    """Hold configuration of value (or critic) model"""
