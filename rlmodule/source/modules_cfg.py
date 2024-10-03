import torch.nn as nn

from dataclasses import MISSING
from collections.abc import Callable
from typing import List, Union

from rlmodule.source.modules import GRU, LSTM, MLP, RNN, RnnMlp

# use isaac-lab native configclass if available to avoid it being declared twice
try:
    from omni.isaac.lab.utils import configclass
except ImportError:
    print("Importing local configclass.")
    from rlmodule.source.nvidia_utils import configclass

#TODO next
# (Done) check how to do own @configclass .. or use one from isaac by default (print something to know version)
# (Done) RNN code modify to use data class
# (Done) RnnMLP
# (Done) Custom function Network
# Move things logically, annotate cfgs in modules
# Handle input shapes better?
# Convert all rest modules to use Configclass
# Rearrange library in the way it can be used for other rllibs and for torch/jax?
# Create examples - mlp, rnn,gru,lstm, Lstmmlp, shared - separate , custom net by fcion
# WRITE README tutorial
# Launch new version to pip

# Import new version in Isaac-lab


@configclass
class NetworkCfg:
    module: Union[nn.Module, Callable[..., nn.Module]] = MISSING
    input_size: int = -1 # change type, -1 is inferred
    

@configclass
class MlpCfg(NetworkCfg):
    module: type[MLP] = MLP

    hidden_units: List[int] = MISSING
    activations: List[nn.Module] = MISSING
    

@configclass
class RnnBaseCfg(NetworkCfg):
    num_envs: int = MISSING
    num_layers: int = MISSING
    hidden_size: int = MISSING
    sequence_length: int = MISSING

@configclass
class RnnCfg(RnnBaseCfg):
    module: type[RNN] = RNN

@configclass
class GruCfg(RnnBaseCfg):
    module: type[GRU] = GRU

@configclass
class LstmCfg(RnnBaseCfg):
    module: type[LSTM] = LSTM

@configclass
class RnnMlpCfg(NetworkCfg):
    module: type[RnnMlp] = RnnMlp

    rnn: type[RnnBaseCfg] = MISSING
    mlp: type[MlpCfg] = MISSING