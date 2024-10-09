import gym
import gymnasium
import torch.nn as nn

from dataclasses import MISSING
from collections.abc import Callable
from typing import List, Sequence, Union

from rlmodule.source.network import GRU, LSTM, MLP, RNN, RnnBase, RnnMlp

# use isaac-lab native configclass if available to avoid it being declared twice
try:
    from omni.isaac.lab.utils import configclass
except ImportError:
    print("Importing local configclass.")
    from rlmodule.source.nvidia_utils import configclass

@configclass
class NetworkCfg:
    module: Union[nn.Module, Callable[..., nn.Module]] = MISSING
    input_size: Union[int, Sequence[int], gym.Space, gymnasium.Space] = -1 # -1 means value should be inferred


@configclass
class MlpCfg(NetworkCfg):
    module: type[MLP] = MLP

    hidden_units: List[int] = MISSING
    activations: List[type[nn.Module]] = MISSING
    

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
    module: type[RnnBase] = RnnMlp

    rnn: RnnBaseCfg = MISSING
    mlp: MlpCfg = MISSING