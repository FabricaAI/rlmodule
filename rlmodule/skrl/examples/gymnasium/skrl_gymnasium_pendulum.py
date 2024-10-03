import gymnasium as gym

import torch
import torch.nn as nn

from rlmodule.source.rlmodel import SharedRLModel, RLModel, GaussianLayer, DeterministicLayer
from rlmodule.source.modules import MLP, RNN, GRU, LSTM, RnnMlp, RnnBase

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_RNN, PPO_DEFAULT_CONFIG

from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper
#from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.envs.wrappers.torch.gymnasium_envs import GymnasiumWrapper

# todo set seed
# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

# BEGIN LIB CODE
# TODO: move it from example :)
from dataclasses import dataclass, MISSING
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
from rlmodule.source.modules import get_output_size

# use native configclass if available to avoid it being declared twice
try:
    from omni.isaac.lab.utils import configclass
except ImportError:
    print("Importing local configclass.")
    from rlmodule.source.nvidia_utils import configclass

# Net
from dataclasses import dataclass, MISSING, field
from collections.abc import Callable
from typing import List

#TODO next
# (Done) check how to do own @configclass .. or use one from isaac by default (print something to know version)
# (Done) RNN code modify to use data class
# RnnMLP
# Custom function Network
# Move things logically, annotate cfgs in modules
# Handle input shapes better?
# Create examples - mlp, rnn,gru,lstm, Lstmmlp, shared - separate , custom net by fcion
# WRITE README tutorial
# Launch new version to pip

# Import new version in Isaac-lab


@configclass
class NetworkCfg:
    module: Union[nn.Module, Callable[..., nn.Module]] = MISSING
    input_size: int = MISSING # change type
    

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
    
# net end

@configclass
class OutputLayerCfg:
    class_type: type[nn.Module] = MISSING,
    output_size: int = MISSING,  # TODO not int but something more clever
    output_activation: nn.Module = MISSING,

    clip_actions: bool = False,  # TODO what is clip action doing

@configclass
class GaussianLayerCfg(OutputLayerCfg):
    class_type: type[GaussianLayer] = GaussianLayer
    output_activation: nn.Module = nn.Tanh()

    clip_log_std: bool = True
    min_log_std: float = -20
    max_log_std: float = 2
    initial_log_std: float = 0
    reduction: str = "sum"

@configclass
class DeterministicLayerCfg(OutputLayerCfg):
    class_type: type[DeterministicLayer] = DeterministicLayer
    output_size: int = 1
    output_activation: nn.Module = nn.Identity()

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
    class_type: type[SharedRLModel] = SharedRLModel
    policy_output_layer: type[OutputLayerCfg] = MISSING
    value_output_layer: type[OutputLayerCfg] = MISSING
    
def build_model(cfg: BaseRLCfg):

    #1) network = create # for now it is created, todo from config
    
    #2) get output size - what type?  ..what is Model init doing with action/observation space

    net = cfg.network.module(cfg.network)
    
    network_output_size = get_output_size(net, cfg.network.input_size)

    def build_output_layer(layer_cfg):
        return layer_cfg.class_type( device = cfg.device, input_size = network_output_size, cfg = layer_cfg)
    
    if isinstance(cfg, RLModelCfg):
        rl_model = cfg.class_type( device = cfg.device,
                        network = net,
                        output_layer = build_output_layer( cfg.output_layer)
                     )
    elif isinstance(cfg, SharedRLModelCfg):
        rl_model = cfg.class_type( device = cfg.device,
                        network = net,
                        policy_output_layer = build_output_layer( cfg.policy_output_layer ),
                        value_output_layer = build_output_layer( cfg.value_output_layer ),
                     )
    else:
        raise TypeError(
                f" Received unsupported class: '{type(cfg)}'."
            )
    return rl_model


# END LIB CODE

def get_shared_model(env):
    # instantiate the agent's models (function approximators).

    # net_cfg = MlpCfg(
    #     input_size = env.observation_space.shape[0],
    #     hidden_units = [64, 64, 64],
    #     activations = [nn.ReLU(), nn.ReLU(), nn.ReLU()],
    # )

    # net_cfg = LstmCfg(
    #     input_size = env.observation_space.shape[0],
    #     num_envs = env.num_envs,
    #     num_layers = 1,
    #     hidden_size = 32,
    #     sequence_length = 16,
    # )

    # TODO input sizes 
    net_cfg = RnnMlpCfg(
        input_size = env.observation_space.shape[0],
        rnn = LstmCfg(
            input_size = env.observation_space.shape[0],
            num_envs = env.num_envs,
            num_layers = 1,
            hidden_size = 32,
            sequence_length = 16,
        ),
        mlp = MlpCfg(
            input_size = -1, # inferred
            hidden_units = [64, 64, 64],
            activations = [nn.ReLU(), nn.ReLU(), nn.ReLU()],
        ),
    )

    # variant IV - all config
    # net also config (optional)?
    model = build_model( 
        SharedRLModelCfg(
            network= net_cfg,
            device= device,
            policy_output_layer= GaussianLayerCfg(
                    output_size=env.action_space.shape[0],
                    min_log_std=-1.2,
                    max_log_std=2,
                    initial_log_std=0.2,
                ),
            value_output_layer= DeterministicLayerCfg(),
        )
    )
    # RLCfg(
    #     Model
    #     AgentCfg
    #     MemoryCfg
    # )

    # build_model(cfg.model)

    # cfg.agent.algoclass(
    #     cfg.agent.algo_config
    #     cfg.memory
    # )

    # SharedRLModelCfg(RLModelCfg){
    #     class = SharedRLModel

    # }

    # net
    # output layers.class()

    # cfg.model.class(  net, output_layers )

    #configclass -> dataclass - easy (Ron said)


    # variant V:
    # ideas?


    return {'policy': model, 'value': model}

def get_separate_model(env):
    # instantiate the agent's models (function approximators).
    params = {'input_size': env.observation_space.shape[0], 
              'hidden_units': [64, 64, 64], 
              'activations': ['relu', 'relu', 'relu']
              }
    net = MLP(params)

    policy_model = RLModel(
        network=net,
        device= device,
        output_layer = GaussianLayer(
                input_size=64,  # TODO
                output_size=env.action_space.shape[0],
                output_activation=nn.Tanh(),
                clip_actions=False,
                clip_log_std=True,
                min_log_std=-1.2,
                max_log_std=2,
                initial_log_std=0.2,
            ),
    )

    value_model = RLModel(
        network=net,
        device= device,
        output_layer = DeterministicLayer(
                input_size=64,
                output_size=1,
                output_activation=nn.Identity(),
            ),
    )

    return {'policy': policy_model, 'value': value_model}


# load and wrap the gymnasium environment.
# note: the environment version may change depending on the gymnasium version
try:
    env = gym.vector.make("Pendulum-v1", num_envs=4, asynchronous=False)
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith("Pendulum-v")][0]
    print("Pendulum-v1 not found. Trying {}".format(env_id))
    env = gym.vector.make(env_id, num_envs=4, asynchronous=False)

env = GymnasiumWrapper(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)

models = get_shared_model(env)

print(models)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1024  # memory_size
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 32
cfg["discount_factor"] = 0.9
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = False
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 0.5
cfg["kl_threshold"] = 0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 500
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/torch/Pendulum"

# TODO(nicer)
if models['policy']._rnn:
    agent = PPO_RNN(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)
else:
    agent = PPO(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()