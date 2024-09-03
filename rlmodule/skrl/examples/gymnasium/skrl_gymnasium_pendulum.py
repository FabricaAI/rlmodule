import gymnasium as gym

import torch
import torch.nn as nn

from rlmodule.source.rlmodel import RLModel, SharedRLModel, GaussianLayer, DeterministicLayer
from rlmodule.source.modules import MLP

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

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

def get_shared_model(env):
    # instantiate the agent's models (function approximators).
    params = {'input_size': env.observation_space.shape[0], 'hidden_units': [64, 64, 64], 'activations': ['relu', 'relu', 'relu']}
    net = MLP(params)
    
    model = SharedRLModel(
        observation_space=env.observation_space.shape[0], # TODO not do shape[0] but process box normally
        action_space=env.observation_space.shape[0],
        device=device,
        network=net,
        output_layer={
            'policy': GaussianLayer(
                input_size=64,  # TODO
                output_size=env.action_space.shape[0],
                device=device,
                output_activation=nn.Tanh(),
                clip_actions=False,
                clip_log_std=True,
                min_log_std=-1.2,
                max_log_std=2,
                initial_log_std=0.2,
            ),
            'value': DeterministicLayer(
                input_size=64,
                output_size=1,
                device=device,
                output_activation=nn.Identity(),
            ),
        },
    ).to(device) # TODO careful about device

    return {'policy': model, 'value': model}


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