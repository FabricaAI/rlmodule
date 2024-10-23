from datetime import datetime
import os
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_RNN, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import torch.nn as nn

from rlmodule.skrl.torch import build_model
from rlmodule.skrl.torch.network import MlpCfg
from rlmodule.skrl.torch.output_layer import DeterministicLayerCfg, GaussianLayerCfg
from rlmodule.skrl.torch import RLModelCfg


from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# define models (stochastic and deterministic models) using mixins
class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.action_layer(x)), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}


def get_model(env):
    """Instantiate the agent's models (function approximators)."""

    net_cfg = MlpCfg(
        input_size=env.observation_space,
        hidden_units=[400, 300],
        activation=nn.ReLU,
    )

    print(env.action_space)
    print(env.observation_space)

    policy_model = build_model(
        RLModelCfg(
            network=net_cfg,
            device=device,
            output_layer=GaussianLayerCfg(
                output_size=env.action_space,
                output_scale = 2.0,
                min_log_std=-1.2,
                max_log_std=2,
                initial_log_std=0.0,
            ),
        )
    )

    value_model_cfg = RLModelCfg(
            network=net_cfg,
            device=device,
            output_layer=DeterministicLayerCfg(),
        )
    

    models = {
        "policy": policy_model,
        "critic_1": build_model(value_model_cfg),
        "critic_2": build_model(value_model_cfg),
        "target_critic_1": build_model(value_model_cfg),
        "target_critic_2": build_model(value_model_cfg)
    }
    print(models)
    return models

# set seed for reproducibility
seed = 12245
set_seed(seed)

# load and wrap the gymnasium environment.
# note: the environment version may change depending on the gymnasium version
try:
    env = gym.make("Pendulum-v1")
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith("Pendulum-v")][0]
    print("Pendulum-v1 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=device, replacement=False)


# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
# models = get_model(env)

models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device, clip_actions=True)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

print(models)

# initialize models' parameters (weights and biases)
for model in models.values():
   model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["discount_factor"] = 0.98
cfg["batch_size"] = 100
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 1000
cfg["learn_entropy"] = True
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 75
cfg["experiment"]["checkpoint_interval"] = 750
cfg["experiment"]["directory"] = "runs/skrl/torch/"
cfg["experiment"]["experiment_name"] = (
    os.path.splitext(os.path.basename(__file__))[0] + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
)

params = {
    "models": models,
    "memory": memory,
    "cfg": cfg,
    "observation_space": env.observation_space,
    "action_space": env.action_space,
    "device": device,
}

if False: #models["policy"].is_rnn:
    agent = SAC_RNN(**params)
else:
    agent = SAC(**params)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 15000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()