from typing import Union

import gym
import gymnasium

import torch
import torch.nn as nn
from torch.distributions import Normal

from rlmodule.source.utils import get_space_size


class OutputLayer(nn.Module):
    def __init__(self, device: Union[str, torch.device], input_size: int, cfg):
        """Base class for OutputLayer class hierarchy."""
        super().__init__()
        self.device = device

        # clip action is only supported for gym/gymnasium space 
        self._clip_actions = False

        if issubclass(type(cfg.output_size), gym.Space) or issubclass(type(cfg.output_size), gymnasium.Space):
            self._clip_actions = cfg.clip_actions
            
        if self._clip_actions:
            self._clip_actions_min = torch.tensor(cfg.output_size.low, device=self.device, dtype=torch.float32)
            self._clip_actions_max = torch.tensor(cfg.output_size.high, device=self.device, dtype=torch.float32)

        self._input_size = input_size
        self._output_size = get_space_size(cfg.output_size)

        self._output_scale = cfg.output_scale


class GaussianLayer(OutputLayer):
    def __init__(self, device: Union[str, torch.device], input_size: int, cfg):
        """Gaussian output layer

        :param device: Device on which a tensor/array is or will be allocated
        :type device: str or torch.device
        :param input_size
        :type int
        :param cfg: Configuration of Gaussian output layer
        :type OutputLayerCfg

        :raises ValueError: If the reduction method is not valid
        """
        super().__init__(device, input_size, cfg)

        self._clip_log_std = cfg.clip_log_std
        self._log_std_min = cfg.min_log_std
        self._log_std_max = cfg.max_log_std

        self._clamped_log_std = None
        self._num_samples = None
        self._distribution = None

        if cfg.reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._reduction = (
            torch.mean
            if cfg.reduction == "mean"
            else torch.sum if cfg.reduction == "sum" else torch.prod if cfg.reduction == "prod" else None
        )

        self._net = nn.Sequential(nn.Linear(self._input_size, self._output_size), cfg.output_activation())

        self._log_std_parameter = nn.Parameter(cfg.initial_log_std * torch.ones(self._output_size))

    def forward(self, input, taken_actions, outputs_dict):
        """Act stochastically in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function.
                 The third component is a dictionary containing the mean actions ``"mean_actions"``
                 and extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> actions, log_prob, outputs = model.act({"states": states})
            >>> print(actions.shape, log_prob.shape, outputs["mean_actions"].shape)
            torch.Size([4096, 8]) torch.Size([4096, 1]) torch.Size([4096, 8])
        """
        mean_actions = self._output_scale * self._net(input)  # TODO in skrl example the self._cfg.output_scale * is done here. -> why understand this (is it correct).

        log_std = self._log_std_parameter

        # clamp log standard deviations
        if self._clip_log_std:
            log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)

        self._clamped_log_std = log_std
        self._num_samples = mean_actions.shape[0]

        # distribution
        self._distribution = Normal(mean_actions, self._clamped_log_std.exp())

        # sample using the reparametrization trick
        actions = self._distribution.rsample()

        # clip actions
        if self._clip_actions:
            actions = torch.clamp(actions, min=self._clip_actions_min, max=self._clip_actions_max)

        if taken_actions is None:
            taken_actions = actions

        # log of the probability density function
        log_prob = self._distribution.log_prob(taken_actions)
        if self._reduction is not None:
            log_prob = self._reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs_dict["mean_actions"] = mean_actions
        return actions, log_prob, outputs_dict

        ####

        # TODO(ll)
        # CNN
        # 1) changing shape if it comes in linear fashion of input, check how I was doing this.
        # def compute(self, inputs, role):
        # # permute (samples, width * height * channels) -> (samples, channels, width, height)
        # return self._net(inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2)),
        # self._log_std, {}
        # 2) what with that weird Shapes?  search for taken_actions, who called it with this input.
        # How should CNN be applied to such things..just in states?

        # TODO(ll) output scale removed .. check that tanh where is
        # return output * self.instantiator_output_scale, self._log_std, {}
        # return output, self._log_std, output_dict

    def get_entropy(self, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model

        :return: Entropy of the model
        :rtype: torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> entropy = model.get_entropy()
            >>> print(entropy.shape)
            torch.Size([4096, 8])
        """
        if self._distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._distribution.entropy().to(self.device)

    def get_log_std(self, role: str = "") -> torch.Tensor:
        """Return the log standard deviation of the model

        :return: Log standard deviation of the model
        :rtype: torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> log_std = model.get_log_std()
            >>> print(log_std.shape)
            torch.Size([4096, 8])
        """
        return self._clamped_log_std.repeat(self._num_samples, 1)

    def distribution(self, role: str = "") -> torch.distributions.Normal:
        """Get the current distribution of the model

        :return: Distribution of the model
        :rtype: torch.distributions.Normal
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> distribution = model.distribution()
            >>> print(distribution)
            Normal(loc: torch.Size([4096, 8]), scale: torch.Size([4096, 8]))
        """
        return self._distribution


class DeterministicLayer(OutputLayer):
    def __init__(self, device: Union[str, torch.device], input_size: int, cfg):
        """Deterministic output layer

        :param device: Device on which a tensor/array is or will be allocated
        :type device: str or torch.device
        :param input_size
        :type int
        :param cfg: Configuration of Gaussian output layer
        :type OutputLayerCfg
        """

        super().__init__(device, input_size, cfg)

        self._net = nn.Sequential(nn.Linear(input_size, cfg.output_size), cfg.output_activation())

    def forward(self, input, taken_actions, outputs_dict):
        """Act deterministically in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is ``None``. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> actions, _, outputs = model.act({"states": states})
            >>> print(actions.shape, outputs)
            torch.Size([4096, 1]) {}
        """

        actions = self._output_scale * self._net(input)

        if self._clip_actions:
            actions = torch.clamp(actions, min=self._clip_actions_min, max=self._clip_actions_max)

        return actions, None, outputs_dict
