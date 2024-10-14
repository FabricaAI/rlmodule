
import torch
import torch.nn as nn

import gym
import gymnasium

from torch.distributions import Normal

from rlmodule.source.utils import get_space_size

from typing import TYPE_CHECKING

# TODO type checking not working
if TYPE_CHECKING:
    from rlmodule.source.output_layer_cfg import OutputLayerCfg, GaussianLayerCfg, DeterministicLayerCfg

class OutputLayer(nn.Module):
    def __init__(self, device, input_size, cfg):
        super().__init__()
        self.device = device

        self._clip_actions = cfg.clip_actions and (issubclass(type(cfg.output_size), gym.Space) or \
            issubclass(type(cfg.output_size), gymnasium.Space))

        if self._clip_actions:  
            self._clip_actions_min = torch.tensor(cfg.output_size.low, device=self.device, dtype=torch.float32)
            self._clip_actions_max = torch.tensor(cfg.output_size.high, device=self.device, dtype=torch.float32)
        
        self._input_size = input_size
        self._output_size = get_space_size(cfg.output_size)


class GaussianLayer(OutputLayer):
    def __init__(self, 
                device,
                input_size,
                cfg
                ):
        """Gaussian model

        :param observation_space: Observation/state space or shape (default: None).
                                If it is not None, the num_observations property will contain the size of that space
        :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: None).
                            If it is not None, the num_actions property will contain the size of that space
        :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
        :param input_shape todo(ll)
        :type input_shape
        :param output_shape todo(ll)
        :type output_shape
        :param network todo(ll) description + optional?
        :type network
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                    If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param clip_actions: Flag to indicate whether the actions should be clipped (default: False)
        :type clip_actions: bool, optional
        :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped (default: True)
        :type clip_log_std: bool, optional
        :param min_log_std: Minimum value of the log standard deviation (default: -20)
        :type min_log_std: float, optional
        :param max_log_std: Maximum value of the log standard deviation (default: 2)
        :type max_log_std: float, optional
        :param initial_log_std: Initial value for the log standard deviation (default: 0)
        :type initial_log_std: float, optional
        :param reduction: Reduction method for returning the log probability density function: (default: ``"sum"``).
                        Supported values are ``"mean"``, ``"sum"``, ``"prod"`` and ``"none"``. If "``none"``, the log probability density
                        function is returned as a tensor of shape ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``
        :type reduction: str, optional
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
        self._reduction = torch.mean if cfg.reduction == "mean" else torch.sum if cfg.reduction == "sum" \
            else torch.prod if cfg.reduction == "prod" else None

        self._net = nn.Sequential(
            nn.Linear(self._input_size,  self._output_size), 
            cfg.output_activation()
        )

        self._log_std_parameter = nn.Parameter(
            cfg.initial_log_std * torch.ones( self._output_size )
        )
        
        
    def forward(self, input, taken_actions, outputs_dict):

        
        mean_actions = self._net(input)
        
        log_std = self._log_std_parameter
        
        # clamp log standard deviations
        if self._clip_log_std:
            log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)

        self._clamped_log_std = log_std
        
        #self._clamped_log_std = log_std
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
        # return self._net(inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2)), self._log_std, {}
        # 2) what with that weird Shapes?  search for taken_actions, who called it with this input. How should CNN be applied to such things..just in states? 

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
    def __init__(self, 
                device,
                input_size,
                cfg
                ):
        """Deterministic model

        TODO(ll) update doc string
        :param observation_space: Observation/state space or shape (default: None).
                                If it is not None, the num_observations property will contain the size of that space
        :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: None).
                            If it is not None, the num_actions property will contain the size of that space
        :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                    If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space (default: False)
        :type clip_actions: bool, optional
        :param input_shape: Shape of the input (default: Shape.STATES)
        :type input_shape: Shape, optional
        :param hiddens: Number of hidden units in each hidden layer
        :type hiddens: int or list of ints
        :param hidden_activation: Activation function for each hidden layer (default: "relu").
        :type hidden_activation: list of strings
        :param output_shape: Shape of the output (default: Shape.ACTIONS)
        :type output_shape: Shape, optional
        :param output_activation: Activation function for the output layer (default: "tanh")
        :type output_activation: str or None, optional
        :param output_scale: Scale of the output layer (default: 1.0).
                            If None, the output layer will not be scaled
        :type output_scale: float, optional

        :return: Deterministic model instance
        :rtype: Model
        """

        super().__init__(device, input_size, cfg)

        self._net = nn.Sequential(
            nn.Linear(input_size, cfg.output_size), 
            cfg.output_activation()
        )

    def forward(self, input, taken_actions, outputs_dict):

        actions = self._net(input)

        if self._clip_actions:
            actions = torch.clamp(actions, min=self._clip_actions_min, max=self._clip_actions_max)
        
        return actions, None, outputs_dict
