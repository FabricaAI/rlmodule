
import torch
import torch.nn as nn

from rlmodule.source.utils import get_space_size
from .deprecated import DeterministicMixin, GaussianMixin  # noqa

from typing import TYPE_CHECKING

# TODO type checking not working
if TYPE_CHECKING:
    from rlmodule.source.output_layer_cfg import OutputLayerCfg, GaussianLayerCfg, DeterministicLayerCfg

# TODO(ll) MultivariateGaussian, Categorical, Multicategorical

# TODO ABC?
class OutputLayer(nn.Module):
    # TODO does it need device?
    def __init__(self, device, input_size, cfg):
        super().__init__()
        self.device = device
        
        cfg.output_size = get_space_size(cfg.output_size)


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

        # TODO pass just config, or get rid of mixin all together
        self.mixin = GaussianMixin(cfg.clip_actions, cfg.clip_log_std, cfg.min_log_std, cfg.max_log_std, cfg.reduction)
        
        self.net = nn.Sequential(
            nn.Linear(input_size,  cfg.output_size), 
            cfg.output_activation()
        )

        self.log_std_parameter = nn.Parameter(
            cfg.initial_log_std * torch.ones( cfg.output_size )
        )
        
        
    def forward(self, input, taken_actions, outputs_dict):

        # TODO(ll) passing by independetly as together sequential has just one input in forward
        # potentially can be done that output dict would be propagated through mean net (mean net overload forward)
        mean_actions = self.net(input)
        
        #TODO
        return self.mixin.forward(mean_actions, taken_actions, self.log_std_parameter, outputs_dict)

        # TODO(ll)
        # CNN
        # 1) changing shape if it comes in linear fashion of input, check how I was doing this.
        # def compute(self, inputs, role):
        # # permute (samples, width * height * channels) -> (samples, channels, width, height)
        # return self.net(inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2)), self.log_std_parameter, {}
        # 2) what with that weird Shapes?  search for taken_actions, who called it with this input. How should CNN be applied to such things..just in states? 

        # TODO(ll) output scale removed .. check that tanh where is
        # return output * self.instantiator_output_scale, self.log_std_parameter, {}
        return output, self.log_std_parameter, output_dict


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
        self.mixin = DeterministicMixin(cfg.clip_actions)

        self.net = nn.Sequential(
            nn.Linear(input_size, cfg.output_size), 
            cfg.output_activation
        )

    def forward(self, input, taken_actions, outputs_dict):

        output = self.net(input)
        return self.mixin.forward(output, outputs_dict)
