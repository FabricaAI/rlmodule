from typing import Any, Mapping, Optional, Sequence, Tuple, Union


import gym
import gymnasium

import torch
import torch.nn as nn

from enum import Enum

# TODO(ll): get independent of skrl
# also remove skrl from pyproject.toml
from .model import Model, Shape
from .deprecated import DeterministicMixin, GaussianMixin  # noqa

# TODO(ll) consider moving this or part of it into skrl.models.torch
# TODO(ll) MultivariateGaussian, Categorical, Multicategorical



# TODO(ll) move to base ... and call from shared as well
def contains_rnn_module(module: nn.Module, module_types):
    for submodule in module.modules():
        if isinstance(submodule, module_types):
            return True
    return False


class RLModel(Model):
    def __init__(self, 
                device: Optional[Union[str, torch.device]] = None,
                # TODO network, if None do perceptron?
                network: Optional[nn.Module] = None,
                # TODO can be combination of various types ... mixed layer
                # but now it is dict
                output_layer: Optional[nn.Module] = None,
                ):
        #TODO(ll) description
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

        super().__init__( device)
        # GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        #TODO(ll) other input types ..gym, tuple?
        #TODO(ll) ignore output scale for now because don't like, but may reintroduce
        #TODO(ll) where is activation tanh propagated (what used to be forgotten as a bug)
        # self.instantiator_output_scale = metadata["output_scale"]

        self.net = network
        self.output_layer = output_layer

        # TODO(ll) maybe other way .. check net for lstm/rnn/gru?
        
        self._lstm = contains_rnn_module( self, nn.LSTM)
        self._rnn = contains_rnn_module( self, (nn.LSTM, nn.RNN, nn.GRU))
        
        
        print("!rnn: ", self._rnn)
        print("!lstm: ", self._lstm)


        print("!rnn: ", self._rnn)
        print("!lstm: ", self._lstm)

        # TODO(ll) one liner:
        # self.net = nn.Sequential(network, self._get_num_units_by_shape(self.output_shape), torch.nn.Tanh())

        # TODO(ll)  output_units = self._get_num_units_by_shape(self.output_shape)

        # TODO(ll) 512 (last hidden layer) is hardcoded.
        #layers = [nn.Linear(512, self._get_num_units_by_shape(self.output_shape)), torch.nn.Tanh()]
        
        # TODO(ll) propagate output activation from outside. Maybe actually propagate function, so it is easily customable and can include scaling.
        #     -- it can default to tanh here, and to identity in deterministic
        #
        # if metadata[0]["output_activation"] is not None:
        #     layers.append(_get_activation_function(metadata[0]["output_activation"]))
        
        #self.mean_net = nn.Sequential(*layers)

        # self.log_std_parameter = nn.Parameter(
        #     initial_log_std * torch.ones(self._get_num_units_by_shape(self.output_shape)))
        
    #TODO(ll) only needed for LSTM, RNN, GRU .. PPO_RNN
    #TODO(ll) consider adding it to Base model.
    # Recurrent Neural Network (RNN) specification for RNN, LSTM and GRU layers/cells
    def get_specification(self):   
        if self._lstm:
            # batch size (N) is the number of envs
            return {"rnn": {"sequence_length": self.net.sequence_length,
                            "sizes": [(self.net.num_layers, self.net.num_envs, self.net.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                      (self.net.num_layers, self.net.num_envs, self.net.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)
        elif self._rnn:
            return {"rnn": {"sequence_length": self.net.sequence_length,
                            "sizes": [(self.net.num_layers, self.net.num_envs, self.net.hidden_size)]}}    # hidden states (D ∗ num_layers, N, Hout)
        else:
            return {}
        
    # TODO(ll) RNN & GRU has this simplier specification
    # def get_specification(self):

    #     # batch size (N) is the number of envs
    #     return {"rnn": {"sequence_length": self.sequence_length,
    #                     "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)
        
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
        return self.output_layer.mixin.get_entropy(role)
    
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
        return self.output_layer.mixin.distribution(role)
        
    def act(self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "") -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        
        # TODO(ll) once shared model is design, you may rethink this to do by call some function pointer or whatever.
        # or consider inheritance GaussianModelRNN
        # TODO(ll) if one of the models is not rnn ... and PPO_RNN is called what happens?
        if self._rnn:
            output, output_dict = self.net(self._get_all_inputs(inputs), inputs.get('terminated', None), inputs['rnn'])
        else:

            output = self.net(self._get_all_inputs(inputs))
            output_dict = {}

        # TODO(ll) passing by independetly as together sequential has just one input in forward
        # potentially can be done that output dict would be propagated through mean net (mean net overload forward)
        
        return self.output_layer(output, inputs.get('taken_actions', None),  output_dict)

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

class SharedRLModel(Model):
    def __init__(self, 
                #  
                device: Optional[Union[str, torch.device]] = None,
                # TODO network, if None do pecreptron?
                network: Optional[nn.Module] = None,
                # TODO can be combination of various types ... mixed layer
                # but now it is dict
                policy_output_layer: Optional[nn.Module] = None,
                value_output_layer: Optional[nn.Module] = None,
                ):
        #TODO(ll) description
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

        super().__init__(device)
        # GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        #TODO(ll) other input types ..gym, tuple?
        #TODO(ll) ignore output scale for now because don't like, but may reintroduce
        #TODO(ll) where is activation tanh propagated (what used to be forgotten as a bug)
        # self.instantiator_output_scale = metadata["output_scale"]
        

        self.net = network
        # self.output_layer = output_layer

        self.policy_output_layer = policy_output_layer
        self.value_output_layer = value_output_layer

        self._lstm = contains_rnn_module( self, nn.LSTM)
        self._rnn = contains_rnn_module( self, (nn.LSTM, nn.RNN, nn.GRU))
        
        print("!rnn: ", self._rnn)
        print("!lstm: ", self._lstm)

        # caching for shared forward pass for policy and value roles
        self._cached_states = None
        self._cached_outputs = None
             
        # TODO(ll) one liner:
        # self.net = nn.Sequential(network, self._get_num_units_by_shape(self.output_shape), torch.nn.Tanh())

        # TODO(ll)  output_units = self._get_num_units_by_shape(self.output_shape)

        # TODO(ll) 512 (last hidden layer) is hardcoded.
        #layers = [nn.Linear(512, self._get_num_units_by_shape(self.output_shape)), torch.nn.Tanh()]
        
        # TODO(ll) propagate output activation from outside. Maybe actually propagate function, so it is easily customable and can include scaling.
        #     -- it can default to tanh here, and to identity in deterministic
        #
        # if metadata[0]["output_activation"] is not None:
        #     layers.append(_get_activation_function(metadata[0]["output_activation"]))
        
        #self.mean_net = nn.Sequential(*layers)

        # self.log_std_parameter = nn.Parameter(
        #     initial_log_std * torch.ones(self._get_num_units_by_shape(self.output_shape)))
        
    #TODO(ll) only needed for LSTM, RNN, GRU .. PPO_RNN
    #TODO(ll) consider adding it to Base model.
    # Recurrent Neural Network (RNN) specification for RNN, LSTM and GRU layers/cells
    def get_specification(self):   
        if self._lstm:
            # batch size (N) is the number of envs
            return {"rnn": {"sequence_length": self.net.sequence_length,
                            "sizes": [(self.net.num_layers, self.net.num_envs, self.net.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                      (self.net.num_layers, self.net.num_envs, self.net.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)
        elif self._rnn:
            return {"rnn": {"sequence_length": self.net.sequence_length,
                            "sizes": [(self.net.num_layers, self.net.num_envs, self.net.hidden_size)]}}    # hidden states (D ∗ num_layers, N, Hout)
        else:
            return {}
        
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
        return self.policy_output_layer.mixin.get_entropy(role)
    
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
        return self.policy_output_layer.mixin.distribution(role)
    
    def _have_cached_states(self, states):
        return (self._cached_states is not None 
               and torch.equal(self._cached_states, states)      
        )
        
    def act(self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "") -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        
        
        #states = self._get_all_inputs(inputs) # TODO(ll) rename to get_all_states
        states = inputs["states"]

        if not self._have_cached_states(states):
            # TODO(ll) if one of the models is not rnn ... and PPO_RNN is called what happens?
            if self._rnn:
                output, output_dict = self.net(states, inputs.get('terminated', None), inputs['rnn'])
            else:
                output = self.net(states)
                output_dict = {}

            self._cached_states = states
            self._cached_output = output, output_dict

        output, output_dict = self._cached_output
        
        if role == "policy":
            return self.policy_output_layer(output, inputs.get('taken_actions', None),  output_dict)
        elif role == "value":
            return self.value_output_layer(output, inputs.get('taken_actions', None),  output_dict)


        

        # TODO(ll)
        # CNN
        # 1) changing shape if it comes in linear fashion of input, check how I was doing this.
        # def compute(self, inputs, role):
        # # permute (samples, width * height * channels) -> (samples, channels, width, height)
        # return self.net(inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2)), self.log_std_parameter, {}
        # 2) what with that weird Shapes?  search for taken_actions, who called it with this input. How should CNN be applied to such things..just in states? 

        # TODO(ll) output scale removed .. check that tanh where is
        # return output * self.instantiator_output_scale, self.log_std_parameter, {}
        # return output, self.log_std_parameter, output_dict

# TODO ABC?
class OutputLayer(nn.Module):
    # TODO how to pass device, needs?
    def __init__(self, device, input_size, cfg):
        super().__init__()
        self.device = device
        self._input_size = input_size
        self._cfg = cfg


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
            cfg.output_activation
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



# class GaussianModel(GaussianMixin, Model):
#     def __init__(self, 
#                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
#                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
#                 # TODO network, if None do pecreptron?
#                 network: Optional[nn.Module] = None,
#                 device: Optional[Union[str, torch.device]] = None,
#                 clip_actions: bool = False,
#                 clip_log_std: bool = True,
#                 min_log_std: float = -20,
#                 max_log_std: float = 2,
#                 initial_log_std: float = 0,
#                 input_shape: Shape = Shape.STATES,
#                 output_shape: Shape = Shape.ACTIONS,
#                 reduction="sum",
#                 ):
#         """Gaussian model

#         :param observation_space: Observation/state space or shape (default: None).
#                                 If it is not None, the num_observations property will contain the size of that space
#         :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
#         :param action_space: Action space or shape (default: None).
#                             If it is not None, the num_actions property will contain the size of that space
#         :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
#         :param input_shape todo(ll)
#         :type input_shape
#         :param output_shape todo(ll)
#         :type output_shape
#         :param network todo(ll) description + optional?
#         :type network
#         :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
#                     If None, the device will be either ``"cuda"`` if available or ``"cpu"``
#         :type device: str or torch.device, optional
#         :param clip_actions: Flag to indicate whether the actions should be clipped (default: False)
#         :type clip_actions: bool, optional
#         :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped (default: True)
#         :type clip_log_std: bool, optional
#         :param min_log_std: Minimum value of the log standard deviation (default: -20)
#         :type min_log_std: float, optional
#         :param max_log_std: Maximum value of the log standard deviation (default: 2)
#         :type max_log_std: float, optional
#         :param initial_log_std: Initial value for the log standard deviation (default: 0)
#         :type initial_log_std: float, optional
#         :param reduction: Reduction method for returning the log probability density function: (default: ``"sum"``).
#                         Supported values are ``"mean"``, ``"sum"``, ``"prod"`` and ``"none"``. If "``none"``, the log probability density
#                         function is returned as a tensor of shape ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``
#         :type reduction: str, optional
#         """

#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

#         #TODO(ll) other input types ..gym, tuple?
#         #TODO(ll) ignore output scale for now because don't like, but may reintroduce
#         #TODO(ll) where is activation tanh propagated (what used to be forgotten as a bug)
#         # self.instantiator_output_scale = metadata["output_scale"]
        
#         self.input_shape = input_shape
#         self.output_shape = output_shape

#         self.net = network

#         # TODO(ll) maybe other way .. check net for lstm/rnn/gru?
#         self._rnn = hasattr(self.net, 'sequence_length')
             
#         # TODO(ll) one liner:
#         # self.net = nn.Sequential(network, self._get_num_units_by_shape(self.output_shape), torch.nn.Tanh())

#         # TODO(ll)  output_units = self._get_num_units_by_shape(self.output_shape)

#         # TODO(ll) 512 (last hidden layer) is hardcoded.
#         layers = [nn.Linear(512, self._get_num_units_by_shape(self.output_shape)), torch.nn.Tanh()]
        
#         # TODO(ll) propagate output activation from outside. Maybe actually propagate function, so it is easily customable and can include scaling.
#         #     -- it can default to tanh here, and to identity in deterministic
#         #
#         # if metadata[0]["output_activation"] is not None:
#         #     layers.append(_get_activation_function(metadata[0]["output_activation"]))
        
#         self.mean_net = nn.Sequential(*layers)

#         self.log_std_parameter = nn.Parameter(
#             initial_log_std * torch.ones(self._get_num_units_by_shape(self.output_shape)))
        
#     #TODO(ll) only needed for LSTM, RNN, GRU .. PPO_RNN
#     #TODO(ll) consider adding it to Base model.
#     # Recurrent Neural Network (RNN) specification for RNN, LSTM and GRU layers/cells
#     def get_specification(self):   
#         if self._rnn:
#             # batch size (N) is the number of envs
#             return {"rnn": {"sequence_length": self.net.sequence_length,
#                             "sizes": [(self.net.num_layers, self.net.num_envs, self.net.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
#                                       (self.net.num_layers, self.net.num_envs, self.net.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)
#         else:
#             return {}
        
#     def compute(self, inputs, role=""):
        
#         # TODO(ll) once shared model is design, you may rethink this to do by call some function pointer or whatever.
#         # or consider inheritance GaussianModelRNN
#         # TODO(ll) if one of the models is not rnn ... and PPO_RNN is called what happens?
#         if self._rnn:
#             output, output_dict = self.net(self._get_all_inputs(inputs), inputs.get('terminated', None), inputs['rnn'])
#         else:
#             output = self.net(self._get_all_inputs(inputs))
#             output_dict = {}

#         # TODO(ll) passing by independetly as together sequential has just one input in forward
#         # potentially can be done that output dict would be propagated through mean net (mean net overload forward)
#         output = self.mean_net(output)

#         # TODO(ll)
#         # CNN
#         # 1) changing shape if it comes in linear fashion of input, check how I was doing this.
#         # def compute(self, inputs, role):
#         # # permute (samples, width * height * channels) -> (samples, channels, width, height)
#         # return self.net(inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2)), self.log_std_parameter, {}
#         # 2) what with that weird Shapes?  search for taken_actions, who called it with this input. How should CNN be applied to such things..just in states? 

#         # TODO(ll) output scale removed .. check that tanh where is
#         # return output * self.instantiator_output_scale, self.log_std_parameter, {}
#         return output, self.log_std_parameter, output_dict

    
# class DeterministicModel(DeterministicMixin, Model):
#     def __init__(self, 
#                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
#                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
#                 network: Optional[nn.Module] = None,
#                 device: Optional[Union[str, torch.device]] = None,
#                 clip_actions: bool = False,
#                 input_shape: Shape = Shape.STATES,
#                 output_shape: Shape = Shape.ONE,
#                 ):
#         """Deterministic model

#         TODO(ll) update doc string
#         :param observation_space: Observation/state space or shape (default: None).
#                                 If it is not None, the num_observations property will contain the size of that space
#         :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
#         :param action_space: Action space or shape (default: None).
#                             If it is not None, the num_actions property will contain the size of that space
#         :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
#         :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
#                     If None, the device will be either ``"cuda"`` if available or ``"cpu"``
#         :type device: str or torch.device, optional
#         :param clip_actions: Flag to indicate whether the actions should be clipped to the action space (default: False)
#         :type clip_actions: bool, optional
#         :param input_shape: Shape of the input (default: Shape.STATES)
#         :type input_shape: Shape, optional
#         :param hiddens: Number of hidden units in each hidden layer
#         :type hiddens: int or list of ints
#         :param hidden_activation: Activation function for each hidden layer (default: "relu").
#         :type hidden_activation: list of strings
#         :param output_shape: Shape of the output (default: Shape.ACTIONS)
#         :type output_shape: Shape, optional
#         :param output_activation: Activation function for the output layer (default: "tanh")
#         :type output_activation: str or None, optional
#         :param output_scale: Scale of the output layer (default: 1.0).
#                             If None, the output layer will not be scaled
#         :type output_scale: float, optional

#         :return: Deterministic model instance
#         :rtype: Model
#         """

#         Model.__init__(self, observation_space, action_space, device)
#         DeterministicMixin.__init__(self, clip_actions)

#         self.input_shape = input_shape
#         self.output_shape = output_shape

#         self.net = network
#         self._rnn = hasattr(self.net, 'sequence_length')

#         # TODO(ll) change identity by custom fction.

#         # TODO(ll) 512 is hardcoded.
#         self.deterministic_layer = nn.Sequential(
#             nn.Linear(512, self._get_num_units_by_shape(self.output_shape)), torch.nn.Identity())

#     def get_specification(self):   
#         if self._rnn:
#             # batch size (N) is the number of envs
#             return {"rnn": {"sequence_length": self.net.sequence_length,
#                             "sizes": [(self.net.num_layers, self.net.num_envs, self.net.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
#                                       (self.net.num_layers, self.net.num_envs, self.net.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)
#         else:
#             return {}

#     def compute(self, inputs, role=""):
#         # TODO(ll) this block can be maybe moved into base.py if it is the same for all (where get_all_inputs live)
#         if self._rnn:
#             output, output_dict = self.net(self._get_all_inputs(inputs), inputs.get('terminated', None), inputs['rnn'])
#         else:
#             output = self.net(self._get_all_inputs(inputs))
#             output_dict = {}
        
#         output = self.deterministic_layer(output)

#         # TODO(ll) output scale removed
#         # return output * self.instantiator_output_scale, {}
#         return output, output_dict
