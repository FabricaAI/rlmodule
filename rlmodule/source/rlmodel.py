from typing import Any, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

# TODO(ll): get independent of skrl
# also remove skrl from pyproject.toml
from .model import Model


# TODO(ll) consider moving this or part of it into skrl.models.torch


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
        
        print("is rnn: ", self._rnn)
        print("is lstm: ", self._lstm)

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
        
        states = inputs["states"]
        
        # TODO(ll) once shared model is design, you may rethink this to do by call some function pointer or whatever.
        # or consider inheritance GaussianModelRNN
        # TODO(ll) if one of the models is not rnn ... and PPO_RNN is called what happens?
        if self._rnn:
            output, output_dict = self.net(states, inputs.get('terminated', None), inputs['rnn'])
        else:

            output = self.net(states)
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
        # return output, self.log_std_parameter, output_dict

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
        
        print("is rnn: ", self._rnn)
        print("is lstm: ", self._lstm)

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
