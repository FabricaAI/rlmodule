from typing import Any, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from .model import Model

class RLModel(Model):
    def __init__(self, 
                device: Union[str, torch.device],
                network: nn.Module,
                output_layer: nn.Module,
                ):
        #TODO(ll) description
        """Reinforcement learning model
        """

        super().__init__(device, network)
        self._output_layer = output_layer
        
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
        return self._output_layer.get_entropy(role)
    
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
        return self._output_layer.distribution(role)
        
    def act(self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "") -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        
        states = inputs["states"]
        
        if self._rnn:
            output, output_dict = self._net(states, inputs.get('terminated', None), inputs['rnn'])
        else:
            output = self._net(states)
            output_dict = {}

        return self._output_layer(output, inputs.get('taken_actions', None),  output_dict)


class SharedRLModel(Model):
    def __init__(self, 
                device: Union[str, torch.device],
                network: nn.Module,
                policy_output_layer: Optional[nn.Module],
                value_output_layer: Optional[nn.Module],
                ):
        #TODO(ll) description
        """Shared Reinforcement learning model
        """

        super().__init__(device, network)

        self.policy_output_layer = policy_output_layer
        self.value_output_layer = value_output_layer

        # caching for shared forward pass for policy and value roles
        self._cached_states = None
        self._cached_outputs = None

        
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
        return self.policy_output_layer.get_entropy(role)
    
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
        return self.policy_output_layer.distribution(role)
    
    def _have_cached_states(self, states):
        return (self._cached_states is not None 
               and torch.equal(self._cached_states, states)      
        )
        
    def act(self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "") -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        
        states = inputs["states"]

        if not self._have_cached_states(states):
            if self._rnn:
                output, output_dict = self._net(states, inputs.get('terminated', None), inputs['rnn'])
            else:
                output = self._net(states)
                output_dict = {}

            self._cached_states = states
            self._cached_output = output, output_dict

        output, output_dict = self._cached_output
        
        if role == "policy":
            return self.policy_output_layer(output, inputs.get('taken_actions', None),  output_dict)
        elif role == "value":
            return self.value_output_layer(output, inputs.get('taken_actions', None),  output_dict)
