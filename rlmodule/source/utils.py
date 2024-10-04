from typing import Sequence, Union
import torch
from torch import nn
import numpy as np

import gym
import gymnasium

# TODO(ll) double function, already defined in model_instantiators.py
def contains_rnn_module(module: nn.Module, module_types):
    for submodule in module.modules():
        if isinstance(submodule, module_types):
            return True
    return False


# TODO check for input sizes, also check if you can't just pass inputs. instead of separate thingies separated in RLModel
def get_output_size(module, input_shape):
    module.train(False)
    if isinstance(input_shape, int):
        input_shape = (input_shape,)
    dummy_input = torch.zeros(1, *input_shape)
    if contains_rnn_module(module, (nn.LSTM, nn.RNN, nn.GRU)):
        # TODO(ll) module.num_layer * bidirectional?
        dummy_hidden = torch.zeros(module.num_layers, 1, module.hidden_size)
        return module(dummy_input, None, (dummy_hidden, dummy_hidden))[0].view(-1).shape[0]
    else:
        return module(dummy_input).view(-1).shape[0]


def get_space_size(space: Union[int, Sequence[int], gym.Space, gymnasium.Space],
                   number_of_elements: bool = True) -> int:
    """Get the size (number of elements) of a space

    :param space: Space or shape from which to obtain the number of elements
    :type space: int, sequence of int, gym.Space, or gymnasium.Space
    :param number_of_elements: Whether the number of elements occupied by the space is returned (default: ``True``).
                                If ``False``, the shape of the space is returned.
                                It only affects Discrete and MultiDiscrete spaces
    :type number_of_elements: bool, optional

    :raises ValueError: If the space is not supported

    :return: Size of the space (number of elements)
    :rtype: int

    Example::

        # from int
        >>> model._get_space_size(2)
        2

        # from sequence of int
        >>> model._get_space_size([2, 3])
        6

        # Box space
        >>> space = gym.spaces.Box(low=-1, high=1, shape=(2, 3))
        >>> model._get_space_size(space)
        6

        # Discrete space
        >>> space = gym.spaces.Discrete(4)
        >>> model._get_space_size(space)
        4
        >>> model._get_space_size(space, number_of_elements=False)
        1

        # MultiDiscrete space
        >>> space = gym.spaces.MultiDiscrete([5, 3, 2])
        >>> model._get_space_size(space)
        10
        >>> model._get_space_size(space, number_of_elements=False)
        3

        # Dict space
        >>> space = gym.spaces.Dict({'a': gym.spaces.Box(low=-1, high=1, shape=(2, 3)),
        ...                          'b': gym.spaces.Discrete(4)})
        >>> model._get_space_size(space)
        10
        >>> model._get_space_size(space, number_of_elements=False)
        7
    """
    size = None
    if type(space) in [int, float]:
        size = space
    elif type(space) in [tuple, list]:
        size = np.prod(space)
    elif issubclass(type(space), gym.Space):
        if issubclass(type(space), gym.spaces.Discrete):
            if number_of_elements:
                size = space.n
            else:
                size = 1
        elif issubclass(type(space), gym.spaces.MultiDiscrete):
            if number_of_elements:
                size = np.sum(space.nvec)
            else:
                size = space.nvec.shape[0]
        elif issubclass(type(space), gym.spaces.Box):
            size = np.prod(space.shape)
        elif issubclass(type(space), gym.spaces.Dict):
            size = sum([self._get_space_size(space.spaces[key], number_of_elements) for key in space.spaces])
    elif issubclass(type(space), gymnasium.Space):
        if issubclass(type(space), gymnasium.spaces.Discrete):
            if number_of_elements:
                size = space.n
            else:
                size = 1
        elif issubclass(type(space), gymnasium.spaces.MultiDiscrete):
            if number_of_elements:
                size = np.sum(space.nvec)
            else:
                size = space.nvec.shape[0]
        elif issubclass(type(space), gymnasium.spaces.Box):
            size = np.prod(space.shape)
        elif issubclass(type(space), gymnasium.spaces.Dict):
            size = sum([self._get_space_size(space.spaces[key], number_of_elements) for key in space.spaces])
    if size is None:
        raise ValueError(f"Space type {type(space)} not supported")
    return int(size)