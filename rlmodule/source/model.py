from typing import Any, Mapping, Optional, Tuple, Union

import collections
import gym
import gymnasium

import torch
import torch.nn as nn

from rlmodule import logger
from rlmodule.source.utils import contains_rnn_module


class Model(torch.nn.Module):
    def __init__(self, device: Union[str, torch.device], network: nn.Module) -> None:
        """Base class representing a function approximator"""
        super().__init__()

        # TODO check if this is done outside. also it is not optional rn.
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )
        self._net = network

        self._lstm = contains_rnn_module(self, nn.LSTM)
        self._rnn = contains_rnn_module(self, (nn.LSTM, nn.RNN, nn.GRU))

        logger.info(f"model contains rnn:  {self._rnn}")
        logger.info(f"model contains lstm: {self._lstm}")

        self._random_distribution = None

    @property
    def is_rnn(self):
        """Return true if there is a submodule with RNN architecture

        Submodules for which it will return true: nn.RNN, nn.GRU, nn.LSTM
        """
        return self._rnn

    @property
    def is_lstm(self):
        """Return true if there is an LSTM submodule"""
        return self._lstm

    
    def _get_policy_output_layer(self):
        if hasattr(self, '_policy_output_layer'):
            return self._policy_output_layer  # SharedModel
        else:
            return self._output_layer  # SeparatedModels
    
    
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
        return self._get_policy_output_layer().get_entropy(role)

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
        return self._get_policy_output_layer().distribution(role)

    def get_specification(self) -> Mapping[str, Any]:
        """Returns the specification of the model

        The following keys are used by the agents for initialization:

        - ``"rnn"``: Recurrent Neural Network (RNN) specification for RNN, LSTM and GRU layers/cells

          - ``"sizes"``: List of RNN shapes (number of layers, number of environments,
            number of features in the RNN state).
            There must be as many tuples as there are states in the recurrent layer/cell.
            E.g., LSTM has 2 states (hidden and cell).

        :return: Dictionary containing advanced specification of the model
        :rtype: dict

        Example::

            # model with a LSTM layer.
            # - number of layers: 1
            # - number of environments: 4
            # - number of features in the RNN state: 64
            >>> model.get_specification()
            {'rnn': {'sizes': [(1, 4, 64), (1, 4, 64)]}}
        """
        if self._lstm:
            return {
                "rnn": {
                    "sequence_length": self._net.sequence_length,
                    "sizes": [
                        (
                            self._net.num_layers,
                            self._net.num_envs,
                            self._net.hidden_size,
                        ),  # hidden states (D ∗ num_layers, N, Hout)
                        (self._net.num_layers, self._net.num_envs, self._net.hidden_size),
                    ],
                }
            }  # cell states   (D ∗ num_layers, N, Hcell)
        elif self._rnn:
            return {
                "rnn": {
                    "sequence_length": self._net.sequence_length,
                    "sizes": [(self._net.num_layers, self._net.num_envs, self._net.hidden_size)],
                }
            }  # hidden states (D ∗ num_layers, N, Hout)
        else:
            return {}

    def tensor_to_space(
        self, tensor: torch.Tensor, space: Union[gym.Space, gymnasium.Space], start: int = 0
    ) -> Union[torch.Tensor, dict]:
        """Map a flat tensor to a Gym/Gymnasium space

        The mapping is done in the following way:

        - Tensors belonging to Discrete spaces are returned without modification
        - Tensors belonging to Box spaces are reshaped to the corresponding space shape
          keeping the first dimension (number of samples) as they are
        - Tensors belonging to Dict spaces are mapped into a dictionary with the same keys as the original space

        :param tensor: Tensor to map from
        :type tensor: torch.Tensor
        :param space: Space to map the tensor to
        :type space: gym.Space or gymnasium.Space
        :param start: Index of the first element of the tensor to map (default: ``0``)
        :type start: int, optional

        :raises ValueError: If the space is not supported

        :return: Mapped tensor or dictionary
        :rtype: torch.Tensor or dict

        Example::

            >>> space = gym.spaces.Dict({'a': gym.spaces.Box(low=-1, high=1, shape=(2, 3)),
            ...                          'b': gym.spaces.Discrete(4)})
            >>> tensor = torch.tensor([[-0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 2]])
            >>>
            >>> model.tensor_to_space(tensor, space)
            {'a': tensor([[[-0.3000, -0.2000, -0.1000],
                           [ 0.1000,  0.2000,  0.3000]]]),
             'b': tensor([[2.]])}
        """
        if issubclass(type(space), gym.Space):
            if issubclass(type(space), gym.spaces.Discrete):
                return tensor
            elif issubclass(type(space), gym.spaces.Box):
                return tensor.view(tensor.shape[0], *space.shape)
            elif issubclass(type(space), gym.spaces.Dict):
                output = {}
                for k in sorted(space.keys()):
                    end = start + self._get_space_size(space[k], number_of_elements=False)
                    output[k] = self.tensor_to_space(tensor[:, start:end], space[k], end)
                    start = end
                return output
        else:
            if issubclass(type(space), gymnasium.spaces.Discrete):
                return tensor
            elif issubclass(type(space), gymnasium.spaces.Box):
                return tensor.view(tensor.shape[0], *space.shape)
            elif issubclass(type(space), gymnasium.spaces.Dict):
                output = {}
                for k in sorted(space.keys()):
                    end = start + self._get_space_size(space[k], number_of_elements=False)
                    output[k] = self.tensor_to_space(tensor[:, start:end], space[k], end)
                    start = end
                return output
        raise ValueError(f"Space type {type(space)} not supported")
    
    def random_act(self,
                   inputs: Mapping[str, Union[torch.Tensor, Any]],
                   role: str = "") -> Tuple[torch.Tensor, None, Mapping[str, Union[torch.Tensor, Any]]]:
        """Act randomly according to the action space

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises NotImplementedError: Unsupported action space

        :return: Model output. The first component is the action to be taken by the agent
        :rtype: tuple of torch.Tensor, None, and dict
        """
        # discrete action space (Discrete)
        if issubclass(type(self.action_space), gym.spaces.Discrete) or issubclass(type(self.action_space), gymnasium.spaces.Discrete):
            return torch.randint(self.action_space.n, (inputs["states"].shape[0], 1), device=self.device), None, {}
        # continuous action space (Box)
        elif issubclass(type(self.action_space), gym.spaces.Box) or issubclass(type(self.action_space), gymnasium.spaces.Box):
            if self._random_distribution is None:
                self._random_distribution = torch.distributions.uniform.Uniform(
                    low=torch.tensor(self.action_space.low[0], device=self.device, dtype=torch.float32),
                    high=torch.tensor(self.action_space.high[0], device=self.device, dtype=torch.float32))

            return self._random_distribution.sample(sample_shape=(inputs["states"].shape[0], self.num_actions)), None, {}
        else:
            raise NotImplementedError(f"Action space type ({type(self.action_space)}) not supported")


    def init_parameters(self, method_name: str = "normal_", *args, **kwargs) -> None:
        """Initialize the model parameters according to the specified method name

        Method names are from the `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ module.
        Allowed method names are *uniform_*, *normal_*, *constant_*, etc.

        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ method name
            (default: ``"normal_"``)
        :type method_name: str, optional
        :param args: Positional arguments of the method to be called
        :type args: tuple, optional
        :param kwargs: Key-value arguments of the method to be called
        :type kwargs: dict, optional

        Example::

            # initialize all parameters with an orthogonal distribution with a gain of 0.5
            >>> model.init_parameters("orthogonal_", gain=0.5)

            # initialize all parameters as a sparse matrix with a sparsity of 0.1
            >>> model.init_parameters("sparse_", sparsity=0.1)
        """
        for parameters in self.parameters():
            exec(f"torch.nn.init.{method_name}(parameters, *args, **kwargs)")

    def init_weights(self, method_name: str = "orthogonal_", *args, **kwargs) -> None:
        """Initialize the model weights according to the specified method name

        Method names are from the `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ module.
        Allowed method names are *uniform_*, *normal_*, *constant_*, etc.

        The following layers will be initialized:
        - torch.nn.Linear

        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ method name
            (default: ``"orthogonal_"``)
        :type method_name: str, optional
        :param args: Positional arguments of the method to be called
        :type args: tuple, optional
        :param kwargs: Key-value arguments of the method to be called
        :type kwargs: dict, optional

        Example::

            # initialize all weights with uniform distribution in range [-0.1, 0.1]
            >>> model.init_weights(method_name="uniform_", a=-0.1, b=0.1)

            # initialize all weights with normal distribution with mean 0 and standard deviation 0.25
            >>> model.init_weights(method_name="normal_", mean=0.0, std=0.25)
        """

        def _update_weights(module, method_name, args, kwargs):
            for layer in module:
                if isinstance(layer, torch.nn.Sequential):
                    _update_weights(layer, method_name, args, kwargs)
                elif isinstance(layer, torch.nn.Linear):
                    exec(f"torch.nn.init.{method_name}(layer.weight, *args, **kwargs)")

        _update_weights(self.children(), method_name, args, kwargs)

    def init_biases(self, method_name: str = "constant_", *args, **kwargs) -> None:
        """Initialize the model biases according to the specified method name

        Method names are from the `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_ module.
        Allowed method names are *uniform_*, *normal_*, *constant_*, etc.

        The following layers will be initialized:
        - torch.nn.Linear

        :param method_name: `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`_
            method name (default: ``"constant_"``)
        :type method_name: str, optional
        :param args: Positional arguments of the method to be called
        :type args: tuple, optional
        :param kwargs: Key-value arguments of the method to be called
        :type kwargs: dict, optional

        Example::

            # initialize all biases with a constant value (0)
            >>> model.init_biases(method_name="constant_", val=0)

            # initialize all biases with normal distribution with mean 0 and standard deviation 0.25
            >>> model.init_biases(method_name="normal_", mean=0.0, std=0.25)
        """

        def _update_biases(module, method_name, args, kwargs):
            for layer in module:
                if isinstance(layer, torch.nn.Sequential):
                    _update_biases(layer, method_name, args, kwargs)
                elif isinstance(layer, torch.nn.Linear):
                    exec(f"torch.nn.init.{method_name}(layer.bias, *args, **kwargs)")

        _update_biases(self.children(), method_name, args, kwargs)

    def forward(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Forward pass of the model

        Implementation of this method manages calling forward passes of all its components.

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function for stochastic models
                 or None for deterministic models. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict
        """
        raise NotImplementedError("The action to be taken by the agent (.forward()) is not implemented")

    def act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """This method calls the ``.forward()`` method and returns its outputs

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function for stochastic models
                 or None for deterministic models. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict
        """
        return self.forward(inputs, role)

    def set_mode(self, mode: str) -> None:
        """Set the model mode (training or evaluation)

        :param mode: Mode: ``"train"`` for training or ``"eval"`` for evaluation.
            See `torch.nn.Module.train
            <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train>`_
        :type mode: str

        :raises ValueError: If the mode is not ``"train"`` or ``"eval"``
        """
        if mode == "train":
            self.train(True)
        elif mode == "eval":
            self.train(False)
        else:
            raise ValueError("Invalid mode. Use 'train' for training or 'eval' for evaluation")

    def save(self, path: str, state_dict: Optional[dict] = None) -> None:
        """Save the model to the specified path

        :param path: Path to save the model to
        :type path: str
        :param state_dict: State dictionary to save (default: ``None``).
                           If None, the model's state_dict will be saved
        :type state_dict: dict, optional

        Example::

            # save the current model to the specified path
            >>> model.save("/tmp/model.pt")

            # save an older version of the model to the specified path
            >>> old_state_dict = copy.deepcopy(model.state_dict())
            >>> # ...
            >>> model.save("/tmp/model.pt", old_state_dict)
        """
        torch.save(self.state_dict() if state_dict is None else state_dict, path)

    def load(self, path: str) -> None:
        """Load the model from the specified path

        The final storage device is determined by the constructor of the model

        :param path: Path to load the model from
        :type path: str

        Example::

            # load the model onto the CPU
            >>> model = Model(observation_space, action_space, device="cpu")
            >>> model.load("model.pt")

            # load the model onto the GPU 1
            >>> model = Model(observation_space, action_space, device="cuda:1")
            >>> model.load("model.pt")
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()

    def migrate(
        self,
        state_dict: Optional[Mapping[str, torch.Tensor]] = None,
        path: Optional[str] = None,
        name_map: Mapping[str, str] = {},
        auto_mapping: bool = True,
        verbose: bool = False,
    ) -> bool:
        """Migrate the specified external model's state dict to the current model

        The final storage device is determined by the constructor of the model

        Only one of ``state_dict`` or ``path`` can be specified.
        The ``path`` parameter allows automatic loading the ``state_dict`` only from files generated
        by the *rl_games* and *stable-baselines3* libraries at the moment

        For ambiguous models (where 2 or more parameters, for source or current model, have equal shape)
        it is necessary to define the ``name_map``, at least for those parameters, to perform the migration successfully

        :param state_dict: External model's state dict to migrate from (default: ``None``)
        :type state_dict: Mapping[str, torch.Tensor], optional
        :param path: Path to the external checkpoint to migrate from (default: ``None``)
        :type path: str, optional
        :param name_map: Name map to use for the migration (default: ``{}``).
                         Keys are the current parameter names and values are the external parameter names
        :type name_map: Mapping[str, str], optional
        :param auto_mapping: Automatically map the external state dict to the current state dict (default: ``True``)
        :type auto_mapping: bool, optional
        :param verbose: Show model names and migration (default: ``False``)
        :type verbose: bool, optional

        :raises ValueError: If neither or both of ``state_dict`` and ``path`` parameters have been set
        :raises ValueError: If the correct file type cannot be identified from the ``path`` parameter

        :return: True if the migration was successful, False otherwise.
                 Migration is successful if all parameters of the current model are found in the external model
        :rtype: bool

        Example::

            # migrate a rl_games checkpoint with unambiguous state_dict
            >>> model.migrate(path="./runs/Ant/nn/Ant.pth")
            True

            # migrate a rl_games checkpoint with ambiguous state_dict
            >>> model.migrate(path="./runs/Cartpole/nn/Cartpole.pth", verbose=False)
            [rlmodule:WARNING] Ambiguous match for log_std_parameter <- [value_mean_std.running_mean,
                               value_mean_std.running_var, a2c_network.sigma]
            [rlmodule:WARNING] Ambiguous match for net.0.bias <- [a2c_network.actor_mlp.0.bias,
                               a2c_network.actor_mlp.2.bias]
            [rlmodule:WARNING] Ambiguous match for net.2.bias <- [a2c_network.actor_mlp.0.bias,
                               a2c_network.actor_mlp.2.bias]
            [rlmodule:WARNING] Ambiguous match for net.4.weight <- [a2c_network.value.weight,
                               a2c_network.mu.weight]
            [rlmodule:WARNING] Ambiguous match for net.4.bias <- [a2c_network.value.bias, a2c_network.mu.bias]
            [rlmodule:WARNING] Multiple use of a2c_network.actor_mlp.0.bias -> [net.0.bias, net.2.bias]
            [rlmodule:WARNING] Multiple use of a2c_network.actor_mlp.2.bias -> [net.0.bias, net.2.bias]
            False
            >>> name_map = {"log_std_parameter": "a2c_network.sigma",
            ...             "net.0.bias": "a2c_network.actor_mlp.0.bias",
            ...             "net.2.bias": "a2c_network.actor_mlp.2.bias",
            ...             "net.4.weight": "a2c_network.mu.weight",
            ...             "net.4.bias": "a2c_network.mu.bias"}
            >>> model.migrate(path="./runs/Cartpole/nn/Cartpole.pth", name_map=name_map, verbose=True)
            [rlmodule:INFO] Models
            [rlmodule:INFO]   |-- current: 7 items
            [rlmodule:INFO]   |    |-- log_std_parameter : torch.Size([1])
            [rlmodule:INFO]   |    |-- net.0.weight : torch.Size([32, 4])
            [rlmodule:INFO]   |    |-- net.0.bias : torch.Size([32])
            [rlmodule:INFO]   |    |-- net.2.weight : torch.Size([32, 32])
            [rlmodule:INFO]   |    |-- net.2.bias : torch.Size([32])
            [rlmodule:INFO]   |    |-- net.4.weight : torch.Size([1, 32])
            [rlmodule:INFO]   |    |-- net.4.bias : torch.Size([1])
            [rlmodule:INFO]   |-- source: 15 items
            [rlmodule:INFO]   |    |-- value_mean_std.running_mean : torch.Size([1])
            [rlmodule:INFO]   |    |-- value_mean_std.running_var : torch.Size([1])
            [rlmodule:INFO]   |    |-- value_mean_std.count : torch.Size([])
            [rlmodule:INFO]   |    |-- running_mean_std.running_mean : torch.Size([4])
            [rlmodule:INFO]   |    |-- running_mean_std.running_var : torch.Size([4])
            [rlmodule:INFO]   |    |-- running_mean_std.count : torch.Size([])
            [rlmodule:INFO]   |    |-- a2c_network.sigma : torch.Size([1])
            [rlmodule:INFO]   |    |-- a2c_network.actor_mlp.0.weight : torch.Size([32, 4])
            [rlmodule:INFO]   |    |-- a2c_network.actor_mlp.0.bias : torch.Size([32])
            [rlmodule:INFO]   |    |-- a2c_network.actor_mlp.2.weight : torch.Size([32, 32])
            [rlmodule:INFO]   |    |-- a2c_network.actor_mlp.2.bias : torch.Size([32])
            [rlmodule:INFO]   |    |-- a2c_network.value.weight : torch.Size([1, 32])
            [rlmodule:INFO]   |    |-- a2c_network.value.bias : torch.Size([1])
            [rlmodule:INFO]   |    |-- a2c_network.mu.weight : torch.Size([1, 32])
            [rlmodule:INFO]   |    |-- a2c_network.mu.bias : torch.Size([1])
            [rlmodule:INFO] Migration
            [rlmodule:INFO]   |-- map:  log_std_parameter <- a2c_network.sigma
            [rlmodule:INFO]   |-- auto: net.0.weight <- a2c_network.actor_mlp.0.weight
            [rlmodule:INFO]   |-- map:  net.0.bias <- a2c_network.actor_mlp.0.bias
            [rlmodule:INFO]   |-- auto: net.2.weight <- a2c_network.actor_mlp.2.weight
            [rlmodule:INFO]   |-- map:  net.2.bias <- a2c_network.actor_mlp.2.bias
            [rlmodule:INFO]   |-- map:  net.4.weight <- a2c_network.mu.weight
            [rlmodule:INFO]   |-- map:  net.4.bias <- a2c_network.mu.bias
            False

            # migrate a stable-baselines3 checkpoint with unambiguous state_dict
            >>> model.migrate(path="./ddpg_pendulum.zip")
            True

            # migrate from any exported model by loading its state_dict (unambiguous state_dict)
            >>> state_dict = torch.load("./external_model.pt")
            >>> model.migrate(state_dict=state_dict)
            True
        """
        if (state_dict is not None) + (path is not None) != 1:
            raise ValueError("Exactly one of state_dict or path may be specified")

        # load state_dict from path
        if path is not None:
            state_dict = {}
            # rl_games checkpoint
            if path.endswith(".pt") or path.endswith(".pth"):
                checkpoint = torch.load(path, map_location=self.device)
                if type(checkpoint) is dict:
                    state_dict = checkpoint.get("model", {})
            # stable-baselines3
            elif path.endswith(".zip"):
                import zipfile

                try:
                    archive = zipfile.ZipFile(path, "r")
                    with archive.open("policy.pth", mode="r") as file:
                        state_dict = torch.load(file, map_location=self.device)
                except KeyError as e:
                    logger.warning(str(e))
                    state_dict = {}
            else:
                raise ValueError("Cannot identify file type")

        # show state_dict
        if verbose:
            logger.info("Models")
            logger.info(f"  |-- current: {len(self.state_dict().keys())} items")
            for name, tensor in self.state_dict().items():
                logger.info(f"  |    |-- {name} : {list(tensor.shape)}")
            logger.info(f"  |-- source: {len(state_dict.keys())} items")
            for name, tensor in state_dict.items():
                logger.info(f"  |    |-- {name} : {list(tensor.shape)}")
            logger.info("Migration")

        # migrate the state_dict to current model
        new_state_dict = collections.OrderedDict()
        match_counter = collections.defaultdict(list)
        used_counter = collections.defaultdict(list)
        for name, tensor in self.state_dict().items():
            for external_name, external_tensor in state_dict.items():
                # mapped names
                if name_map.get(name, "") == external_name:
                    if tensor.shape == external_tensor.shape:
                        new_state_dict[name] = external_tensor
                        match_counter[name].append(external_name)
                        used_counter[external_name].append(name)
                        if verbose:
                            logger.info(f"  |-- map:  {name} <- {external_name}")
                        break
                    else:
                        logger.warning(
                            f"Shape mismatch for {name} <- {external_name} : {tensor.shape} != {external_tensor.shape}"
                        )
                # auto-mapped names
                if auto_mapping and name not in name_map:
                    if tensor.shape == external_tensor.shape:
                        if name.endswith(".weight"):
                            if external_name.endswith(".weight"):
                                new_state_dict[name] = external_tensor
                                match_counter[name].append(external_name)
                                used_counter[external_name].append(name)
                                if verbose:
                                    logger.info(f"  |-- auto: {name} <- {external_name}")
                        elif name.endswith(".bias"):
                            if external_name.endswith(".bias"):
                                new_state_dict[name] = external_tensor
                                match_counter[name].append(external_name)
                                used_counter[external_name].append(name)
                                if verbose:
                                    logger.info(f"  |-- auto: {name} <- {external_name}")
                        else:
                            if not external_name.endswith(".weight") and not external_name.endswith(".bias"):
                                new_state_dict[name] = external_tensor
                                match_counter[name].append(external_name)
                                used_counter[external_name].append(name)
                                if verbose:
                                    logger.info(f"  |-- auto: {name} <- {external_name}")

        # show ambiguous matches
        status = True
        for name, tensor in self.state_dict().items():
            if len(match_counter.get(name, [])) > 1:
                logger.warning("Ambiguous match for {} <- [{}]".format(name, ", ".join(match_counter.get(name, []))))
                status = False
        # show missing matches
        for name, tensor in self.state_dict().items():
            if not match_counter.get(name, []):
                logger.warning(f"Missing match for {name}")
                status = False
        # show multiple uses
        for name, tensor in state_dict.items():
            if len(used_counter.get(name, [])) > 1:
                logger.warning("Multiple use of {} -> [{}]".format(name, ", ".join(used_counter.get(name, []))))
                status = False

        # load new state dict
        self.load_state_dict(new_state_dict, strict=False)
        self.eval()

        return status

    def freeze_parameters(self, freeze: bool = True) -> None:
        """Freeze or unfreeze internal parameters

        - Freeze: disable gradient computation (``parameters.requires_grad = False``)
        - Unfreeze: enable gradient computation (``parameters.requires_grad = True``)

        :param freeze: Freeze the internal parameters if True, otherwise unfreeze them (default: ``True``)
        :type freeze: bool, optional

        Example::

            # freeze model parameters
            >>> model.freeze_parameters(True)

            # unfreeze model parameters
            >>> model.freeze_parameters(False)
        """
        for parameters in self.parameters():
            parameters.requires_grad = not freeze

    def update_parameters(self, model: torch.nn.Module, polyak: float = 1) -> None:
        """Update internal parameters by hard or soft (polyak averaging) update

        - Hard update: :math:`\\theta = \\theta_{net}`
        - Soft (polyak averaging) update: :math:`\\theta = (1 - \\rho) \\theta + \\rho \\theta_{net}`

        :param model: Model used to update the internal parameters
        :type model: torch.nn.Module (skrl.models.torch.Model)
        :param polyak: Polyak hyperparameter between 0 and 1 (default: ``1``).
                       A hard update is performed when its value is 1
        :type polyak: float, optional

        Example::

            # hard update (from source model)
            >>> model.update_parameters(source_model)

            # soft update (from source model)
            >>> model.update_parameters(source_model, polyak=0.005)
        """
        with torch.no_grad():
            # hard update
            if polyak == 1:
                for parameters, model_parameters in zip(self.parameters(), model.parameters()):
                    parameters.data.copy_(model_parameters.data)
            # soft update (use in-place operations to avoid creating new parameters)
            else:
                for parameters, model_parameters in zip(self.parameters(), model.parameters()):
                    parameters.data.mul_(1 - polyak)
                    parameters.data.add_(polyak * model_parameters.data)
