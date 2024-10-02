import torch
import torch.nn as nn

# # todo consider get rid of this and just pass torch.nn stuff
# def _get_activation_function(activation: str) -> nn.Module:
#     """Get the activation function

#     Supported activation functions:

#     - "elu"
#     - "leaky_relu"
#     - "relu"
#     - "selu"
#     - "sigmoid"
#     - "softmax"
#     - "softplus"
#     - "softsign"
#     - "tanh"

#     :param activation: activation function name.
#                        If activation is an empty string, a placeholder will be returned (``torch.nn.Identity()``)
#     :type activation: str

#     :raises: ValueError if activation is not a valid activation function

#     :return: activation function
#     :rtype: nn.Module
#     """
#     if not activation:
#         return torch.nn.Identity()
#     elif activation == "relu":
#         return torch.nn.ReLU()
#     elif activation == "tanh":
#         return torch.nn.Tanh()
#     elif activation == "sigmoid":
#         return torch.nn.Sigmoid()
#     elif activation == "leaky_relu":
#         return torch.nn.LeakyReLU()
#     elif activation == "elu":
#         return torch.nn.ELU()
#     elif activation == "softplus":
#         return torch.nn.Softplus()
#     elif activation == "softsign":
#         return torch.nn.Softsign()
#     elif activation == "selu":
#         return torch.nn.SELU()
#     elif activation == "softmax":
#         return torch.nn.Softmax()
#     else:
#         raise ValueError(f"Unknown activation function: {activation}")



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


class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()

        input_size = params['input_size']
        hidden_units = params['hidden_units']
        activations = params['activations']

        # input layer
        layers = [
            nn.Linear(input_size, hidden_units[0]),
            activations[0],
        ]

        # hidden layers
        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            layers.append(activations[i + 1])

        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        return self.mlp(input)


def example_MLP():
    params = {'input_size': 517, 'hidden_units': [2048, 1024, 1024, 512], 'activations': ['elu', 'elu', 'elu', 'elu']}
    return MLP(params)


def get_cnn_layer(params):
    """
    Create a CNN layer based on the provided parameters and activation function.

    Args:
        params (dict): Dictionary containing the parameters for the layer.
            Expected keys are:
                - 'type' (str): Type of the layer ('conv' for convolutional, 'pool' for pooling,
                                                   'dense' for fully connected).
                - 'in_channels' (int): Number of input channels (required for 'conv' type).
                - 'out_channels' (int): Number of output channels (required for 'conv' type).
                - 'kernel_size' (int or tuple): Size of the kernel (required for 'conv' or 'pool' type).
                - 'stride' (int or tuple): Stride of the convolution or pooling operation
                                           (required for 'conv' or 'pool' type).
                - 'in_features' (int): Number of input features (required for 'dense' type).
                - 'out_features' (int): Number of output features (required for 'dense' type).
                - 'activation' (str): Activation function to use after the layer (only for 'conv' and 'dense' types).

    Returns:
        list: List containing the created layer(s). For 'conv' type, it includes the convolutional
              layer followed by the activation function. For 'pool' type, it includes only the pooling layer.
              For 'dense' type, it includes the fully connected layer followed by the activation function.

    Raises:
        ValueError: If the 'type' specified in params is not supported.
    """
    if params['type'] == 'conv':
        return [
            nn.Conv2d(
                in_channels=params['in_channels'],
                out_channels=params['out_channels'],
                kernel_size=params['kernel_size'],
                stride=params['stride'],
            ),
            _get_activation_function(params['activation']),
        ]
    elif params['type'] == 'pool':
        return [
            nn.MaxPool2d(
                kernel_size=params['kernel_size'],
                stride=params['stride'],
            )
        ]
    elif params['type'] == 'dense':
        return [
            nn.Flatten(),  # if there is 2D layer before need to be flatten to 1D.
            nn.Linear(
                in_features=params['in_features'],
                out_features=params['out_features'],
            ),
            _get_activation_function(params['activation']),
        ]
    else:
        raise ValueError(f"Unsupported layer type: {params['type']}")


class CNN(nn.Module):
    def __init__(self, params):
        super().__init__()

        layers = sum([get_cnn_layer(layer_params) for layer_params in params['layers']], [])
        self.cnn = nn.Sequential(*layers, nn.Flatten())

    def forward(self, input):
        return self.cnn(input)


def example_CNN():
    params = {
        'input_shape': [1, 13, 13],
        'layers': [
            {'type': 'conv', 'kernel_size': 3, 'stride': 2, 'in_channels': 1, 'out_channels': 32, 'activation': 'relu'},
            {
                'type': 'conv',
                'kernel_size': 3,
                'stride': 1,
                'in_channels': 32,
                'out_channels': 64,
                'activation': 'relu',
            },
        ],
    }

    return CNN(params)


class TripleCnnAndMlp(nn.Module):
    """
    Split the observation space into parts and pass some parts through different CNNs.

    ----------------------------------------------------------------------
    |                       Network architecture  (cnn shape 1x13x13)    |
    |--------------------------------------------------------------------|
    |                                                                    |
    |       10  1x13x13  1x13x13  1x13x13                                |
    |       |    |         |         |                                   |
    |       |   CNN0      CNN1      CNN2                                 |
    |       |    |         |         |                                   |
    |       | Flatten   Flatten   Flatten                                |
    |       |    |         |         |                                   |
    |       |____|_________|_________|                                   |
    |       |                                                            |
    |       Join to shape 10 + OutUnits0 + OutUnits1 + OutUnits2         |
    |       |                                                            |
    |       MLP: UNITS[0]                                                |
    |       |                                                            |
    |       ...                                                          |
    |       |                                                            |
    |       MLP: UNITS[-1]                                               |
    |       |                                                            |
    |       Output                                                       |
    |                                                                    |
    ----------------------------------------------------------------------

    """

    def __init__(self, cnn_params, mlp_params):
        super().__init__()
        self.input_shape = cnn_params['input_shape']

        self.prefix_length = 10
        self.cnn_number = 3

        self.cnns = nn.ModuleList([CNN(cnn_params) for _ in range(self.cnn_number)])

        mlp_params['input_size'] = self.prefix_length + self.cnn_number * get_output_size(
            self.cnns[0], self.input_shape
        )

        self.mlp = MLP(mlp_params)

    def forward(self, input):
        # Split the input
        prefix = input[:, : self.prefix_length]

        c, h, w = self.input_shape
        cnn_input = input[:, self.prefix_length :].view(-1, self.cnn_number * c, h, w)

        # Forward pass through CNNs
        cnn_outputs = []
        for i in range(self.cnn_number):
            cnn_output = self.cnns[i](cnn_input[:, i : i + 1, :, :]).view(input.size(0), -1)
            cnn_outputs.append(cnn_output)

        # Concatenate the outputs
        combined = torch.cat([prefix] + cnn_outputs, dim=1)

        # Forward pass through MLP
        output = self.mlp(combined)
        return output


def triple_cnn_and_mlp_example():
    cnn_params = {
        'input_shape': [1, 13, 13],
        'layers': [
            {'type': 'conv', 'kernel_size': 3, 'stride': 2, 'in_channels': 1, 'out_channels': 32, 'activation': 'relu'},
            {
                'type': 'conv',
                'kernel_size': 3,
                'stride': 1,
                'in_channels': 32,
                'out_channels': 64,
                'activation': 'relu',
            },
        ],
    }

    mlp_params = {'hidden_units': [1024, 1024, 512], 'activations': ['elu', 'elu', 'elu']}

    return TripleCnnAndMlp(cnn_params, mlp_params)


class RnnBase(nn.Module):
    def __init__(self, params):
        super().__init__()

        # parameters necessery for RNN, GRU and LSTM networks
        self.num_envs = params['num_envs']
        self.num_layers = params['num_layers']
        self.hidden_size = params['hidden_size']
        self.sequence_length = params['sequence_length']
        if 'input_size' in params:
            self.input_size = params['input_size']


class LSTM(RnnBase):

    def __init__(self, params):
        super().__init__(params)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(self, states, terminated, rnn_inputs):

        # Process rnn inputs
        hidden_states, cell_states = rnn_inputs[0], rnn_inputs[1]

        # Dimensions Explained
        # ---------------------
        # 'N'       - Batch size
        # 'L'       - Sequence length
        # 'Hin'     - Input size (number of features per input at each time step)
        # 'Hout'    - Hidden size (number of features in the hidden state of the LSTM)
        # 'HCell'   - Cell state size (same as hidden size in LSTM context)
        # 'D'       - Number of directions (1 for unidirectional, 2 for bidirectional)
        # 'num_layers' - Number of stacked LSTM layers

        # training
        if self.training:
            # reshape (N * L, Hin) -> (N, L, Hin)
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])

            # reshape (D * num_layers, N * L, Hout) -> (D * num_layers, N, L, Hout)
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])

            # reshape (D * num_layers, N * L, Hcell) -> (D * num_layers, N, L, Hcell)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])

            # get the hidden/cell states corresponding to the first time step of the sequence for each batch.
            hidden_states = hidden_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    # Compute the RNN output for the segment of the sequence
                    rnn_output, (hidden_states, cell_states) = self.lstm(
                        rnn_input[:, i0:i1, :], (hidden_states, cell_states)
                    )
                    # Reset hidden and cell states where sequences are terminated
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    cell_states[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # if no termination reset is needed, simply compute the RNN output
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            # reshape (N, Hin) -> (N, 1, Hin)
            # This is done to add a sequence length dimension of 1 for processing one time step at a time
            rnn_input = states.view(-1, 1, states.shape[-1])
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # reshape (N, L, D * Hout) -> (N * L, D * Hout)
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        return rnn_output, {'rnn': [rnn_states[0], rnn_states[1]]}


class RnnModule(RnnBase):

    def __init__(self, params):
        super().__init__(params)

    def forward(self, states, terminated, rnn_inputs):

        # Process rnn inputs
        hidden_states = rnn_inputs[0]

        # Dimensions Explained
        # ---------------------
        # 'N'       - Batch size
        # 'L'       - Sequence length
        # 'Hin'     - Input size (number of features per input at each time step)
        # 'Hout'    - Hidden size (number of features in the hidden state of the RNN)
        # 'D'       - Number of directions (1 for unidirectional, 2 for bidirectional)
        # 'num_layers' - Number of stacked RNN layers

        # training
        if self.training:
            # reshape (N * L, Hin) -> (N, L, Hin)
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])

            # reshape (D * num_layers, N * L, Hout) -> (D * num_layers, N, L, Hout)
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])

            # get the hidden states corresponding to the first time step of the sequence for each batch.
            hidden_states = hidden_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hout)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    # Compute the RNN output for the segment of the sequence
                    rnn_output, hidden_states = self.rnn(rnn_input[:, i0:i1, :], hidden_states)
                    # Reset hidden states where sequences are terminated
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # if no termination reset is needed, simply compute the RNN output
            else:
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
        # rollout
        else:
            # reshape (N, Hin) -> (N, 1, Hin)
            # This is done to add a sequence length dimension of 1 for processing one time step at a time
            rnn_input = states.view(-1, 1, states.shape[-1])
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

        # reshape (N, L, D * Hout) -> (N * L, D * Hout)
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        return rnn_output, {'rnn': [hidden_states]}


class RNN(RnnModule):
    def __init__(self, params):
        super().__init__(params)

        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )


class GRU(RnnModule):
    def __init__(self, params):
        super().__init__(params)

        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )


def example_RNN():
    params = {
        'num_envs': 2048,
        'input_size': 517,
        'num_layers': 1,
        'hidden_size': 512 + 256,
        'sequence_length': 128,
    }
    return RNN(params)


def example_GRU():
    params = {
        'num_envs': 2048,
        'input_size': 517,
        'num_layers': 1,
        'hidden_size': 512 + 256,
        'sequence_length': 128,
    }
    return GRU(params)


def example_LSTM():
    params = {
        'num_envs': 2048,
        'input_size': 517,
        'num_layers': 1,
        'hidden_size': 512 + 256,
        'sequence_length': 32,
    }
    return LSTM(params)


class RnnMlp(RnnBase):

    def __init__(self, rnn_class, rnn_params, mlp_params):
        super().__init__(rnn_params)

        self.rnn = rnn_class(rnn_params)

        mlp_params['input_size'] = self.hidden_size

        self.mlp = MLP(mlp_params)

    def forward(self, states, terminated, rnn_inputs):
        rnn_output, output_dict = self.rnn(states, terminated, rnn_inputs)
        return self.mlp(rnn_output), output_dict


def example_LstmMlp():
    rnn_params = {
        'num_envs': 2048,
        'input_size': 517,
        'num_layers': 1,
        'hidden_size': 512 + 256,
        'sequence_length': 128,
    }
    mlp_params = {
        'hidden_units': [2048, 1024, 1024, 512],
        'activations': ['elu', 'elu', 'elu', 'elu'],
    }

    return RnnMlp(LSTM, rnn_params, mlp_params)


def example_GruMlp():
    rnn_params = {
        'num_envs': 2048,
        'input_size': 517,
        'num_layers': 1,
        'hidden_size': 512 + 256,
        'sequence_length': 128,
    }
    mlp_params = {
        'hidden_units': [2048, 1024, 1024, 512],
        'activations': ['elu', 'elu', 'elu', 'elu'],
    }

    return RnnMlp(GRU, rnn_params, mlp_params)


def example_RnnMlp():
    rnn_params = {
        'num_envs': 2048,
        'input_size': 517,
        'num_layers': 1,
        'hidden_size': 512 + 256,
        'sequence_length': 128,
    }
    mlp_params = {
        'hidden_units': [2048, 1024, 1024, 512],
        'activations': ['elu', 'elu', 'elu', 'elu'],
    }

    return RnnMlp(RNN, rnn_params, mlp_params)


class RnnMlpWithForwardedInput(RnnBase):

    def __init__(self, rnn_class, rnn_params, mlp_params):
        super().__init__(rnn_params)

        self.rnn = rnn_class(rnn_params)

        mlp_params['input_size'] = self.input_size + self.hidden_size

        self.mlp = MLP(mlp_params)

    def forward(self, states, terminated, rnn_inputs):
        rnn_output, output_dict = self.rnn(states, terminated, rnn_inputs)
        mlp_input = torch.cat((states, rnn_output), dim=1)

        return self.mlp(mlp_input), output_dict


def example_LstmMlpWithForwardedInput():
    rnn_params = {
        'num_envs': 2048,
        'input_size': 517,
        'num_layers': 1,
        'hidden_size': 512 + 256,
        'sequence_length': 128,
    }
    mlp_params = {
        'hidden_units': [2048, 1024, 1024, 512],
        'activations': ['elu', 'elu', 'elu', 'elu'],
    }

    return RnnMlpWithForwardedInput(LSTM, rnn_params, mlp_params)


def example_GruMlpWithForwardedInput():
    rnn_params = {
        'num_envs': 2048,
        'input_size': 517,
        'num_layers': 1,
        'hidden_size': 512 + 256,
        'sequence_length': 128,
    }
    mlp_params = {
        'hidden_units': [2048, 1024, 1024, 512],
        'activations': ['elu', 'elu', 'elu', 'elu'],
    }

    return RnnMlpWithForwardedInput(GRU, rnn_params, mlp_params)


def example_RnnMlpWithForwardedInput():
    rnn_params = {
        'num_envs': 2048,
        'input_size': 517,
        'num_layers': 1,
        'hidden_size': 512 + 256,
        'sequence_length': 128,
    }
    mlp_params = {
        'hidden_units': [2048, 1024, 1024, 512],
        'activations': ['elu', 'elu', 'elu', 'elu'],
    }

    return RnnMlpWithForwardedInput(RNN, rnn_params, mlp_params)


class CrazyNet(RnnBase):
    def __init__(self, rnn_class, rnn_params, cnn_params, mlp_params):
        super().__init__(rnn_params)

        self.input_shape = cnn_params['input_shape']

        self.prefix_length = 10
        self.cnn_number = 3

        self.cnns = nn.ModuleList([CNN(cnn_params) for _ in range(self.cnn_number)])

        rnn_params['input_size'] = self.prefix_length + self.cnn_number * get_output_size(
            self.cnns[0], self.input_shape
        )

        self.rnn = rnn_class(rnn_params)

        mlp_params['input_size'] = self.rnn.hidden_size + self.rnn.input_size

        self.mlp = MLP(mlp_params)

    def forward(self, states, terminated, rnn_inputs):
        # Split the input
        prefix = states[:, : self.prefix_length]

        c, h, w = self.input_shape
        cnn_input = states[:, self.prefix_length :].view(-1, self.cnn_number * c, h, w)

        # Forward pass through CNNs
        cnn_outputs = []
        for i in range(self.cnn_number):
            cnn_output = self.cnns[i](cnn_input[:, i : i + 1, :, :]).view(states.size(0), -1)
            cnn_outputs.append(cnn_output)

        # Concatenate the outputs of prefix and CNNs
        cnn_combined = torch.cat([prefix] + cnn_outputs, dim=1)

        # Forward pass through RNN
        rnn_output, output_dict = self.rnn(cnn_combined, terminated, rnn_inputs)

        # Concatenate the outputs of RNN, CNNs and prefix
        combined = torch.cat([cnn_combined, rnn_output], dim=1)

        # Forward pass through MLP
        output = self.mlp(combined)

        return output, output_dict


def crazy_net_example():
    rnn_params = {
        'num_envs': 2048,
        'num_layers': 1,
        'hidden_size': 512 + 256,
        'sequence_length': 128,
    }

    cnn_params = {
        'input_shape': [1, 13, 13],
        'layers': [
            {'type': 'conv', 'kernel_size': 3, 'stride': 2, 'in_channels': 1, 'out_channels': 32, 'activation': 'relu'},
            {
                'type': 'conv',
                'kernel_size': 3,
                'stride': 1,
                'in_channels': 32,
                'out_channels': 64,
                'activation': 'relu',
            },
        ],
    }

    mlp_params = {'hidden_units': [1024, 1024, 512], 'activations': ['elu', 'elu', 'elu']}

    return CrazyNet(LSTM, rnn_params, cnn_params, mlp_params)


# def lstm_module(model_params, observation_space, action_space, device, double_pass):

#     class LSTM(nn.Module):

#         def __init__(self):
#             super().__init__()

#             # lstm
#             self.num_envs = 2048
#             self.num_layers = 1
#             self.hidden_size = 256 + 512
#             self.sequence_length = 128

#             self.lstm = nn.LSTM(
#                 input_size=observation_space.shape[0],
#                 hidden_size=self.hidden_size,
#                 num_layers=self.num_layers,
#                 batch_first=True,
#             )

#             # mlp
#             input_size = self.hidden_size

#             mlp_params = model_params['mlp']
#             self.net = DEPRECATEDMLP(input_size, mlp_params['units'], device, mlp_params['activation'])

#         # TODO(ll) can inherit from LSTM ..without mlp
#         def forward(self, inputs):

#             # Process rnn inputs
#             hidden_states, cell_states = rnn_inputs[0], rnn_inputs[1]

#             # Dimensions Explained
#             # ---------------------
#             # 'N'       - Batch size
#             # 'L'       - Sequence length
#             # 'Hin'     - Input size (number of features per input at each time step)
#             # 'Hout'    - Hidden size (number of features in the hidden state of the LSTM)
#             # 'HCell'   - Cell state size (same as hidden size in LSTM context)
#             # 'D'       - Number of directions (1 for unidirectional, 2 for bidirectional)
#             # 'num_layers' - Number of stacked LSTM layers

#             # training
#             if self.training:
#                 # reshape (N * L, Hin) -> (N, L, Hin)
#                 rnn_input = states.view(-1, self.sequence_length, states.shape[-1])

#                 # reshape (D * num_layers, N * L, Hout) -> (D * num_layers, N, L, Hout)
#                 hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])

#                 # reshape (D * num_layers, N * L, Hcell) -> (D * num_layers, N, L, Hcell)
#                 cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])

#                 # get the hidden/cell states corresponding to the first time step of the sequence for each batch.
#                 hidden_states = hidden_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hout)
#                 cell_states = cell_states[:, :, 0, :].contiguous()  # (D * num_layers, N, Hcell)

#                 # reset the RNN state in the middle of a sequence
#                 if terminated is not None and torch.any(terminated):
#                     rnn_outputs = []
#                     terminated = terminated.view(-1, self.sequence_length)
#                     indexes = (
#                         [0]
#                         + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
#                         + [self.sequence_length]
#                     )

#                     for i in range(len(indexes) - 1):
#                         i0, i1 = indexes[i], indexes[i + 1]
#                         # Compute the RNN output for the segment of the sequence
#                         rnn_output, (hidden_states, cell_states) = self.lstm(
#                             rnn_input[:, i0:i1, :], (hidden_states, cell_states)
#                         )
#                         # Reset hidden and cell states where sequences are terminated
#                         hidden_states[:, (terminated[:, i1 - 1]), :] = 0
#                         cell_states[:, (terminated[:, i1 - 1]), :] = 0
#                         rnn_outputs.append(rnn_output)

#                     rnn_states = (hidden_states, cell_states)
#                     rnn_output = torch.cat(rnn_outputs, dim=1)
#                 # if no termination reset is needed, simply compute the RNN output
#                 else:
#                     rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
#             # rollout
#             else:
#                 # reshape (N, Hin) -> (N, 1, Hin)
#                 # This is done to add a sequence length dimension of 1 for processing one time step at a time
#                 rnn_input = states.view(-1, 1, states.shape[-1])
#                 rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

#             # reshape (N, L, D * Hout) -> (N * L, D * Hout)
#             rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

#             return self.net(rnn_output), {'rnn': [rnn_states[0], rnn_states[1]]}

#     # TODO(ll) if not giving in params then just fast return.
#     lstm = LSTM()
#     return lstm


def get_models_shared(model_params, observation_space, action_space, device, double_pass):
    # input_size = observation_space.shape[0]

    # MLP -- change mlp in grouting PPO
    # mlp_params = model_params['mlp']
    # net = MLP(input_size, mlp_params['units'], device, mlp_params['activation'])

    # LSTM -- change lstm in grouting PPO
    # net = lstm_module(model_params, observation_space, action_space, device, double_pass)

    # net = example_MLP()
    # net = crazy_net_example()
    # net = triple_cnn_and_mlp_example()
    net = example_LstmMlp()
    # net = example_LstmMlpWithForwardedInput()

    # TODO --infere this inside of Layer?
    net_output_size = get_output_size(net, observation_space.shape[0])

    model = SharedRLModel(
        observation_space=observation_space,
        action_space=action_space,
        input_shape=Shape.STATES,
        device=device,
        network=net,
        output_layer={
            'policy': GaussianLayer(
                input_size=net_output_size,
                output_size=action_space.shape[0],
                device=device,
                output_shape=Shape.ACTIONS,
                output_activation=nn.Tanh(),
                clip_actions=False,
                clip_log_std=True,
                min_log_std=-1.2,
                max_log_std=2,
                initial_log_std=0.2,
            ),
            'value': DeterministicLayer(
                input_size=net_output_size,
                output_size=1,
                device=device,
                output_shape=Shape.ONE,
                output_activation=nn.Identity(),
            ),
        },
    ).to(device)

    return {'policy': model, 'value': model}


def get_models_separate(model_params, observation_space, action_space, device, double_pass):

    net = example_MLP()
    value_net = example_MLP()

    # change for lstm/mlp
    # net = lstm_module(model_params, observation_space, action_space, device, double_pass)
    # value_net = lstm_module(model_params, observation_space, action_space, device, double_pass)

    policy_model = RLModel(
        observation_space=observation_space,
        action_space=action_space,
        input_shape=Shape.STATES,
        device=device,
        network=net,
        output_layer=GaussianLayer(
            input_size=get_output_size(net, observation_space.shape[0]),
            output_size=action_space.shape[0],
            device=device,
            output_shape=Shape.ACTIONS,
            output_activation=nn.Tanh(),
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20,
            max_log_std=2,
            initial_log_std=0,
        ),
    ).to(device)

    value_model = RLModel(
        observation_space=observation_space,
        action_space=action_space,
        input_shape=Shape.STATES,
        device=device,
        network=value_net,
        output_layer=DeterministicLayer(
            input_size=get_output_size(value_net, observation_space.shape[0]),
            output_size=1,
            device=device,
            output_shape=Shape.ONE,
            output_activation=nn.Identity(),
        ),
    ).to(device)

    return {'policy': policy_model, 'value': value_model}


# def get_models_separate_in(model_params, observation_space, action_space, device, double_pass):

#     if 'mlp' in model_params:

#         # if 'lstm' in model_params:
#         #     input_size = model_params['lstm']['hidden_size']
#         # else:
#         input_size = observation_space.shape[0]  # TODO: is this approach too simplistic

#         if 'cnn' in model_params:
#             net = MixedCNN(model_params, input_size, device)
#         else:
#             mlp_params = model_params['mlp']
#             net = MLP(input_size, mlp_params['units'], device, mlp_params['activation'])
#             value_net = MLP(input_size, mlp_params['units'], device, mlp_params['activation'])
#     else:
#         raise ValueError('Invalid model type')

#     # change for lstm/mlp
#     # net = lstm_module(model_params, observation_space, action_space, device, double_pass)
#     # value_net = lstm_module(model_params, observation_space, action_space, device, double_pass)

#     policy_model = GaussianModel(
#         observation_space=observation_space,
#         action_space=action_space,
#         network=net,
#         device=device,
#         clip_actions=False,
#         clip_log_std=True,
#         min_log_std=-20,
#         max_log_std=2,
#         initial_log_std=0,
#     )

#     value_model = DeterministicModel(
#         observation_space=observation_space,
#         action_space=action_space,
#         network=value_net,
#         device=device,
#         clip_actions=False,
#     )

#     return {'policy': policy_model, 'value': value_model}


# def get_models(model_params, observation_space, action_space, device, double_pass):

#     if 'mlp' in model_params:

#         if 'lstm' in model_params:
#             input_size = model_params['lstm']['hidden_size']
#         else:
#             input_size = observation_space.shape[0]  # TODO: is this approach too simplistic

#         if 'cnn' in model_params:
#             net = MixedCNN(model_params, input_size, device)
#         else:
#             mlp_params = model_params['mlp']
#             net = MLP(input_size, mlp_params['units'], device, mlp_params['activation'])
#     else:
#         raise ValueError('Invalid model type')

#     shared_config = {
#         'clip_actions': False,
#         'clip_log_std': True,
#         'min_log_std': -20,
#         'max_log_std': 2,
#         'initial_log_std': 0,
#         'input_shape': Shape.STATES,
#         'output_scale': 1,
#         'hiddens': model_params['mlp']['units'],
#         'hidden_activation': [model_params['mlp']['activation'] for _ in range(len(model_params['mlp']['units']))],
#         'output_activation': None,
#     }

#     policy_config = shared_config.copy()
#     value_config = shared_config.copy()

#     policy_config['output_activation'] = 'tanh'
#     policy_config['output_shape'] = Shape.ACTIONS
#     value_config['output_shape'] = Shape.ONE

#     if 'lstm' in model_params:
#         model = shared_model_lstm(
#             observation_space=observation_space,
#             action_space=action_space,
#             device=device,
#             roles=['policy', 'value'],
#             parameters=[policy_config, value_config],
#             net=net,
#             double_pass=double_pass,
#         )
#     else:
#         model = shared_model(
#             observation_space=observation_space,
#             action_space=action_space,
#             device=device,
#             roles=['policy', 'value'],
#             parameters=[policy_config, value_config],
#             net=net,
#             double_pass=double_pass,
#         )

#     return {'policy': model, 'value': model}
