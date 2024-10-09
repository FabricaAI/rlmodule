import torch
import torch.nn as nn

from rlmodule.source.utils import get_space_size


class MLP(nn.Module):
    """Configurable multilayer perceptron module.
    
    Architecture is defined by providing modules_cfg.MlpCfg

    Example of use:

        cfg = MlpCfg(
            input_size = 517,
            hidden_units = [2048, 1024, 1024, 512],
            activations = [nn.ELU(), nn.ELU(), nn.ELU(), nn.ELU()],
        )
        net = MLP(cfg)
    """
    def __init__(self, cfg):
        super().__init__()

        cfg.input_size = get_space_size(cfg.input_size)

        # input layer
        layers = [
            nn.Linear(cfg.input_size, cfg.hidden_units[0]),
            cfg.activations[0](),
        ]

        # hidden layers
        for i in range(len(cfg.hidden_units) - 1):
            layers.append(nn.Linear(cfg.hidden_units[i], cfg.hidden_units[i + 1]))
            layers.append(cfg.activations[i + 1]())

        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        return self.mlp(input)


# TODO annotation types to config
class RnnBase(nn.Module):
    """Base class for all modules containing RNN (including GRU, LSTM) substructure.

    All modules that have RNN substructure should inherit from this module.
    It makes sure all the necessary variables used by RL algorithm are defined.
    """
    def __init__(self, cfg):
        super().__init__()
        
        # parameters necessary for RNN, GRU and LSTM networks
        self.num_envs = cfg.num_envs
        self.num_layers = cfg.num_layers
        self.hidden_size = cfg.hidden_size
        self.sequence_length = cfg.sequence_length

        cfg.input_size = get_space_size(cfg.input_size)


class LSTM(RnnBase):
    """Configurable LSTM module.
    
    Architecture is defined by providing modules_cfg.LstmCfg.
    
    Contains implementation of forward pass handling hidden and cell states 
    necessary for run with PPO_RNN algorithm.

    Example of use:

        cfg = LstmCfg(
            input_size = 517,
            num_envs = 2048,
            num_layers = 1,
            hidden_size = 512 + 256,
            sequence_length = 128,
        )
        net = LSTM(cfg)
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
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
    """Common base class for RNN and GRU module.
    
    Contains implementation of forward pass handling hidden states 
    necessary for run with PPO_RNN algorithm.
    """
    def __init__(self, cfg):
        super().__init__(cfg)

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
    """Configurable RNN module.
    
    Architecture is defined by providing modules_cfg.RnnCfg.
    
    Contains implementation of forward pass handling hidden states 
    necessary for run with PPO_RNN algorithm.

    Example of use:

        cfg = RnnCfg(
            input_size = 517,
            num_envs = 2048,
            num_layers = 1,
            hidden_size = 512 + 256,
            sequence_length = 128,
        )
        net = Rnn(cfg)
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.rnn = nn.RNN(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
        )


class GRU(RnnModule):
    """Configurable GRU module.
    
    Architecture is defined by providing modules_cfg.GruCfg.
    
    Contains implementation of forward pass handling hidden states 
    necessary for run with PPO_RNN algorithm.

    Example of use:

        cfg = GruCfg(
            input_size = 517,
            num_envs = 2048,
            num_layers = 1,
            hidden_size = 512 + 256,
            sequence_length = 128,
        )
        net = Gru(cfg)
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.rnn = nn.GRU(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
        )


class RnnMlp(RnnBase):
    """Configurable module for Rnn-based module followed by MLP.
    
    Architecture is defined by providing modules_cfg.RnnMlpCfg,
    based on which the RNN, GRU or LSTM module will be used.
    
    Inherits implementation of forward pass handling hidden states 
    necessary for run with PPO_RNN algorithm.

    1) Example of use (with RNN module):

        cfg = RnnMlpCfg(
                input_size = 517,
                rnn = RnnCfg(
                    num_envs = 2048,
                    num_layers = 1,
                    hidden_size = 512 + 256,
                    sequence_length = 128,
                ),
                mlp = MlpCfg(
                    hidden_units = [2048, 1024, 1024, 512],
                    activations = [nn.ELU(), nn.ELU(), nn.ELU(), nn.ELU()],
                )
            )
        net = RnnMlp(cfg)

    2) Example of use (with GRU module):

        cfg = RnnMlpCfg(
                input_size = 517,
                rnn = GruCfg(
                    num_envs = 2048,
                    num_layers = 1,
                    hidden_size = 512 + 256,
                    sequence_length = 128,
                ),
                mlp = MlpCfg(
                    hidden_units = [2048, 1024, 1024, 512],
                    activations = [nn.ELU(), nn.ELU(), nn.ELU(), nn.ELU()],
                )
            )
        net = RnnMlp(cfg)

    3) Example of use (with LSTM module):

        cfg = RnnMlpCfg(
                input_size = 517,
                rnn = LstmCfg(
                    num_envs = 2048,
                    num_layers = 1,
                    hidden_size = 512 + 256,
                    sequence_length = 128,
                ),
                mlp = MlpCfg(
                    hidden_units = [2048, 1024, 1024, 512],
                    activations = [nn.ELU(), nn.ELU(), nn.ELU(), nn.ELU()],
                )
            )
        net = RnnMlp(cfg)
    """
    def __init__(self, cfg):
        super().__init__(cfg.rnn)
        cfg.input_size = get_space_size(cfg.input_size)
        cfg.rnn.input_size = cfg.input_size
        self.rnn = cfg.rnn.module(cfg.rnn)
        cfg.mlp.input_size = self.hidden_size
        self.mlp = MLP(cfg.mlp)

    def forward(self, states, terminated, rnn_inputs):
        rnn_output, output_dict = self.rnn(states, terminated, rnn_inputs)
        return self.mlp(rnn_output), output_dict


class RnnMlpWithForwardedInput(RnnBase):
    """Configurable module for Rnn-based module followed by MLP with
    fast-forward input into MLP.
    
    Architecture is defined by providing modules_cfg.RnnMlpCfg,
    based on which the RNN, GRU or LSTM module will be used.
    
    Inherits implementation of forward pass handling hidden states 
    necessary for run with PPO_RNN algorithm.

    1) Example of use (with RNN module):

        cfg = RnnMlpCfg(
                input_size = 517,
                module = RnnMlpWithForwardedInput,
                rnn = RnnCfg(
                    num_envs = 2048,
                    num_layers = 1,
                    hidden_size = 512 + 256,
                    sequence_length = 128,
                ),
                mlp = MlpCfg(
                    hidden_units = [2048, 1024, 1024, 512],
                    activations = [nn.ELU(), nn.ELU(), nn.ELU(), nn.ELU()],
                )
            )
        net = RnnMlp(cfg)

    2) Example of use (with GRU module):

        cfg = RnnMlpCfg(
                input_size = 517,
                module = RnnMlpWithForwardedInput,
                rnn = GruCfg(
                    num_envs = 2048,
                    num_layers = 1,
                    hidden_size = 512 + 256,
                    sequence_length = 128,
                ),
                mlp = MlpCfg(
                    hidden_units = [2048, 1024, 1024, 512],
                    activations = [nn.ELU(), nn.ELU(), nn.ELU(), nn.ELU()],
                )
            )
        net = RnnMlp(cfg)

    3) Example of use (with LSTM module):

        cfg = RnnMlpCfg(
                input_size = 517,
                module = RnnMlpWithForwardedInput,
                rnn = LstmCfg(
                    num_envs = 2048,
                    num_layers = 1,
                    hidden_size = 512 + 256,
                    sequence_length = 128,
                ),
                mlp = MlpCfg(
                    hidden_units = [2048, 1024, 1024, 512],
                    activations = [nn.ELU(), nn.ELU(), nn.ELU(), nn.ELU()],
                )
            )
        net = RnnMlp(cfg)

    Note: `module = RnnMlpWithForwardedInput` is not necessary for standalone example
        to work. However it is included here because it is necessary for use in rlmodule.build_model to
        differentiate it from RnnMlp class.
    """
    def __init__(self, cfg):
        super().__init__(cfg.rnn)
        cfg.input_size = get_space_size(cfg.input_size)
        cfg.rnn.input_size = cfg.input_size
        self.rnn = cfg.rnn.module(cfg.rnn)
        cfg.mlp.input_size = cfg.input_size + self.hidden_size
        self.mlp = MLP(cfg.mlp)

    def forward(self, states, terminated, rnn_inputs):
        rnn_output, output_dict = self.rnn(states, terminated, rnn_inputs)
        mlp_input = torch.cat((states, rnn_output), dim=1)

        return self.mlp(mlp_input), output_dict



# CNN - coming soon 

# def get_cnn_layer(params):
#     """
#     Create a CNN layer based on the provided parameters and activation function.

#     Args:
#         params (dict): Dictionary containing the parameters for the layer.
#             Expected keys are:
#                 - 'type' (str): Type of the layer ('conv' for convolutional, 'pool' for pooling,
#                                                    'dense' for fully connected).
#                 - 'in_channels' (int): Number of input channels (required for 'conv' type).
#                 - 'out_channels' (int): Number of output channels (required for 'conv' type).
#                 - 'kernel_size' (int or tuple): Size of the kernel (required for 'conv' or 'pool' type).
#                 - 'stride' (int or tuple): Stride of the convolution or pooling operation
#                                            (required for 'conv' or 'pool' type).
#                 - 'in_features' (int): Number of input features (required for 'dense' type).
#                 - 'out_features' (int): Number of output features (required for 'dense' type).
#                 - 'activation' (str): Activation function to use after the layer (only for 'conv' and 'dense' types).

#     Returns:
#         list: List containing the created layer(s). For 'conv' type, it includes the convolutional
#               layer followed by the activation function. For 'pool' type, it includes only the pooling layer.
#               For 'dense' type, it includes the fully connected layer followed by the activation function.

#     Raises:
#         ValueError: If the 'type' specified in params is not supported.
#     """
#     if params['type'] == 'conv':
#         return [
#             nn.Conv2d(
#                 in_channels=params['in_channels'],
#                 out_channels=params['out_channels'],
#                 kernel_size=params['kernel_size'],
#                 stride=params['stride'],
#             ),
#             _get_activation_function(params['activation']),
#         ]
#     elif params['type'] == 'pool':
#         return [
#             nn.MaxPool2d(
#                 kernel_size=params['kernel_size'],
#                 stride=params['stride'],
#             )
#         ]
#     elif params['type'] == 'dense':
#         return [
#             nn.Flatten(),  # if there is 2D layer before need to be flatten to 1D.
#             nn.Linear(
#                 in_features=params['in_features'],
#                 out_features=params['out_features'],
#             ),
#             _get_activation_function(params['activation']),
#         ]
#     else:
#         raise ValueError(f"Unsupported layer type: {params['type']}")


# class CNN(nn.Module):
#     def __init__(self, params):
#         super().__init__()

#         layers = sum([get_cnn_layer(layer_params) for layer_params in params['layers']], [])
#         self.cnn = nn.Sequential(*layers, nn.Flatten())

#     def forward(self, input):
#         return self.cnn(input)


# def example_CNN():
#     params = {
#         'input_shape': [1, 13, 13],
#         'layers': [
#             {'type': 'conv', 'kernel_size': 3, 'stride': 2, 'in_channels': 1, 'out_channels': 32, 'activation': 'relu'},
#             {
#                 'type': 'conv',
#                 'kernel_size': 3,
#                 'stride': 1,
#                 'in_channels': 32,
#                 'out_channels': 64,
#                 'activation': 'relu',
#             },
#         ],
#     }

#     return CNN(params)


# class TripleCnnAndMlp(nn.Module):
#     """
#     Split the observation space into parts and pass some parts through different CNNs.

#     ----------------------------------------------------------------------
#     |                       Network architecture  (cnn shape 1x13x13)    |
#     |--------------------------------------------------------------------|
#     |                                                                    |
#     |       10  1x13x13  1x13x13  1x13x13                                |
#     |       |    |         |         |                                   |
#     |       |   CNN0      CNN1      CNN2                                 |
#     |       |    |         |         |                                   |
#     |       | Flatten   Flatten   Flatten                                |
#     |       |    |         |         |                                   |
#     |       |____|_________|_________|                                   |
#     |       |                                                            |
#     |       Join to shape 10 + OutUnits0 + OutUnits1 + OutUnits2         |
#     |       |                                                            |
#     |       MLP: UNITS[0]                                                |
#     |       |                                                            |
#     |       ...                                                          |
#     |       |                                                            |
#     |       MLP: UNITS[-1]                                               |
#     |       |                                                            |
#     |       Output                                                       |
#     |                                                                    |
#     ----------------------------------------------------------------------

#     """

#     def __init__(self, cnn_params, mlp_params):
#         super().__init__()
#         self.input_shape = cnn_params['input_shape']

#         self.prefix_length = 10
#         self.cnn_number = 3

#         self.cnns = nn.ModuleList([CNN(cnn_params) for _ in range(self.cnn_number)])

#         mlp_params['input_size'] = self.prefix_length + self.cnn_number * get_output_size(
#             self.cnns[0], self.input_shape
#         )

#         self.mlp = MLP(mlp_params)

#     def forward(self, input):
#         # Split the input
#         prefix = input[:, : self.prefix_length]

#         c, h, w = self.input_shape
#         cnn_input = input[:, self.prefix_length :].view(-1, self.cnn_number * c, h, w)

#         # Forward pass through CNNs
#         cnn_outputs = []
#         for i in range(self.cnn_number):
#             cnn_output = self.cnns[i](cnn_input[:, i : i + 1, :, :]).view(input.size(0), -1)
#             cnn_outputs.append(cnn_output)

#         # Concatenate the outputs
#         combined = torch.cat([prefix] + cnn_outputs, dim=1)

#         # Forward pass through MLP
#         output = self.mlp(combined)
#         return output


# def triple_cnn_and_mlp_example():
#     cnn_params = {
#         'input_shape': [1, 13, 13],
#         'layers': [
#             {'type': 'conv', 'kernel_size': 3, 'stride': 2, 'in_channels': 1, 'out_channels': 32, 'activation': 'relu'},
#             {
#                 'type': 'conv',
#                 'kernel_size': 3,
#                 'stride': 1,
#                 'in_channels': 32,
#                 'out_channels': 64,
#                 'activation': 'relu',
#             },
#         ],
#     }

#     mlp_params = {'hidden_units': [1024, 1024, 512], 'activations': ['elu', 'elu', 'elu']}

#     return TripleCnnAndMlp(cnn_params, mlp_params)



# class CrazyNet(RnnBase):
#     def __init__(self, rnn_class, rnn_params, cnn_params, mlp_params):
#         super().__init__(rnn_params)

#         self.input_shape = cnn_params['input_shape']

#         self.prefix_length = 10
#         self.cnn_number = 3

#         self.cnns = nn.ModuleList([CNN(cnn_params) for _ in range(self.cnn_number)])

#         rnn_params['input_size'] = self.prefix_length + self.cnn_number * get_output_size(
#             self.cnns[0], self.input_shape
#         )

#         self.rnn = rnn_class(rnn_params)

#         mlp_params['input_size'] = self.rnn.hidden_size + self.rnn.input_size

#         self.mlp = MLP(mlp_params)

#     def forward(self, states, terminated, rnn_inputs):
#         # Split the input
#         prefix = states[:, : self.prefix_length]

#         c, h, w = self.input_shape
#         cnn_input = states[:, self.prefix_length :].view(-1, self.cnn_number * c, h, w)

#         # Forward pass through CNNs
#         cnn_outputs = []
#         for i in range(self.cnn_number):
#             cnn_output = self.cnns[i](cnn_input[:, i : i + 1, :, :]).view(states.size(0), -1)
#             cnn_outputs.append(cnn_output)

#         # Concatenate the outputs of prefix and CNNs
#         cnn_combined = torch.cat([prefix] + cnn_outputs, dim=1)

#         # Forward pass through RNN
#         rnn_output, output_dict = self.rnn(cnn_combined, terminated, rnn_inputs)

#         # Concatenate the outputs of RNN, CNNs and prefix
#         combined = torch.cat([cnn_combined, rnn_output], dim=1)

#         # Forward pass through MLP
#         output = self.mlp(combined)

#         return output, output_dict


# def crazy_net_example():
#     rnn_params = {
#         'num_envs': 2048,
#         'num_layers': 1,
#         'hidden_size': 512 + 256,
#         'sequence_length': 128,
#     }

#     cnn_params = {
#         'input_shape': [1, 13, 13],
#         'layers': [
#             {'type': 'conv', 'kernel_size': 3, 'stride': 2, 'in_channels': 1, 'out_channels': 32, 'activation': 'relu'},
#             {
#                 'type': 'conv',
#                 'kernel_size': 3,
#                 'stride': 1,
#                 'in_channels': 32,
#                 'out_channels': 64,
#                 'activation': 'relu',
#             },
#         ],
#     }

#     mlp_params = {'hidden_units': [1024, 1024, 512], 'activations': ['elu', 'elu', 'elu']}

#     return CrazyNet(LSTM, rnn_params, cnn_params, mlp_params)
