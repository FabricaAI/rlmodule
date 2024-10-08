from rlmodule.source.rlmodel_cfg import BaseRLCfg, RLModelCfg, SharedRLModelCfg
from rlmodule.source.utils import get_output_size

#TODO this, or inside RLModel class that have a config passed
def build_model(cfg: BaseRLCfg):

    #1) network = create # for now it is created, todo from config
    
    #2) get output size - what type?  ..what is Model init doing with action/observation space

    net = cfg.network.module(cfg.network)
    
    network_output_size = get_output_size(net, cfg.network.input_size)

    def build_output_layer(layer_cfg):
        return layer_cfg.class_type( device = cfg.device, input_size = network_output_size, cfg = layer_cfg)
    
    if isinstance(cfg, RLModelCfg):
        rl_model = cfg.class_type( device = cfg.device,
                        network = net,
                        output_layer = build_output_layer( cfg.output_layer)
                     )
    elif isinstance(cfg, SharedRLModelCfg):
        rl_model = cfg.class_type( device = cfg.device,
                        network = net,
                        policy_output_layer = build_output_layer( cfg.policy_output_layer ),
                        value_output_layer = build_output_layer( cfg.value_output_layer ),
                     )
    else:
        raise TypeError(
                f" Received unsupported class: '{type(cfg)}'."
            )
    return rl_model