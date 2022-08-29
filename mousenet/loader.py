from doctest import REPORTING_FLAGS
import os
from .cmouse.mousenet_complete_pool import MouseNetCompletePool
import torch.nn as nn
import pathlib
from .cmouse.anatomy import gen_anatomy
from .mouse_cnn.architecture import Architecture
from .cmouse import network
import pathlib
import os

def generate_net(forced=False):
    """
    Generate and return cmouse.Network object. Caches into file from
    subfolder data_files

    param forced (boolean): default False
        Ignores cache and generates and saves a new network object if True
    """
    root = pathlib.Path(__file__).parent.resolve()
    cached = os.path.join(root, "data_files", f"net_cache.pkl")
    if not forced and os.path.isfile(cached):
        return network.load_network_from_pickle(cached)
    architecture = Architecture()
    anet = gen_anatomy(architecture)
    net = network.Network()
    net.construct_from_anatomy(anet, architecture)
    network.save_network_to_pickle(net, cached)
    return net

def load(pretraining=None):
    """
    Loads a mousenet model. You can define initialization methods for mousenet here.

    params: pretraining
        Determines initialization. Must be one of imagenet (TODO), kaiming, or None
    """
    if pretraining not in (None, "imagenet", "kaiming"):
        raise ValueError("Pretraining must be one of imagenet (TODO), kaiming, or None")
    
    #path to this file
    path = pathlib.Path(__file__).parent.resolve()

    # Is this imagenet pretrained?
    # with open(os.path.join(path, "example", "network_complete_updated_number(3,64,64).pkl"), "rb") as file:
    #     net = pickle.load(file)
    #     pdb.set_trace()

    net = generate_net()
    model = MouseNetCompletePool(net)
    

    if pretraining == "kaiming" or None:
        def _kaiming_init_ (m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
        model.apply(_kaiming_init_)

    return model