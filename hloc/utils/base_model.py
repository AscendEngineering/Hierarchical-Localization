import inspect
import sys
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Dict, Optional

import torch
from torch import nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    default_conf = {}
    required_inputs = []

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.conf = conf = {**self.default_conf, **conf}
        self.required_inputs = copy(self.required_inputs)
        self._init(conf)
        sys.stdout.flush()

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_inputs:
            assert key in data, "Missing key {} in data".format(key)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError


def dynamic_load(root, model):
    module_path = f"{root.__name__}.{model}"
    module = __import__(module_path, fromlist=[""])
    classes = inspect.getmembers(module, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == module_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseModel)]
    assert len(classes) == 1, classes
    return classes[0][1]
    # return getattr(module, 'Model')


def load_model(module, conf: Dict, device: Optional[str] = None):
    """
    Load a model from a module (extractors or matchers).
    
    Args:
        module: The module to load from (e.g., hloc.extractors, hloc.matchers)
        conf: Configuration dict with 'model' key containing model config
        device: Device to load model on ("cuda" or "cpu"). Auto-detects if None.
    
    Returns:
        Loaded model in eval mode on the specified device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(module, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)
    return model
