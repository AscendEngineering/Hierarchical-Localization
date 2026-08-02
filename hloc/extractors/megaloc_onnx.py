"""
MegaLoc ONNX extractor for hloc.

TensorRT-accelerated global descriptor extraction for image retrieval.
"""

import sys
from pathlib import Path

# Add onnx_models to path
_onnx_models_path = Path(__file__).parent.parent.parent
if str(_onnx_models_path) not in sys.path:
    sys.path.insert(0, str(_onnx_models_path))

from ..utils.base_model import BaseModel
from onnx_models.hloc_wrappers import MegaLocONNXWrapper as _Wrapper


class MegaLocONNX(BaseModel):
    """hloc-compatible MegaLoc ONNX wrapper."""

    default_conf = _Wrapper.default_conf
    required_inputs = _Wrapper.required_inputs

    def _init(self, conf):
        self._wrapper = _Wrapper(conf)

    def _forward(self, data):
        return self._wrapper(data)
