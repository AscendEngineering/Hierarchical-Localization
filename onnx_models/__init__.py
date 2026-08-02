"""
ONNX Models for Accelerated Visual Localization.

Provides ONNX Runtime / TensorRT accelerated versions of:
- SuperPoint: Local feature detection and description
- LightGlue: Feature matching
- MegaLoc: Global descriptor extraction

Quick start:
    from onnx_models import SuperPointONNX, LightGlueONNX, MegaLocONNX
    
    # Extract features
    sp = SuperPointONNX()
    keypoints, scores, descriptors = sp.extract(grayscale_image)
    
    # Match features
    lg = LightGlueONNX()
    matches, scores = lg.match(kp0, kp1, desc0, desc1)
    
    # Global descriptors
    ml = MegaLocONNX()
    descriptor = ml.extract(rgb_image)
"""

from .superpoint_onnx.model import SuperPointONNX
from .lightglue_onnx.model import LightGlueONNX
from .megaloc_onnx.model import MegaLocONNX

__all__ = ["SuperPointONNX", "LightGlueONNX", "MegaLocONNX"]
